"""Minimal single-device ProtoIENet training on ImageFolder or synthetic data."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from protoienet import ProtoIENet
from protoienet.losses import prototype_loss


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="contains train/ and val/ class directories")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--pretrained", type=Path, help="local ImageNet CMT-S checkpoint")
    parser.add_argument("--from-scratch", action="store_true", help="explicitly allow random CMT initialization")
    parser.add_argument("--resume", type=Path, help="resume last.pt; use the original training options")
    parser.add_argument("--eval-only", type=Path, help="evaluate a release checkpoint on the validation split")
    parser.add_argument("--smoke-test", action="store_true", help="use 4 synthetic training/validation images")
    parser.add_argument("--output", type=Path, default=Path("runs/protoienet"))
    parser.add_argument("--num-classes", type=int, default=None, help="inferred from ImageFolder, or 5 for smoke test")
    parser.add_argument("--prototypes-per-class", type=int, default=10)
    parser.add_argument("--prototype-dim", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--activation-scale", type=float, default=1.0,
                        help="multiplier applied after per-class softmax (default: 1)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warm-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--warm-lr", type=float, default=3e-4)
    parser.add_argument("--joint-new-lr", type=float, default=3e-5)
    parser.add_argument("--joint-backbone-lr", type=float, default=3e-4)
    parser.add_argument("--lr-step", type=int, default=5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cluster-weight", type=float, default=0.8)
    parser.add_argument("--separation-weight", type=float, default=0.08)
    parser.add_argument("--balanced-sampling", action="store_true", help="class-balanced sampling with replacement")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="CPU computation threads")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "lr_step", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.warm_epochs < 0:
        parser.error("workers and warm epochs must be nonnegative")
    for name in ("warm_lr", "joint_new_lr", "joint_backbone_lr", "weight_decay",
                 "cluster_weight", "separation_weight", "momentum"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.smoke_test and args.data_root:
        parser.error("choose synthetic --smoke-test or --data-root")
    if args.resume and args.eval_only:
        parser.error("choose --resume or --eval-only")
    if (args.resume or args.eval_only) and args.pretrained:
        parser.error("a saved release checkpoint already contains its backbone weights")
    if not args.smoke_test and not args.data_root:
        parser.error("provide --data-root, or use --smoke-test")
    if not (args.smoke_test or args.pretrained or args.from_scratch or args.resume or args.eval_only):
        parser.error("provide --pretrained CMT_SMALL.pth or explicitly use --from-scratch")
    return args


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    return torch.device(name)


def build_datasets(args, checkpoint=None):
    if checkpoint:
        args.image_size = checkpoint["model_config"]["img_size"]
    normalize = transforms.Normalize(MEAN, STD)
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,
    ])
    if args.smoke_test:
        classes = checkpoint["model_config"]["num_classes"] if checkpoint else (args.num_classes or 5)
        train = datasets.FakeData(size=4, image_size=(3, args.image_size, args.image_size),
                                  num_classes=classes, transform=train_transform, random_offset=0)
        val = datasets.FakeData(size=4, image_size=(3, args.image_size, args.image_size),
                                num_classes=classes, transform=eval_transform, random_offset=1000)
        class_to_idx = {f"class_{i}": i for i in range(classes)}
    else:
        val = datasets.ImageFolder(args.data_root / args.val_split, transform=eval_transform)
        class_to_idx = val.class_to_idx
        train = None if args.eval_only else datasets.ImageFolder(
            args.data_root / args.train_split, transform=train_transform)
        if train is not None and train.class_to_idx != class_to_idx:
            raise ValueError("train/val class folders must have exactly the same names")
    if args.num_classes is not None and args.num_classes != len(class_to_idx):
        raise ValueError("--num-classes disagrees with dataset class folders")
    if checkpoint and checkpoint["class_to_idx"] != class_to_idx:
        raise ValueError("dataset class mapping differs from the saved checkpoint")
    if train is not None and len(train) < 2:
        raise ValueError("training needs at least two images")
    if args.image_size == 32 and args.batch_size < 2 and not args.eval_only:
        raise ValueError("image-size 32 training requires batch-size >=2 for BatchNorm")
    return train, val, class_to_idx


def build_optimizer(model, args):
    backbone_ids = {id(p) for p in model.features.parameters()}
    new_parameters = [p for p in model.parameters() if id(p) not in backbone_ids]
    groups = [{"params": list(model.features.parameters()), "name": "backbone"},
              {"params": new_parameters, "name": "new"}]
    if args.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=args.warm_lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    return torch.optim.Adam(groups, lr=args.warm_lr, weight_decay=args.weight_decay)


def configure_stage(model, optimizer, epoch, args):
    """Freeze both CMT weights and its BN statistics during warm-up.

    Gradients pass through CMT to update the preceding EIB. The optimizer
    includes both EIB and ASSA parameters.
    """
    warm = epoch < args.warm_epochs
    model.train()
    model.features.requires_grad_(not warm)
    if warm:
        model.features.eval()
    decay = 0.1 ** (max(0, epoch - args.warm_epochs) // args.lr_step)
    for group in optimizer.param_groups:
        if warm:
            group["lr"] = args.warm_lr
        else:
            group["lr"] = decay * (args.joint_backbone_lr if group["name"] == "backbone"
                                   else args.joint_new_lr)
    return "warm" if warm else "joint"


def run_epoch(model, loader, device, args, optimizer=None):
    if optimizer is None:
        model.eval()
    totals = {key: 0.0 for key in ("loss", "cross_entropy", "cluster", "separation")}
    confusion = torch.zeros(model.num_classes, model.num_classes, dtype=torch.int64)
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, distances = model(images)
            loss, parts = prototype_loss(logits, distances, labels,
                                         model.prototype_class_identity,
                                         args.cluster_weight, args.separation_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):
                        raise FloatingPointError(f"missing/non-finite gradient: {name}")
                optimizer.step()
            batch = labels.numel()
            totals["loss"] += loss.item() * batch
            for key, value in parts.items():
                totals[key] += value.item() * batch
            indices = (labels * model.num_classes + logits.argmax(1)).cpu()
            confusion += torch.bincount(indices, minlength=model.num_classes ** 2).reshape_as(confusion)
            count += batch
    if not count:
        raise ValueError("empty data loader")
    result = {key: value / count for key, value in totals.items()}
    tp = confusion.diag().float()
    precision = tp / confusion.sum(0).clamp_min(1)
    recall = tp / confusion.sum(1).clamp_min(1)
    weights = confusion.sum(1).float() / count
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    result.update(accuracy=tp.sum().item() / count,
                  precision_weighted=(precision * weights).sum().item(),
                  recall_weighted=(recall * weights).sum().item(),
                  f1_weighted=(f1 * weights).sum().item(), samples=count)
    return result


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint_path = args.resume or args.eval_only
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path else None
    train_set, val_set, class_to_idx = build_datasets(args, checkpoint)
    model = ProtoIENet(**checkpoint["model_config"]) if checkpoint else ProtoIENet(
        num_classes=len(class_to_idx), prototypes_per_class=args.prototypes_per_class,
        prototype_dim=args.prototype_dim, img_size=args.image_size,
        activation_scale=args.activation_scale)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    elif args.pretrained:
        print("pretrained:", json.dumps(model.load_pretrained_backbone(args.pretrained)))
    model.to(device)
    generator = torch.Generator().manual_seed(args.seed)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
    if args.eval_only:
        print(json.dumps({"validation": run_epoch(model, val_loader, device, args)}, sort_keys=True))
        return
    if args.resume:
        # Epoch count/output/device may change, but training semantics must match.
        keys = ("optimizer", "warm_epochs", "warm_lr", "joint_new_lr", "joint_backbone_lr",
                "lr_step", "momentum", "weight_decay", "cluster_weight", "separation_weight",
                "batch_size", "balanced_sampling", "seed", "workers")
        different = [key for key in keys if checkpoint["training_config"][key] != getattr(args, key)]
        if different:
            raise ValueError("resume requires the original training options: " + ", ".join(different))
    sampler = None
    if args.balanced_sampling:
        if not hasattr(train_set, "targets"):
            raise ValueError("balanced sampling requires an ImageFolder dataset")
        targets = torch.tensor(train_set.targets)
        counts = torch.bincount(targets, minlength=len(class_to_idx))
        sampler = WeightedRandomSampler(1.0 / counts[targets], len(targets),
                                       replacement=True, generator=generator)
    # Avoid a singleton batch at 32x32 where the final BN has one spatial unit.
    drop_last = args.image_size == 32 and len(train_set) % args.batch_size == 1
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=args.workers, generator=generator,
                              pin_memory=device.type == "cuda", drop_last=drop_last)
    optimizer = build_optimizer(model, args)
    start_epoch, best_accuracy = 0, -1.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_accuracy"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        generator.set_state(checkpoint["loader_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if start_epoch >= args.epochs:
        raise ValueError("--epochs must be greater than the saved epoch count")
    if not args.resume and any((args.output / name).exists() for name in ("last.pt", "metrics.jsonl")):
        raise ValueError("output already contains a run; choose a new --output or use --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "config.json").write_text(json.dumps(
        dict(training=config, model=model.config, class_to_idx=class_to_idx), indent=2) + "\n")
    print(json.dumps(dict(device=str(device), model=model.config,
                          parameters=sum(p.numel() for p in model.parameters()),
                          train_images=len(train_set), validation_images=len(val_set))))
    for epoch in range(start_epoch, args.epochs):
        stage = configure_stage(model, optimizer, epoch, args)
        train_metrics = run_epoch(model, train_loader, device, args, optimizer)
        val_metrics = run_epoch(model, val_loader, device, args)
        record = dict(epoch=epoch + 1, stage=stage, train=train_metrics, validation=val_metrics,
                      learning_rates={group["name"]: group["lr"] for group in optimizer.param_groups})
        print(json.dumps(record, sort_keys=True), flush=True)
        with (args.output / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        is_best = val_metrics["accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, val_metrics["accuracy"])
        saved = dict(epoch=epoch, best_accuracy=best_accuracy, model_config=model.config,
                     training_config=config, class_to_idx=class_to_idx,
                     model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     torch_rng_state=torch.get_rng_state(), python_rng_state=random.getstate(),
                     loader_rng_state=generator.get_state(),
                     cuda_rng_state=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        torch.save(saved, args.output / "last.pt")
        if is_best:
            torch.save(saved, args.output / "best.pt")


if __name__ == "__main__":
    main()

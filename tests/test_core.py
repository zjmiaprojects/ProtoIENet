import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from protoienet import ProtoIENet
from protoienet.losses import prototype_loss
from train import build_optimizer, configure_stage


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.model = ProtoIENet(num_classes=3, prototypes_per_class=4,
                                prototype_dim=16, img_size=64)

    def test_spatial_distances_and_log_similarity(self):
        features = torch.rand(2, 16, 2, 2)
        actual = self.model._l2_convolution(features)
        prototypes = self.model.prototype_vectors[:, :, 0, 0]
        expected = (features[:, None] - prototypes[None, :, :, None, None]).square().sum(2)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.model.eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            _, distances = self.model.push_forward(images)
            logits, minimum = self.model(images)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(distances.shape), (2, 12, 2, 2))
        torch.testing.assert_close(minimum, distances.flatten(2).min(-1).values)
        torch.testing.assert_close(self.model.distance_2_similarity(minimum),
                                   self.model.distance_2_similarity(distances).flatten(2).max(-1).values)

    def test_loss_class_masks_and_separation_gradient_sign(self):
        distances = torch.tensor([[1., 3., 4., 5.], [7., 8., 2., 6.]], requires_grad=True)
        identity = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        loss, parts = prototype_loss(torch.zeros(2, 2), distances, torch.tensor([0, 1]), identity)
        self.assertEqual(parts["cluster"].item(), 1.5)
        self.assertEqual(parts["separation"].item(), -5.5)
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.)).item() + .8 * 1.5 - .08 * 5.5, places=6)
        loss.backward()
        self.assertGreater(distances.grad[0, 0].item(), 0)  # descent pulls same-class closer
        self.assertLess(distances.grad[0, 2].item(), 0)  # descent pushes other-class away

    def test_training_stage_updates_eib_assa_and_backbone(self):
        args = SimpleNamespace(optimizer="adam", warm_lr=1e-3, momentum=.9, weight_decay=0.,
                               warm_epochs=1, lr_step=5, joint_backbone_lr=1e-3, joint_new_lr=1e-3)
        optimizer = build_optimizer(self.model, args)
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(optimized, {id(p) for p in self.model.parameters()})
        images, targets = torch.randn(2, 3, 64, 64), torch.tensor([0, 2])
        for epoch in (0, 1):
            configure_stage(self.model, optimizer, epoch, args)
            before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
            bn_before = self.model.features.stem_norm1.running_mean.clone()
            logits, distances = self.model(images)
            loss, _ = prototype_loss(logits, distances, targets, self.model.prototype_class_identity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
            parameters = dict(self.model.named_parameters())
            for name in ("fem.down.weight", "sa.value.weight", "prototype_vectors", "add_on_layers.0.weight"):
                self.assertFalse(torch.equal(before[name], parameters[name]), name)
            if epoch == 0:
                for name in parameters:
                    if name.startswith("features."):
                        torch.testing.assert_close(before[name], parameters[name], rtol=0, atol=0)
                torch.testing.assert_close(bn_before, self.model.features.stem_norm1.running_mean, rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(before["features.stem_conv1.weight"],
                                             self.model.features.stem_conv1.weight))

    def test_state_dict_roundtrip_and_pretrained_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save(dict(config=self.model.config, state=self.model.state_dict()), path)
            saved = torch.load(path, weights_only=True)
            restored = ProtoIENet(**saved["config"])
            restored.load_state_dict(saved["state"], strict=True)
            self.model.eval()
            restored.eval()
            images = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                torch.testing.assert_close(self.model(images)[0], restored(images)[0], rtol=0, atol=0)
            state = {"module." + key: value for key, value in self.model.features.state_dict().items()}
            state["module.head.weight"] = torch.rand(1000, 1280)
            torch.save({"model": state}, path)
            report = restored.load_pretrained_backbone(path)
            self.assertEqual(report["ignored"], ["module.head.weight"])
            self.assertEqual(report["mismatched"], [])
            state.pop("module.blocks_d.0.attn.q.weight")
            torch.save({"model": state}, path)
            with self.assertRaisesRegex(ValueError, "missing/incompatible"):
                restored.load_pretrained_backbone(path)


if __name__ == "__main__":
    unittest.main()

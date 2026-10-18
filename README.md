# ProtoIENet

We present **ProtoIENet: An interpretable approach for medical image classification**, published in *Pattern Recognition*, 179 (2026), 113772. This repository provides our PyTorch model and training code.

[Paper](https://doi.org/10.1016/j.patcog.2026.113772)

## Installation

We use Python 3.10+ with PyTorch 2.8.0 and torchvision 0.23.0.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Prepare each dataset split in ImageFolder format:

```text
data/SIPaKMeD/fold_1/
  train/<class_name>/<image>.png
  val/<class_name>/<image>.png
  test/<class_name>/<image>.png
```

Use the same class directory names across splits. We infer the class count automatically. Run each cross-validation fold separately with its own data and output directories. For an existing `train_augmented` directory, add `--train-split train_augmented`.

We apply random resized cropping and horizontal flipping during training, including when using `train_augmented`. Validation and test images use deterministic resizing. All splits use ImageNet normalization.

## Training

Download the [CMT-S ImageNet weights](https://github.com/ggjy/CMT.pytorch/releases/download/release-v1/cmt_small.pth) and supply their local path:

```bash
python train.py --data-root data/SIPaKMeD/fold_1 \
  --pretrained weights/cmt_small.pth \
  --epochs 100 --batch-size 8 --device cuda --output runs/sipakmed_fold1
```

Our default configuration uses 224 x 224 inputs, 10 prototypes per class, 512-dimensional prototypes, SGD, and 5 warm-up epochs. Set paths and training parameters for your experiment; use `python train.py --help` for all options. Use `--device cpu` without CUDA, or `--from-scratch` instead of `--pretrained` to train with random backbone initialization.

We save `best.pt`, selected by validation accuracy, and `last.pt`, which includes the optimizer state. The output directory also contains `config.json` and per-epoch metrics in `metrics.jsonl`.

## Evaluation and resume

Evaluate the selected model on the test split:

```bash
python train.py --data-root data/SIPaKMeD/fold_1 \
  --eval-only runs/sipakmed_fold1/best.pt --val-split test --device cuda
```

Resume training with the same training settings and a larger total epoch count:

```bash
python train.py --data-root data/SIPaKMeD/fold_1 \
  --resume runs/sipakmed_fold1/last.pt \
  --epochs 150 --batch-size 8 --device cuda --output runs/sipakmed_fold1
```

## Quick check

```bash
python train.py --smoke-test --epochs 2 --warm-epochs 1 \
  --batch-size 2 --device cpu --output runs/smoke
python -m unittest discover -s tests -v
```

## Citation

If you use our code, please cite our paper:

```bibtex
@article{wang2026protoienet,
  title = {ProtoIENet: An interpretable approach for medical image classification},
  author = {Wang, Xiaoyan and Yu, Jianhao and Zhang, Wenzhong and Zhu, Yating and
            Xia, Ming and Huang, Xiaojie and Cui, Ying and Guo, Dongyan and Zhang, Xiaoqin},
  journal = {Pattern Recognition},
  volume = {179},
  pages = {113772},
  year = {2026},
  doi = {10.1016/j.patcog.2026.113772}
}
```

## Acknowledgements

Our implementation builds on [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) and [CMT](https://github.com/ggjy/CMT.pytorch). We thank their authors for sharing their work. We credit our contributions to the ProtoIENet authors in `LICENSE`, retain the ProtoPNet copyright notice there, and preserve the CMT attribution in the backbone source.

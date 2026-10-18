"""ProtoIENet: EIB -> CMT-S -> prototype layer -> ASSA."""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .cmt import cmt_small


class EnhancedInvertedBottleneck(nn.Module):
    """7x7 depthwise convolution and spatial calibration, with a residual."""

    def __init__(self, channels=3):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channels, 4 * channels, 1),
            nn.BatchNorm2d(4 * channels),
            nn.LeakyReLU(),
        )
        self.conv1_2 = nn.Sequential(nn.Conv2d(4 * channels, 1, 1), nn.Sigmoid())
        self.down = nn.Conv2d(8 * channels, channels, 1)

    def forward(self, x):
        expanded = self.conv1_1(self.dwconv(x))
        calibrated = expanded * self.conv1_2(expanded)
        return x + self.down(torch.cat((expanded, calibrated), dim=1))


class SelfAttention(nn.Module):
    """Scaled dot-product attention over class-wise prototype vectors.

    Input/output: [batch, classes, prototypes_per_class].
    Attention matrix: [batch, classes, classes].
    """

    def __init__(self, prototypes_per_class):
        super().__init__()
        self.scale = math.sqrt(prototypes_per_class)
        self.query = nn.Linear(prototypes_per_class, prototypes_per_class)
        self.key = nn.Linear(prototypes_per_class, prototypes_per_class)
        self.value = nn.Linear(prototypes_per_class, prototypes_per_class)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attention = ((q @ k.transpose(-2, -1)) / self.scale).softmax(dim=-1)
        return attention @ v


class ProtoIENet(nn.Module):
    def __init__(self, num_classes=5, prototypes_per_class=10, prototype_dim=512,
                 img_size=224, activation_scale=1.0):
        super().__init__()
        if num_classes < 2 or prototypes_per_class < 1 or prototype_dim < 1:
            raise ValueError("need >=2 classes and positive prototype count/dimension")
        if not math.isfinite(activation_scale) or activation_scale <= 0:
            raise ValueError("activation_scale must be finite and positive")
        self.config = dict(num_classes=num_classes, prototypes_per_class=prototypes_per_class,
                           prototype_dim=prototype_dim, img_size=img_size,
                           activation_scale=activation_scale)
        self.num_classes = num_classes
        self.num_prototypes_per_class = prototypes_per_class
        self.num_prototypes = num_classes * prototypes_per_class
        self.prototype_shape = (self.num_prototypes, prototype_dim, 1, 1)
        self.img_size = img_size
        self.epsilon = 1e-4
        self.activation_scale = activation_scale
        self.features = cmt_small(img_size)
        self.fem = EnhancedInvertedBottleneck()
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(512, prototype_dim, 1), nn.ReLU(),
            nn.Conv2d(prototype_dim, prototype_dim, 1), nn.Sigmoid(),
        )
        self.sa = SelfAttention(prototypes_per_class)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))
        self.activation_weight = nn.Parameter(torch.randn(self.num_prototypes))
        # Buffers follow the model's device and are included in checkpoints.
        self.register_buffer("ones", torch.ones(self.prototype_shape))
        class_ids = torch.arange(self.num_prototypes) // prototypes_per_class
        self.register_buffer("prototype_class_identity", F.one_hot(class_ids, num_classes).float())
        for module in self.add_on_layers.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)
        for module in self.sa.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def conv_features(self, x):
        return self.add_on_layers(self.features(self.fem(x)))

    def _l2_convolution(self, features):
        patch_norm = F.conv2d(features.square(), self.ones)
        proto_norm = self.prototype_vectors.square().sum((1, 2, 3)).view(-1, 1, 1)
        inner_product = F.conv2d(features, self.prototype_vectors)
        return F.relu(patch_norm - 2 * inner_product + proto_norm)

    def distance_2_similarity(self, distances):
        return torch.log((distances + 1) / (distances + self.epsilon))

    def forward_SA_module(self, activations):
        weights = self.activation_weight.reshape(self.num_classes, -1).softmax(dim=-1)
        weights = weights.flatten() * self.activation_scale
        weighted = (activations * weights).reshape(
            -1, self.num_classes, self.num_prototypes_per_class)
        return self.sa(weighted).sum(dim=-1)

    def push_forward(self, x):
        """Return latent features and spatial prototype distances."""
        features = self.conv_features(x)
        return features, self._l2_convolution(features)

    def prototype_distances(self, x):
        return self.push_forward(x)[1]

    def forward(self, x):
        distances = self.prototype_distances(x)
        min_distances = distances.flatten(2).min(dim=-1).values
        activations = self.distance_2_similarity(min_distances)
        return self.forward_SA_module(activations), min_distances

    @torch.no_grad()
    def load_pretrained_backbone(self, path):
        """Load compatible local CMT-S weights, reporting every excluded key.

        Accepts a state_dict or a checkpoint with 'model'/'state_dict'. No
        weights are downloaded implicitly; the ImageNet classifier is ignored.
        """
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = checkpoint
        for key in ("model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        target = self.features.state_dict()
        compatible, unexpected, mismatched = {}, [], []
        for key, value in state.items():
            name = key.removeprefix("module.").removeprefix("features.")
            if name not in target:
                unexpected.append(key)
            elif not isinstance(value, torch.Tensor) or value.shape != target[name].shape:
                mismatched.append(key)
            else:
                compatible[name] = value
        # Avoid reporting success for a wrong backbone or a tiny matching subset.
        missing_parameters = sorted(set(dict(self.features.named_parameters())) - compatible.keys())
        if missing_parameters:
            raise ValueError("CMT-S checkpoint is missing/incompatible with backbone parameters: "
                             + ", ".join(missing_parameters[:12])
                             + ". Use matching CMT-S weights and image size (normally 224).")
        missing, _ = self.features.load_state_dict(compatible, strict=False)
        return dict(loaded=len(compatible), missing=list(missing),
                    ignored=unexpected, mismatched=mismatched)

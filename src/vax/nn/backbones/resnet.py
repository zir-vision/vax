from typing import Callable
import jax.numpy as jnp
from functools import partial
from vax.nn.convolution import CNA, ResidualBlock, BottleneckResidualBlock
from flax import nnx
import logging

log = logging.getLogger(__name__)


class ResNet(nnx.Module):
    """
    ResNet backbone
    """

    def __init__(self, stage_sizes, block_cls, num_filters=64, channels=3, activation=nnx.relu, *, rngs):
        self.block_cls = block_cls
        self.conv = nnx.Conv(
            in_features=channels, out_features=num_filters, kernel_size=(7, 7), strides=(2, 2), rngs=rngs
        )
        self.bn = nnx.BatchNorm(num_filters, rngs=rngs)
        self.activation = activation

        blocks = []
        for i, block_size in enumerate(stage_sizes):
            for j in range(block_size):
                stride = (2,2) if i > 0 and j == 0 else (1,1)
                log.debug(f"Block {i+1}, {j+1} stride: {stride}")
                if i == 0 and j == 0:
                    in_features = num_filters
                elif j == 0:
                    in_features = num_filters * 2 ** (i - 1)
                    in_features = in_features * block_cls.expansion
                else:
                    in_features = num_filters * 2**i
                    in_features = in_features * block_cls.expansion
                
                blocks.append(
                    block_cls(
                        in_features=in_features,
                        out_features=num_filters * 2**i,
                        stride=stride,
                        activation=activation,
                        rngs=rngs,
                    )
                )
        self.layers = nnx.Sequential(*blocks)

    def __call__(self, x):
        log.debug(f"Input shape: {x.shape}")
        x = self.conv(x)
        log.debug(f"Conv shape: {x.shape}")
        x = self.bn(x)
        log.debug(f"BN shape: {x.shape}")
        x = self.activation(x)
        log.debug(f"Activation shape: {x.shape}")
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        log.debug(f"Max pool shape: {x.shape}")
        x = self.layers(x)
        log.debug(f"Layers shape: {x.shape}")
        return x


class ResNetClassification(nnx.Module):
    """
    ResNet backbone for classification
    """

    def __init__(self, backbone, num_classes, *, rngs):
        self.backbone = backbone
        self.num_classes = num_classes
        self.fc = nnx.Linear(backbone.block_cls.expansion * 512, num_classes, rngs=rngs)

    def __call__(self, x):
        x = self.backbone(x)
        x = jnp.mean(x, axis=(1, 2))
        x = self.fc(x)
        return x


ResNet18 = partial(ResNet, [2, 2, 2, 2], ResidualBlock)
ResNet34 = partial(ResNet, [3, 4, 6, 3], ResidualBlock)
ResNet50 = partial(ResNet, [3, 4, 6, 3], BottleneckResidualBlock)
ResNet101 = partial(ResNet, [3, 4, 23, 3], BottleneckResidualBlock)
ResNet152 = partial(ResNet, [3, 8, 36, 3], BottleneckResidualBlock)
ResNet200 = partial(ResNet, [3, 24, 36, 3], BottleneckResidualBlock)

if __name__ == "__main__":
    from vax.nn.backbones import resnet
    from flax import nnx
    from jax import numpy
    import vax.console
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.NOTSET,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(console=vax.console.console, rich_tracebacks=True)],
    )
    logging.getLogger("jax").setLevel(logging.WARNING)
    rngs = nnx.Rngs(0)
    bb = resnet.ResNet200(rngs=rngs)
    m = ResNetClassification(bb, 10, rngs=rngs)
    y = m(numpy.zeros((2, 224, 224, 3)))
    log.debug(f"Output shape: {y.shape}")


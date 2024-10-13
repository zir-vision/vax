from jax import Array
from typing import Callable, Any
from flax import nnx
import logging

log = logging.getLogger(__name__)


# From ultralytics
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class CNA(nnx.Module):
    """
    Convolution -> BatchNorm -> Activation
    """

    def __init__(
        self, in_features, out_features, kernel_size, stride=1, padding=None, dilation=1, activation=nnx.relu, *, rngs
    ):
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=stride,
            padding=autopad(kernel_size, padding, dilation),
            input_dilation=dilation,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(out_features, rngs=rngs)
        self.activation = activation

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nnx.Module):
    """
    Residual block with two CNA blocks
    """

    expansion = 1

    def __init__(self, in_features, out_features, stride=(1,1), activation=nnx.relu, *, rngs):
        self.conv1 = CNA(in_features, out_features, (3, 3), stride, (1, 1), activation=activation, rngs=rngs)
        self.conv2 = CNA(out_features, out_features, (3, 3), (1, 1), (1, 1), activation=None, rngs=rngs)
        log.debug(f"In features: {self.conv1.conv.in_features}")
        log.debug(f"Out features: {self.conv1.conv.out_features}")
        if in_features != out_features * self.expansion:
            self.proj = CNA(
                in_features, out_features * self.expansion, (1, 1), stride, (0, 0), activation=None, rngs=rngs
            )
        else:
            self.proj = None
        self.activation = activation

    def __call__(self, x):
        log.debug(f"In features: {self.conv1.conv.in_features}")

        log.debug(f"X shape: {x.shape}")
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.proj:
            residual = self.proj(residual)
        x = self.activation(x + residual)
        log.debug(f"ResidualBlock Out shape: {x.shape}")
        return x


class BottleneckResidualBlock(nnx.Module):
    """
    Bottleneck residual block with three CNA blocks
    """

    expansion = 4

    def __init__(self, in_features, out_features, stride=(1,1), activation=nnx.relu, *, rngs):
        log.debug(f"In features: {in_features}")
        log.debug(f"Out features: {out_features}")
        self.conv1 = CNA(in_features, out_features, (1,1), (1,1), (0,0), activation=activation, rngs=rngs)
        self.conv2 = CNA(out_features, out_features, (3,3), stride, (1,1), activation=activation, rngs=rngs)
        self.conv3 = CNA(out_features, out_features * self.expansion, (1,1), (1,1), (0,0), activation=None, rngs=rngs)
        if in_features != out_features * self.expansion:
            self.proj = CNA(in_features, out_features * self.expansion, (1,1), stride, (0,0), activation=None, rngs=rngs)
        else:
            self.proj = None
        self.activation = activation

    def __call__(self, x):
        log.debug(f"In features: {self.conv1.conv.in_features}")
        log.debug(f"X shape: {x.shape}")
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.proj:
            residual = self.proj(residual)
        x = self.activation(x + residual)
        log.debug(f"BottleneckResidualBlock Out shape: {x.shape}")
        return x

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

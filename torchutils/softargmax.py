import torch
import torch.nn.functional as F

from .common_types import Tensor


__all__ = ["softargmax1d", "softargmax2d", "softargmax3d"]


# Helper Functions

def _softargmaxnd(x: Tensor, weight: Tensor, beta: float) -> Tensor:
    x = x.view(*x.shape[:2], -1).mul(beta).softmax(2).view_as(x)
    x = getattr(F, f"conv{x.ndim - 2}d")(x, weight)
    return x.reshape(-1, x.ndim - 2)


# Functional

def softargmax1d(x: Tensor, beta: float = 1e+3) -> Tensor:
    b, c, w = x.shape

    weight = torch.arange(0, w, dtype=torch.float32)[None, :].unsqueeze(1)

    r = _softargmaxnd(x, weight, beta)

    return r


def softargmax2d(x: Tensor, beta: float = 1e+3) -> Tensor:
    b, c, h, w = x.shape

    weight = torch.stack([
        torch.arange(0, h, dtype=torch.float32)[:, None].expand(-1, w),
        torch.arange(0, w, dtype=torch.float32)[None, :].expand(h, -1),
    ]).unsqueeze(1)

    r = _softargmaxnd(x, weight, beta)

    return r


def softargmax3d(x: Tensor, beta: float = 1e+3) -> Tensor:
    b, c, d, h, w = x.shape

    weight = torch.stack([
        torch.arange(0, d, dtype=torch.float32)[:, None, None].expand(-1, h, w),
        torch.arange(0, h, dtype=torch.float32)[None, :, None].expand(d, -1, w),
        torch.arange(0, w, dtype=torch.float32)[None, None, :].expand(d, h, -1),
    ]).unsqueeze(1)

    r = _softargmaxnd(x, weight, beta)

    return r

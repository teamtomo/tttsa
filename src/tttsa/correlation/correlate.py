"""DFT based cross-correlation between images."""

import einops
import torch


def correlate_2d(
    a: torch.Tensor, b: torch.Tensor, normalize: bool = False
) -> torch.Tensor:
    """Calculate the 2D cross correlation between images of the same size.

    The position of the maximum relative to the center of the image gives a shift.
    This is the shift that when applied to `b` best aligns it to `a`.
    """
    if normalize is True:
        h, w = a.shape[-2:]
        a_norm = einops.reduce(a**2, "... h w -> ... 1 1", reduction="mean") ** 0.5
        b_norm = einops.reduce(b**2, "... h w -> ... 1 1", reduction="mean") ** 0.5
        a = a / a_norm
        b = b / b_norm
    fta = torch.fft.rfftn(a, dim=(-2, -1))
    ftb = torch.fft.rfftn(b, dim=(-2, -1))
    result = fta * torch.conj(ftb)
    # AreTomo using some like this (filtered FFT-based approach):
    # result = result / torch.sqrt(result.abs() + .0001)
    # result = bfactor_dft(result, 300, (result.shape[-2], ) * 2, 1, True)
    result = torch.fft.irfftn(result, dim=(-2, -1), s=a.shape[-2:])
    result = torch.real(torch.fft.ifftshift(result, dim=(-2, -1)))
    if normalize is True:
        result = result / (h * w)
    return result

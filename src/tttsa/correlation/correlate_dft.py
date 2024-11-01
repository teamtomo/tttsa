import torch

from tttsa.utils import ifftshift_2d


def correlate_dft_2d(
    a: torch.Tensor,
    b: torch.Tensor,
    rfft: bool,
    fftshifted: bool
) -> torch.Tensor:
    """Correlate discrete Fourier transforms of images."""
    result = a * torch.conj(b)
    # AreTomo using some like this (filtered FFT-based approach):
    # result = result / torch.sqrt(result.abs() + .0001)
    # result = bfactor_dft(result, 300, (result.shape[-2], ) * 2, 1, True)
    if fftshifted is True:
        result = ifftshift_2d(result, rfft=rfft)
    if rfft is True:
        result = torch.fft.irfftn(result, dim=(-2, -1))
    else:
        result = torch.fft.ifftn(result, dim=(-2, -1))
    return torch.real(torch.fft.ifftshift(result, dim=(-2, -1)))

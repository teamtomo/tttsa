import torch
import torch.nn.functional as F
from typing import Tuple, Sequence

from .transformations import T_2d, R_2d
from tttsa.affine import affine_transform_2d


def rfft_shape(input_shape: Sequence[int]) -> Tuple[int]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def dft_center(
    image_shape: Tuple[int, ...],
    rfft: bool,
    fftshifted: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return the position of the DFT center for a given input shape."""
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape).float()
    if rfft is True:
        image_shape = torch.tensor(rfft_shape(image_shape))
    if fftshifted is True:
        fft_center = torch.divide(image_shape, 2, rounding_mode='floor')
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()


def ifftshift_2d(input: torch.Tensor, rfft: bool):
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-2,))
    return output


def homogenise_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """3D coordinates to 4D homogenous coordinates with ones in the last column.

    Parameters
    ----------
    coords: torch.Tensor
        `(..., 3)` array of 3D coordinates

    Returns
    -------
    output: torch.Tensor
        `(..., 4)` array of homogenous coordinates
    """
    return F.pad(torch.as_tensor(coords), pad=(0, 1), mode='constant', value=1)


def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

    These coordinates should be used with `align_corners=True` in
    `torch.nn.functional.grid_sample`.


    Parameters
    ----------
    array_coordinates: torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape: Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    dtype, device = array_coordinates.dtype, array_coordinates.device
    array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
    grid_sample_coordinates = (array_coordinates / (0.5 * array_shape - 0.5)) - 1
    grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
    return grid_sample_coordinates


def stretch_image(image, stretch, tilt_axis_angle):
    """Stretch an image along the tilt axis."""
    image_center = dft_center(image.shape, rfft=False, fftshifted=True)
    # construct matrix
    s0 = T_2d(-image_center)
    r_forward = R_2d(tilt_axis_angle, yx=True)
    r_backward = torch.linalg.inv(r_forward)
    m_stretch = torch.eye(3)
    m_stretch[1, 1] = stretch  # this is a shear matrix
    s1 = T_2d(image_center)
    m_affine = s1 @ r_forward @ m_stretch @ r_backward @ s0
    # transform image
    stretched = affine_transform_2d(
        image,
        m_affine,
    )
    return stretched

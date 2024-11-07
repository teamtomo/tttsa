"""Real-space projection of tilt-series images and tomograms."""

from typing import Tuple

import einops
import torch
import torch.nn.functional as F
from torch_grid_utils import coordinate_grid
from torch_image_lerp import insert_into_image_2d

from tttsa.affine import affine_transform_2d
from tttsa.transformations import R_2d, Ry, Rz, T, T_2d
from tttsa.utils import dft_center, homogenise_coordinates


def common_lines_projection(
    images: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
    # this might as well takes shifts
) -> torch.Tensor:
    """Predict a projection from an intermediate reconstruction.

    For now only assumes to project with a single matrix, but should also work for
    sets of matrices.
    """
    device = images.device
    image_dimensions = images.shape[-2:]

    # TODO pad image if not square

    image_center = dft_center(image_dimensions, rfft=False, fftshifted=True)

    # time for real space projection
    s0 = T_2d(-image_center)
    r0 = R_2d(tilt_axis_angles, yx=True)
    s1 = T_2d(image_center)
    # invert because the tilt axis angle is forward in the sample projection model
    M = torch.linalg.inv(s1 @ r0 @ s0).to(device)

    rotated = affine_transform_2d(
        images,
        M,
    )
    projections = rotated.mean(axis=-1).squeeze()
    return projections


def tomogram_reprojection(
    tomogram: torch.Tensor,
    tilt_image_dimensions: Tuple[int, int],
    tilt_angles: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict a projection from an intermediate reconstruction.

    For now only assumes to project with a single matrix, but should also work for
    sets of matrices.
    """
    device = tomogram.device
    tomogram_dimensions = tomogram.shape
    tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
    transform_shape = (tomogram_dimensions[0], *tilt_image_dimensions)
    transform_center = dft_center(transform_shape, rfft=False, fftshifted=True)

    # time for real space projection
    s0 = T(-transform_center)
    r0 = Ry(tilt_angles, zyx=True)
    r1 = Rz(tilt_axis_angles, zyx=True)
    s1 = T(F.pad(-shifts, pad=(1, 0), value=0))
    s2 = T(tomogram_center)
    M = s2 @ s1 @ r1 @ r0 @ s0
    Mproj = M[:, 1:3, :]
    Mproj = einops.rearrange(Mproj, "... i j -> ... 1 1 i j").to(device)

    grid = homogenise_coordinates(coordinate_grid(tomogram_dimensions, device=device))
    grid = einops.rearrange(grid, "d h w coords -> d h w coords 1")
    grid = Mproj @ grid
    grid = einops.rearrange(grid, "... d h w coords 1 -> ... d h w coords")
    projection, weights = insert_into_image_2d(
        tomogram.view(-1),  # flatten
        grid.view(-1, 2),
        torch.zeros(tilt_image_dimensions, device=device),
    )
    return projection, weights

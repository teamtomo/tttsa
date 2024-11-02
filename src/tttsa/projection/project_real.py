import torch
import torch.nn.functional as F
import einops
from torch_grid_utils import coordinate_grid

from tttsa.transformations import T_2d, R_2d, T, Ry, Rz
from tttsa.utils import dft_center, homogenise_coordinates, array_to_grid_sample


def common_lines_projection(
        image,
        angles,
        # this should also take shifts
):
    """Predict a projection from an intermediate reconstruction.

    For now only assumes to project with a single matrix, but should also work for
    sets of matrices.
    """
    # square image for possible fourier space extraction
    device = image.device
    image_dimensions = image.shape[-2:]
    image_center = dft_center(image_dimensions, rfft=False, fftshifted=True)

    # time for real space projection
    s0 = T_2d(-image_center)
    r0 = R_2d(angles, yx=True)
    s1 = T_2d(image_center)
    M = einops.rearrange(
        torch.linalg.inv(s1 @ r0 @ s0), 
        "... i j -> ... 1 1 i j"
    ).to(device)

    grid = homogenise_coordinates(coordinate_grid(image_dimensions, device=device))
    grid = einops.rearrange(grid, "h w coords -> 1 h w coords 1")
    grid = M @ grid
    grid = einops.rearrange(grid, "... h w coords 1 -> ... h w coords")[
        ..., :2
    ].contiguous()
    grid_sample_coordinates = array_to_grid_sample(grid, image_dimensions)
    rotated = torch.squeeze(
        F.grid_sample(
            einops.rearrange(image, "n h w -> n 1 h w"),
            grid_sample_coordinates,
            align_corners=True,
            mode="bicubic",
        )
    )
    projections = rotated.mean(axis=-1).squeeze()
    return projections


def tomogram_reprojection(
    tomogram,
    tilt_image_dimensions,
    tilt_angles,
    tilt_axis_angles,
    shifts,
):
    """Predict a projection from an intermediate reconstruction.

    For now only assumes to project with a single matrix, but should also work for
    sets of matrices.
    """
    device = tomogram.device
    tomogram_dimensions = tomogram.shape
    tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
    transform_shape = (tomogram_dimensions[0],) + tuple(tilt_image_dimensions)
    transform_center = dft_center(transform_shape, rfft=False, fftshifted=True)

    # time for real space projection
    s0 = T(-tomogram_center)
    r0 = Ry(tilt_angles, zyx=True)
    r1 = Rz(tilt_axis_angles, zyx=True)
    s1 = T(F.pad(shifts, pad=(1, 0), value=0))
    s2 = (transform_center)
    M = einops.rearrange(
        s2 @ s1 @ r1 @ r0 @ s0, 
        "... i j -> ... 1 1 i j"
    ).to(device)

    grid = homogenise_coordinates(coordinate_grid(tomogram_dimensions, device=device))
    grid = einops.rearrange(grid, "d h w coords -> d h w coords 1")
    torch.matmul(M, grid, out=grid)
    grid = einops.rearrange(grid, "... d h w coords 1 -> ... d h w coords")[
        ..., :3
    ].contiguous()
    grid_sample_coordinates = array_to_grid_sample(grid, tomogram_dimensions)
    rotated = torch.squeeze(
        F.grid_sample(
            einops.rearrange(tomogram, "d h w -> 1 1 d h w"),
            einops.rearrange(grid_sample_coordinates, "d h w coords -> 1 d h w coords"),
            align_corners=True,
            mode="bilinear",
        )
    )
    projection = rotated.mean(axis=-3)
    weights = (rotated != 0).sum(axis=-3)
    weights = weights / weights.max()
    return projection, weights

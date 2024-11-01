import torch
import einops
from typing import Sequence
import torch.nn.functional as F
from torch_grid_utils import coordinate_grid

from tttsa.utils import homogenise_coordinates, array_to_grid_sample


def affine_transform_2d(
    images: torch.Tensor,  # shape: '... h w'
    affine_matrices: torch.Tensor,  # shape: '... 3 3'
    out_shape: Sequence[int] | None = None,
    interpolation: str = "bicubic",
):
    """Affine transform 1 or a batch of images."""
    if out_shape is None:
        out_shape = images.shape[-2:]
    device = images.device
    grid = homogenise_coordinates(coordinate_grid(out_shape, device=device))
    grid = einops.rearrange(grid, "h w coords -> 1 h w coords 1")
    M = einops.rearrange(affine_matrices, "... i j -> ... 1 1 i j").to(device)
    grid = M @ grid
    grid = einops.rearrange(grid, "... h w coords 1 -> ... h w coords")[
        ..., :2
    ].contiguous()
    grid_sample_coordinates = array_to_grid_sample(grid, images.shape[-2:])
    if images.dim() == 2:  # needed for grid sample
        images = einops.repeat(images, "h w -> n h w", n=M.shape[0])
    transformed = einops.rearrange(
        F.grid_sample(
            einops.rearrange(images, "... h w -> ... 1 h w"),
            grid_sample_coordinates,
            align_corners=True,
            mode=interpolation,
        ),
        "... 1 h w -> ... h w",
    ).squeeze()  # remove starter dimensions in case we got one image
    return transformed


# TODO write some functions like
# def affine_transform_3d(
#       volumes: torch.Tensor,  # shape: 'n d h w'
#       affine_matrices: torch.Tensor,  # shape: 'n 4 4'
#       interpolation: str = 'bilinear',  # is actually trilinear in grid_sample
# ):
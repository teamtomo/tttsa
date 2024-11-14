"""2D and 3D affine transform for torch with precise control of matrices."""

from typing import Sequence

import einops
import torch
import torch.nn.functional as F
from torch_grid_utils import coordinate_grid

from tttsa.utils import array_to_grid_sample, homogenise_coordinates


def affine_transform_2d(
    images: torch.Tensor,  # shape: '... h w'
    affine_matrices: torch.Tensor,  # shape: '... 3 3'
    out_shape: Sequence[int] | None = None,
    interpolation: str = "bicubic",
) -> torch.Tensor:
    """Affine transform 1 or a batch of images."""
    if out_shape is None:
        out_shape = images.shape[-2:]
    device = images.device
    grid = homogenise_coordinates(coordinate_grid(out_shape, device=device))
    grid = einops.rearrange(grid, "h w coords -> 1 h w coords 1")
    M = einops.rearrange(
        torch.linalg.inv(affine_matrices),  # invert so that each grid cell points
        "... i j -> ... 1 1 i j",  # to where it needs to get data from
    ).to(device)
    grid = M @ grid
    grid = einops.rearrange(grid, "... h w coords 1 -> ... h w coords")[
        ..., :2
    ].contiguous()
    grid_sample_coordinates = array_to_grid_sample(grid, images.shape[-2:])
    samples = (
        einops.repeat(  # needed for grid sample
            images, "h w -> n h w", n=M.shape[0]
        )
        if images.dim() == 2
        else images
    )
    if samples.shape[0] != grid_sample_coordinates.shape[0]:
        raise ValueError(
            "Provide either an equal batch of images and matrices or "
            "multiple matrices for a single image."
        )
    transformed = einops.rearrange(
        F.grid_sample(
            einops.rearrange(samples, "... h w -> ... 1 h w"),
            grid_sample_coordinates,
            align_corners=True,
            mode=interpolation,
        ),
        "... 1 h w -> ... h w",  # remove channel
    )
    if images.dim() == 2:  # remove starter dimensions in case we got one image
        transformed = transformed.squeeze()
    return transformed


def affine_transform_3d(
    images: torch.Tensor,  # shape: '... h w'
    affine_matrices: torch.Tensor,  # shape: '... 3 3'
    out_shape: Sequence[int] | None = None,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """Affine transform 1 or a batch of images."""
    if out_shape is None:
        out_shape = images.shape[-3:]
    device = images.device
    grid = homogenise_coordinates(coordinate_grid(out_shape, device=device))
    grid = einops.rearrange(grid, "d h w coords -> 1 d h w coords 1")
    M = einops.rearrange(
        torch.linalg.inv(affine_matrices),  # invert so that each grid cell points
        "... i j -> ... 1 1 1 i j",  # to where it needs to get data from
    ).to(device)
    grid = M @ grid
    grid = einops.rearrange(grid, "... d h w coords 1 -> ... d h w coords")[
        ..., :3
    ].contiguous()
    grid_sample_coordinates = array_to_grid_sample(grid, images.shape[-3:])
    samples = (
        einops.repeat(  # needed for grid sample
            images, "d h w -> n d h w", n=M.shape[0]
        )
        if images.dim() == 3
        else images
    )
    if samples.shape[0] != grid_sample_coordinates.shape[0]:
        raise ValueError(
            "Provide either an equal batch of images and matrices or "
            "multiple matrices for a single image."
        )
    transformed = einops.rearrange(
        F.grid_sample(
            einops.rearrange(samples, "... d h w -> ... 1 d h w"),
            grid_sample_coordinates,
            align_corners=True,
            mode=interpolation,
        ),
        "... 1 d h w -> ... d h w",  # remove channel
    )
    if len(images.shape) == 3:  # remove starter dimensions in case we got one image
        transformed = transformed.squeeze()
    return transformed

"""Real-space projection of tilt-series images and tomograms."""

from typing import Tuple

import einops
import torch
from cryotypes.projectionmodel import ProjectionModel
from cryotypes.projectionmodel import ProjectionModelDataLabels as PMDL
from torch_image_lerp import insert_into_image_2d

from tttsa.transformations import (
    projection_model_to_projection_matrix,
)
from tttsa.utils import prep_grid_cached

# update shift
PMDL.SHIFT = [PMDL.SHIFT_Y, PMDL.SHIFT_X]


def tomogram_reprojection(
    tomogram: torch.Tensor,
    tilt_image_dimensions: Tuple[int, int],
    projection_model: ProjectionModel,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict a projection from an intermediate reconstruction.

    For now only assumes to project with a single matrix, but should also work for
    sets of matrices.
    """
    device = tomogram.device
    tomogram_dimensions = tomogram.shape

    # time for real space projection
    M = projection_model_to_projection_matrix(
        projection_model, tilt_image_dimensions, tomogram_dimensions
    )
    Mproj = M[:, 1:3, :]
    Mproj = einops.rearrange(Mproj, "... i j -> ... 1 1 i j").to(device)

    grid = prep_grid_cached(tomogram_dimensions, device)
    grid = Mproj @ grid
    grid = einops.rearrange(grid, "... d h w coords 1 -> ... d h w coords")

    projection, weights = insert_into_image_2d(
        tomogram.view(-1),  # flatten
        grid.view(-1, 2),
        torch.zeros(tilt_image_dimensions, device=device),
    )
    return projection, weights

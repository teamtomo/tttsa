"""Run the projection matching algorithm."""

from typing import Tuple

import einops
import torch
from cryotypes.projectionmodel import ProjectionModel
from cryotypes.projectionmodel import ProjectionModelDataLabels as PMDL
from rich.progress import track

from .alignment import find_image_shift
from .back_projection import filtered_back_projection_3d
from .projection import tomogram_reprojection

# update shift
PMDL.SHIFT = [PMDL.SHIFT_Y, PMDL.SHIFT_X]


def projection_matching(
    tilt_series: torch.Tensor,
    projection_model_in: ProjectionModel,
    reference_tilt_id: int,
    alignment_mask: torch.Tensor,
    tomogram_dimensions: Tuple[int, int, int],
    reconstruction_weighting: str = "hamming",
    exact_weighting_object_diameter: float | None = None,
) -> Tuple[ProjectionModel, torch.Tensor]:
    """Run projection matching."""
    device = tilt_series.device
    n_tilts, size, _ = tilt_series.shape
    aligned_set = [reference_tilt_id]
    # copy the model to update with new shifts
    projection_model_out = projection_model_in.copy(deep=True)
    tilt_angles = torch.tensor(  # to tensor as we need it to calculate weights
        projection_model_out[PMDL.ROTATION_Y].to_numpy(), dtype=tilt_series.dtype
    )

    # generate indices by alternating postive/negative tilts
    max_offset = max(reference_tilt_id, n_tilts - reference_tilt_id - 1)
    index_sequence = []
    for i in range(1, max_offset + 1):  # skip reference
        if reference_tilt_id + i < n_tilts:
            index_sequence.append(reference_tilt_id + i)
        if i > 0 and reference_tilt_id - i >= 0:
            index_sequence.append(reference_tilt_id - i)

    # for debugging:
    projections = torch.zeros((n_tilts, size, size))
    projections[reference_tilt_id] = tilt_series[reference_tilt_id]

    for i in track(index_sequence):
        tilt_angle = tilt_angles[i]
        weights = einops.rearrange(
            torch.cos(torch.deg2rad(torch.abs(tilt_angles - tilt_angle))),
            "n -> n 1 1",
        ).to(device)
        intermediate_recon = filtered_back_projection_3d(
            tilt_series[aligned_set,] * weights[aligned_set,],
            tomogram_dimensions,
            projection_model_out.iloc[aligned_set,],
            weighting=reconstruction_weighting,
            object_diameter=exact_weighting_object_diameter,
        )
        projection, projection_weights = tomogram_reprojection(
            intermediate_recon,
            (size, size),
            projection_model_out.iloc[[i],],
        )

        # ensure correlation in relevant area
        projection_weights = projection_weights / projection_weights.max()
        projection_weights *= alignment_mask
        projection *= projection_weights
        projection = (projection - projection.mean()) / projection.std()
        raw = tilt_series[i] * projection_weights
        raw = (raw - raw.mean()) / raw.std()
        shift = find_image_shift(
            raw,
            projection,
        )
        projection_model_out.loc[i, PMDL.SHIFT] += shift.numpy()
        aligned_set.append(i)

        # for debugging:
        projections[i] = projection.detach().cpu()

    return projection_model_out, projections

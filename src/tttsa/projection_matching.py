"""Run the projection matching algorithm."""

from typing import Tuple

import einops
import torch
from rich.progress import track

from .alignment import find_image_shift
from .back_projection import filtered_back_projection_3d
from .projection import tomogram_reprojection


def projection_matching(
    tilt_series: torch.Tensor,
    tomogram_dimensions: Tuple[int, int, int],
    reference_tilt_id: int,
    tilt_angles: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
    current_shifts: torch.Tensor,
    alignment_mask: torch.Tensor,
    reconstruction_weighting: str = "hamming",
    exact_weighting_object_diameter: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run projection matching."""
    device = tilt_series.device
    n_tilts, size, _ = tilt_series.shape
    aligned_set = [reference_tilt_id]
    shifts = current_shifts.detach().clone()

    # generate indices by alternating postive/negative tilts
    max_offset = max(reference_tilt_id, len(tilt_angles) - reference_tilt_id - 1)
    index_sequence = []
    for i in range(1, max_offset + 1):  # skip reference
        if reference_tilt_id + i < len(tilt_angles):
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
        )
        intermediate_recon, _ = filtered_back_projection_3d(
            tilt_series[aligned_set,] * weights[aligned_set,].to(device),
            tomogram_dimensions,
            tilt_angles[aligned_set,],
            tilt_axis_angles[aligned_set,],
            shifts[aligned_set,],
            weighting=reconstruction_weighting,
            object_diameter=exact_weighting_object_diameter,
        )
        projection, projection_weights = tomogram_reprojection(
            intermediate_recon,
            (size, size),
            tilt_angles[[i],],
            tilt_axis_angles[[i],],
            shifts[[i],],
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
        shifts[i] -= shift
        aligned_set.append(i)

        # for debugging:
        projections[i] = projection.detach().cpu()

    return shifts, projections

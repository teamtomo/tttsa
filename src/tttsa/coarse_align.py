"""Coarse tilt-series alignment functions, also with stretching."""

import einops
import torch

from .affine import affine_transform_2d
from .alignment import find_image_shift
from .transformations import stretch_matrix


def coarse_align(
    tilt_series: torch.Tensor,
    reference_tilt_id: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Find coarse shifts of images without stretching along tilt axis."""
    n_tilts = len(tilt_series)
    shifts = torch.zeros((n_tilts, 2), dtype=torch.float32)
    ts_masked = tilt_series * mask
    ts_masked -= einops.reduce(ts_masked, "tilt h w -> tilt 1 1", reduction="mean")
    ts_masked /= torch.std(ts_masked, dim=(-2, -1), keepdim=True)

    # find coarse alignment for negative tilts
    current_shift = torch.zeros(2)
    for i in range(reference_tilt_id, 0, -1):
        shift = find_image_shift(ts_masked[i], ts_masked[i - 1])
        current_shift += shift
        shifts[i - 1] = current_shift

    # find coarse alignment positive tilts
    current_shift = torch.zeros(2)
    for i in range(reference_tilt_id, n_tilts - 1, 1):
        shift = find_image_shift(
            ts_masked[i],
            ts_masked[i + 1],
        )
        current_shift += shift
        shifts[i + 1] = current_shift
    return shifts


def stretch_align(
    tilt_series: torch.Tensor,
    reference_tilt_id: int,
    mask: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
) -> torch.Tensor:
    """Find coarse shifts of images while stretching each pair along the tilt axis."""
    n_tilts, h, w = tilt_series.shape
    tilt_image_dimensions = (h, w)
    shifts = torch.zeros((n_tilts, 2), dtype=torch.float32)
    cos_ta = torch.cos(torch.deg2rad(tilt_angles))

    # find coarse alignment for negative tilts
    current_shift = torch.zeros(2)
    for i in range(reference_tilt_id, 0, -1):
        M = stretch_matrix(
            tilt_image_dimensions,
            tilt_axis_angles[i - 1],
            scale_factor=cos_ta[i : i + 1] / cos_ta[i - 1 : i],
        )
        stretched = affine_transform_2d(tilt_series[i - 1], M) * mask
        stretched = (stretched - stretched.mean()) / stretched.std()
        raw = tilt_series[i] * mask
        raw = (raw - raw.mean()) / raw.std()
        shift = find_image_shift(raw, stretched)
        current_shift += shift
        shifts[i - 1] = current_shift
    # find coarse alignment positive tilts
    current_shift = torch.zeros(2)
    for i in range(reference_tilt_id, n_tilts - 1, 1):
        M = stretch_matrix(
            tilt_image_dimensions,
            tilt_axis_angles[i + 1],
            scale_factor=cos_ta[i : i + 1] / cos_ta[i + 1 : i + 2],
        )
        stretched = affine_transform_2d(tilt_series[i + 1], M) * mask
        stretched = (stretched - stretched.mean()) / stretched.std()
        raw = tilt_series[i] * mask
        raw = (raw - raw.mean()) / raw.std()
        shift = find_image_shift(raw, stretched)
        current_shift += shift
        shifts[i + 1] = current_shift
    return shifts

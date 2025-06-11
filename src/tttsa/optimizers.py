"""Optimization of tilt-series parameters."""

import torch

from .affine import affine_transform_2d
from .transformations import T_2d, stretch_matrix


def stretch_loss(
    tilt_series: torch.Tensor,
    reference_tilt_id: int,
    mask: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Find coarse shifts of images while stretching each pair along the tilt axis."""
    device = tilt_series.device
    n_tilts, h, w = tilt_series.shape
    tilt_image_dimensions = (h, w)
    cos_ta = torch.cos(torch.deg2rad(tilt_angles))

    sq_diff = torch.tensor(0.0, device=device)
    for i in range(reference_tilt_id, 0, -1):
        # multiply stretch matrix by shift for full alignment
        M = T_2d(shifts[i - 1] - shifts[i]) @ stretch_matrix(
            tilt_image_dimensions,
            tilt_axis_angles[i - 1],
            scale_factor=cos_ta[i : i + 1] / cos_ta[i - 1 : i],
        )  # slicing cos_ta ensure gradient calculation
        stretched = affine_transform_2d(tilt_series[i - 1], M)
        non_empty = (stretched != 0) * 1.0
        correlation_mask = non_empty * mask
        stretched = stretched * correlation_mask
        stretched = (stretched - stretched.mean()) / stretched.std()
        ref = tilt_series[i] * correlation_mask
        ref = (ref - ref.mean()) / ref.std()
        sq_diff = sq_diff + ((ref - stretched) ** 2).sum() / stretched.numel()

    # find coarse alignment positive tilts
    for i in range(reference_tilt_id, n_tilts - 1, 1):
        # multiply stretch matrix by shift for full alignment
        M = T_2d(shifts[i + 1] - shifts[i]) @ stretch_matrix(
            tilt_image_dimensions,
            tilt_axis_angles[i + 1],
            scale_factor=cos_ta[i : i + 1] / cos_ta[i + 1 : i + 2],
        )  # slicing cos_ta ensure gradient calculation
        stretched = affine_transform_2d(tilt_series[i + 1], M)
        non_empty = (stretched != 0) * 1.0
        correlation_mask = non_empty * mask
        stretched = stretched * correlation_mask
        stretched = (stretched - stretched.mean()) / stretched.std()
        ref = tilt_series[i] * correlation_mask
        ref = (ref - ref.mean()) / ref.std()
        sq_diff = sq_diff + ((ref - stretched) ** 2).sum() / stretched.numel()
    return sq_diff.cpu()


def optimize_tilt_angle_offset(
    tilt_series: torch.Tensor,
    mask: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Optimize a tilt-angle offset for the lowest stretch correlation loss.

    TODO use a grid based search.
    """
    tilt_angle_offset = torch.tensor(0.0, requires_grad=True)
    lbfgs = torch.optim.LBFGS(
        [tilt_angle_offset],
        history_size=10,
        max_iter=4,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        new_tilt_angles = tilt_angles + tilt_angle_offset
        new_ref = int((new_tilt_angles).abs().argmin())
        lbfgs.zero_grad()
        loss = stretch_loss(
            tilt_series, new_ref, mask, new_tilt_angles, tilt_axis_angles, shifts
        )
        loss.backward()
        return loss

    for _ in range(3):
        lbfgs.step(closure)

    return tilt_angle_offset.detach()

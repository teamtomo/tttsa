"""Optimization of tilt-series parameters."""

import einops
import torch
from torch_cubic_spline_grids import CubicBSplineGrid1d

from .affine import affine_transform_2d, stretch_image
from .projection import common_lines_projection
from .transformations import T_2d


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
    sq_diff = torch.tensor(0.0, device=device)
    for i in range(reference_tilt_id, 0, -1):
        scale_factor = torch.cos(torch.deg2rad(tilt_angles[i : i + 1])) / torch.cos(
            torch.deg2rad(tilt_angles[i - 1 : i])
        )
        stretched = stretch_image(  # stretch image i - 1
            tilt_series[i - 1],
            scale_factor,
            tilt_axis_angles[i - 1],
        )
        stretched = affine_transform_2d(  # shift to the same position as i
            stretched,
            T_2d(shifts[i - 1] - shifts[i]),
        )
        non_empty = (stretched != 0) * 1.0
        correlation_mask = non_empty * mask
        stretched = stretched * correlation_mask
        stretched = (stretched - stretched.mean()) / stretched.std()
        ref = tilt_series[i] * correlation_mask
        ref = (ref - ref.mean()) / ref.std()
        sq_diff = sq_diff + ((ref - stretched) ** 2).sum() / stretched.numel()

    # find coarse alignment positive tilts
    for i in range(reference_tilt_id, tilt_series.shape[0] - 1, 1):
        scale_factor = torch.cos(torch.deg2rad(tilt_angles[i : i + 1])) / torch.cos(
            torch.deg2rad(tilt_angles[i + 1 : i + 2])
        )
        stretched = stretch_image(  # stretch image i + 1
            tilt_series[i + 1],
            scale_factor,
            tilt_axis_angles[i + 1],
        )
        stretched = affine_transform_2d(  # shift to positions of i
            stretched,
            T_2d(shifts[i + 1] - shifts[i]),
        )
        non_empty = (stretched != 0) * 1.0
        correlation_mask = non_empty * mask
        stretched = stretched * correlation_mask
        stretched = (stretched - stretched.mean()) / stretched.std()
        ref = tilt_series[i] * correlation_mask
        ref = (ref - ref.mean()) / ref.std()
        sq_diff = sq_diff + ((ref - stretched) ** 2).sum() / stretched.numel()
    return sq_diff.cpu()


def optimize_tilt_axis_angle(
    aligned_ts: torch.Tensor,
    coarse_alignment_mask: torch.Tensor,
    initial_tilt_axis_angle: torch.Tensor | float,
    grid_points: int = 1,
) -> torch.Tensor:
    """Optimize tilt axis angles on a spline grid using the LBFGS optimizer."""
    coarse_aligned_masked = aligned_ts * coarse_alignment_mask

    # generate a weighting for the common line ROI by projecting the mask
    mask_weights = common_lines_projection(
        einops.rearrange(coarse_alignment_mask, "h w -> 1 h w"),
        0.0,  # angle does not matter
    )
    mask_weights /= mask_weights.max()  # normalise to 0 and 1

    # optimize tilt axis angle
    tilt_axis_grid = CubicBSplineGrid1d(resolution=grid_points, n_channels=1)
    tilt_axis_grid.data = torch.tensor(
        [
            torch.mean(initial_tilt_axis_angle),
        ]
        * grid_points,
        dtype=torch.float32,
    )
    interpolation_points = torch.linspace(0, 1, len(aligned_ts))

    lbfgs = torch.optim.LBFGS(
        tilt_axis_grid.parameters(),
        history_size=10,
        max_iter=4,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        tilt_axis_angles = tilt_axis_grid(interpolation_points)
        projections = common_lines_projection(coarse_aligned_masked, tilt_axis_angles)
        projections = projections - einops.reduce(
            projections, "tilt w -> tilt 1", reduction="mean"
        )
        projections = projections / torch.std(projections, dim=(-1), keepdim=True)
        # weight the lines by the projected mask
        projections = projections * mask_weights

        lbfgs.zero_grad()
        squared_differences = (
            projections - einops.rearrange(projections, "b d -> b 1 d")
        ) ** 2
        loss = einops.reduce(squared_differences, "b1 b2 d -> 1", reduction="sum")
        loss.backward()
        return loss.cpu()

    for _ in range(3):
        lbfgs.step(closure)

    tilt_axis_angles = tilt_axis_grid(interpolation_points)

    return tilt_axis_angles.detach()


def optimize_tilt_angle_offset(
    tilt_series: torch.Tensor,
    mask: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angles: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Optimize a tilt-angle offset for the lowest stretch correlation loss."""
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

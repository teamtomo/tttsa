"""Main program for aligning tilt-series."""

from typing import Tuple

import torch
from rich.console import Console
from rich.progress import track
from torch_fourier_shift import fourier_shift_image_2d

from .coarse_align import coarse_align, stretch_align
from .optimizers import optimize_tilt_angle_offset, optimize_tilt_axis_angle
from .projection_matching import projection_matching
from .utils import circle

# import logging
# log = logging.getLogger(__name__)

console = Console()


def tilt_series_alignment(
    tilt_series: torch.Tensor,
    tilt_angle_priors: torch.Tensor,
    tilt_axis_angle_prior: torch.Tensor,
    alignment_z_height: int,
    find_tilt_angle_offset: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align a tilt-series using AreTomo-style projection matching.

    AreTomo paper:
        Zheng, Shawn, et al. "AreTomo: An integrated software package for automated
        marker-free, motion-corrected cryo-electron tomographic alignment and
        reconstruction." Journal of Structural Biology: X 6 (2022): 100068.
    On GitHub:
        https://github.com/czimaginginstitute/AreTomo2
        https://github.com/czimaginginstitute/AreTomo3
    """
    # set tomogram and tilt-series shape
    n_tilts, h, w = tilt_series.shape
    size = min(h, w)
    tomogram_dimensions = (alignment_z_height, size, size)
    tilt_dimensions = (size,) * 2
    reference_tilt = int(tilt_angle_priors.abs().argmin())

    # mask for coarse alignment
    coarse_alignment_mask = circle(  # ttmask -> tt-shapes; maybe add function
        radius=size // 3,
        smoothing_radius=size // 6,
        image_shape=tilt_dimensions,
    )

    # do an IMOD style coarse tilt-series alignment
    coarse_shifts = coarse_align(tilt_series, reference_tilt, coarse_alignment_mask)

    tilt_axis_angles = torch.tensor(tilt_axis_angle_prior)
    shifts = coarse_shifts.clone()
    tilt_angles = tilt_angle_priors.clone()
    start_taa_grid_points = 1  # taa = tilt-axis angle
    pm_taa_grid_points = 3  # pm = projection matching
    console.print("=== Starting teamtomo tilt-series alignment!", style="bold blue")
    console.print(
        f"=== Optimizing tilt-axis angle with {start_taa_grid_points} grid point."
    )
    for _ in track(range(3)):  # optimize tilt axis angle
        tilt_axis_angles = optimize_tilt_axis_angle(
            fourier_shift_image_2d(tilt_series, shifts=shifts),
            coarse_alignment_mask,
            tilt_axis_angles,
            grid_points=start_taa_grid_points,
        )

        shifts = stretch_align(
            tilt_series,
            reference_tilt,
            coarse_alignment_mask,
            tilt_angles,
            tilt_axis_angles,
        )

    console.print(
        f"=== New tilt axis angle: {tilt_axis_angles.mean():.2f}° +-"
        f" {tilt_axis_angles.std():.2f}°"
    )

    if find_tilt_angle_offset:
        full_offset = torch.tensor(0.0)
        console.print("=== Optimizing tilt-angle offset.")
        for _ in track(range(3)):
            tilt_angle_offset = optimize_tilt_angle_offset(
                tilt_series,
                coarse_alignment_mask,
                tilt_angles,
                tilt_axis_angles,
                shifts,
            )
            full_offset += tilt_angle_offset.detach()
            tilt_angles = tilt_angles + tilt_angle_offset.detach()
            reference_tilt = int((tilt_angles).abs().argmin())

            shifts = stretch_align(
                tilt_series,
                reference_tilt,
                coarse_alignment_mask,
                tilt_angles,
                tilt_axis_angles,
            )
        console.print(f"=== Detected tilt-angle offset: {full_offset:.2f}°")

    # some optimizations parameters
    max_iter = 10  # this seems solid
    tolerance = 0.1  # should probably be related to pixel size
    predicted_tilts = []
    console.print(
        f"=== Starting projection matching with"
        f" {pm_taa_grid_points} grid points for the tilt-axis angle."
    )
    for i in range(max_iter):
        tilt_axis_angles = optimize_tilt_axis_angle(
            fourier_shift_image_2d(tilt_series, shifts=shifts),
            coarse_alignment_mask,
            tilt_axis_angles,
            grid_points=pm_taa_grid_points,
        )

        new_shifts, pred = projection_matching(
            tilt_series,
            tomogram_dimensions,
            reference_tilt,  # REFERENCE_TILT,
            tilt_angles,
            tilt_axis_angles,
            shifts,
            coarse_alignment_mask,
        )
        predicted_tilts.append(pred)

        console.print(
            f"--> Iteration {i + 1}, "
            f"sum of translation differences ="
            f" {torch.abs(shifts - new_shifts).sum():.2f}"
        )

        if torch.all(torch.abs(shifts - new_shifts) < tolerance):
            break

        shifts = new_shifts

    console.print(
        f"=== Final tilt-axis angle: {tilt_axis_angles.mean():.2f}° +-"
        f" {tilt_axis_angles.std():.2f}°"
    )
    console.print("===== Done!")

    return tilt_angles, tilt_axis_angles, shifts

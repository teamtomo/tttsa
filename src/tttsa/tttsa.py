"""Main program for aligning tilt-series."""

# set logger => TODO use rich instead
import logging
from typing import Tuple

import torch
from torch_fourier_shift import fourier_shift_image_2d

from .coarse_align import coarse_align, stretch_align
from .optimizers import optimize_tilt_angle_offset, optimize_tilt_axis_angle
from .projection_matching import projection_matching
from .utils import circle

log = logging.getLogger(__name__)


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
    coarse_aligned = fourier_shift_image_2d(tilt_series, shifts=coarse_shifts)

    tilt_axis_angles = torch.tensor(tilt_axis_angle_prior)
    shifts = coarse_shifts.clone()
    tilt_angles = tilt_angle_priors.clone()

    for _ in range(3):  # optimize tilt axis angle
        tilt_axis_angles = optimize_tilt_axis_angle(
            coarse_aligned,
            coarse_alignment_mask,
            tilt_axis_angles,
        )
        print(
            f"new tilt axis angle: {tilt_axis_angles.mean():.2f} +-"
            f" {tilt_axis_angles.std():.2f}"
        )  # use rich logging?

        shifts = stretch_align(
            tilt_series,
            reference_tilt,
            coarse_alignment_mask,
            tilt_angles,
            tilt_axis_angles,
        )

        coarse_aligned = fourier_shift_image_2d(tilt_series, shifts=shifts)

    if find_tilt_angle_offset:
        for _ in range(3):
            tilt_angle_offset = optimize_tilt_angle_offset(
                tilt_series,
                coarse_alignment_mask,
                tilt_angles,
                tilt_axis_angles,
                shifts,
            )
            print(f"detected tilt angle offset: {tilt_angle_offset}")
            tilt_angles = tilt_angles + tilt_angle_offset.detach()
            reference_tilt = int((tilt_angles).abs().argmin())

            shifts = stretch_align(
                tilt_series,
                reference_tilt,
                coarse_alignment_mask,
                tilt_angles,
                tilt_axis_angles,
            )

    # some optimizations parameters
    max_iter = 10  # this seems solid
    tolerance = 0.1  # should probably be related to pixel size
    predicted_tilts = []
    for i in range(max_iter):
        print(f"projection matching iteration {i}")
        tilt_axis_angles = optimize_tilt_axis_angle(
            fourier_shift_image_2d(tilt_series, shifts=shifts),
            coarse_alignment_mask,
            tilt_axis_angles,
            grid_points=3,
        )
        print("new tilt axis angle:", tilt_axis_angles)

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

        if torch.all(torch.abs(shifts - new_shifts) < tolerance):
            break

        shifts = new_shifts

    return tilt_angles, tilt_axis_angles, shifts

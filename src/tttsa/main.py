import torch

from torch_fourier_shift import fourier_shift_image_2d
from libtilt.shapes import circle

from .utils import dft_center
from .coarse_align import coarse_align, stretch_align
from .optimizers import optimize_tilt_axis_angle, optimize_tilt_angle_offset
from .projection_matching import projection_matching
from tttsa.back_projection import filtered_back_projection_3d

# set logger
import logging
log = logging.getLogger(__name__)


def tilt_series_alignment(
        tilt_series: torch.Tensor,
        tilt_angle_priors: torch.Tensor,
        tilt_axis_angle_prior: torch.Tensor,
        reference_tilt: int,
        alignment_z_height
):

    # set tomogram and tilt-series shape
    size = min(h, w)
    tomogram_dimensions = (size,) * 3
    tilt_dimensions = (size,) * 2

    # mask for coarse alignment
    coarse_alignment_mask = circle(  # ttmask -> tt-shapes; maybe add function
        radius=size // 3,
        smoothing_radius=size // 6,
        image_shape=tilt_dimensions,
    )

    tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
    tilt_image_center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)

    # do an IMOD style coarse tilt-series alignment
    coarse_shifts = coarse_align(tilt_series, REFERENCE_TILT, coarse_alignment_mask)
    coarse_aligned = fourier_shift_image_2d(tilt_series, shifts=coarse_shifts)

    tilt_axis_angle = torch.tensor(TILT_AXIS_ANGLE_PRIOR)
    shifts = coarse_shifts.clone()
    reference_tilt = REFERENCE_TILT
    tilt_angles = STAGE_TILT_ANGLE_PRIORS.clone()

    for _ in range(3):  # optimize tilt axis angle

        tilt_axis_angle = optimize_tilt_axis_angle(
            coarse_aligned,
            coarse_alignment_mask,
            tilt_axis_angle,
        )
        print(f"new tilt axis angle: {tilt_axis_angle.mean():.2f} +-"
              f" {tilt_axis_angle.std():.2f}")  # use rich logging?

        shifts = stretch_align(
            tilt_series, reference_tilt, coarse_alignment_mask, tilt_angles, tilt_axis_angle
        )

        coarse_aligned = fourier_shift_image_2d(tilt_series, shifts=shifts)

    for _ in range(3):
        tilt_angle_offset = optimize_tilt_angle_offset(
            tilt_series,
            coarse_alignment_mask,
            STAGE_TILT_ANGLE_PRIORS,
            tilt_axis_angle,
            shifts
        )
        print(f"detected tilt angle offset: {tilt_angle_offset}")
        tilt_angles = STAGE_TILT_ANGLE_PRIORS + tilt_angle_offset.detach()
        reference_tilt = int((tilt_angles).abs().argmin())

        shifts = stretch_align(
            tilt_series, reference_tilt, coarse_alignment_mask, tilt_angles, tilt_axis_angle
        )


    # coarse reconstruction
    coarse_aligned = fourier_shift_image_2d(tilt_series, shifts=shifts)
    initial_reconstruction, _ = filtered_back_projection_3d(
        tilt_series,
        (RECON_Z, size, size),
        tilt_angles,  # STAGE_TILT_ANGLE_PRIORS
        tilt_axis_angle,
        shifts,
        weighting=WEIGHTING,
        object_diameter=OBJECT_DIAMETER,
    )
    viewer = napari.Viewer()
    viewer.add_image(initial_reconstruction.detach().numpy())
    napari.run()

    # some optimizations parameters
    max_iter = 10  # this seems solid
    tolerance = 0.1  # should probably be related to pixel size
    predicted_tilts = []
    for i in range(max_iter):
        print(f"projection matching iteration {i}")
        tilt_axis_angle = optimize_tilt_axis_angle(
            fourier_shift_image_2d(tilt_series, shifts=shifts),
            coarse_alignment_mask,
            tilt_axis_angle,
            grid_points=3,
        )
        print("new tilt axis angle:", tilt_axis_angle)

        new_shifts, pred = projection_matching(
            tilt_series,
            (ALIGN_Z, size, size),
            reference_tilt,  # REFERENCE_TILT,
            tilt_angles,
            tilt_axis_angle,
            shifts,
            coarse_alignment_mask,
            debug=False,
        )
        predicted_tilts.append(pred)

        if torch.all(torch.abs(shifts - new_shifts) < tolerance):
            break

        shifts = new_shifts

    # viewer = napari.Viewer()
    # viewer.add_image(tilt_series.detach().numpy(), name="raw tilts")
    # for i, p in enumerate(predicted_tilts):
    #     viewer.add_image(p.detach().numpy(), name=f"prediction at iter {i}")
    # napari.run()



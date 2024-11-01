from pathlib import Path

import einops
import mrcfile
import napari
import numpy as np
import torch
import torch.nn.functional as F

from libtilt.patch_extraction import extract_squares
from libtilt.rescaling.rescale_fourier import rescale_2d
from libtilt.shapes import circle
from libtilt.shift.shift_image import shift_2d

from .utils import dft_center
from .transformations import Ry, Rz, T
from .coarse_align import coarse_align, stretch_align
from .optimizers import optimize_tilt_axis_angle, optimize_tilt_angle_offset
from .projection_matching import projection_matching
from tttsa.back_projection import filtered_back_projection_3d

# set logger
import logging
log = logging.getLogger(__name__)


IMAGE_FILE = Path("data/tomo200528_100.st")
IMAGE_PIXEL_SIZE = 1.724
STAGE_TILT_ANGLE_PRIORS = torch.arange(-51, 51, 3)  # 107: 54, 100: 51
TILT_AXIS_ANGLE_PRIOR = -90.0  # -88.7 according to mdoc, but I set it faulty to see if
# the optimization works
ALIGNMENT_PIXEL_SIZE = IMAGE_PIXEL_SIZE * 8
# set 0 degree tilt as reference
REFERENCE_TILT = int(STAGE_TILT_ANGLE_PRIORS.abs().argmin())
ALIGN_Z = int(2000 / ALIGNMENT_PIXEL_SIZE)  # number is in A
RECON_Z = int(3000 / ALIGNMENT_PIXEL_SIZE)
WEIGHTING = "hamming"  # weighting scheme for filtered back projection
# the object diameter in number of pixels
OBJECT_DIAMETER = 300 / ALIGNMENT_PIXEL_SIZE


def tilt_series_alignment():

    tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))

    tilt_series, _ = rescale_2d(
        image=tilt_series,
        source_spacing=IMAGE_PIXEL_SIZE,
        target_spacing=ALIGNMENT_PIXEL_SIZE,
        maintain_center=True,
    )

    tilt_series -= einops.reduce(tilt_series, "tilt h w -> tilt 1 1", reduction="mean")
    tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)
    n_tilts, h, w = tilt_series.shape
    center = dft_center((h, w), rfft=False, fftshifted=True)
    center = einops.repeat(center, "yx -> b yx", b=len(tilt_series))
    tilt_series = extract_squares(
        image=tilt_series,
        positions=center,
        sidelength=min(h, w),
    )

    # set tomogram and tilt-series shape
    size = min(h, w)
    tomogram_dimensions = (size,) * 3
    tilt_dimensions = (size,) * 2

    # mask for coarse alignment
    coarse_alignment_mask = circle(
        radius=size // 3,
        smoothing_radius=size // 6,
        image_shape=tilt_dimensions,
    )

    tomogram_center = dft_center(tomogram_dimensions, rfft=False, fftshifted=True)
    tilt_image_center = dft_center(tilt_dimensions, rfft=False, fftshifted=True)

    # do an IMOD style coarse tilt-series alignment
    coarse_shifts = coarse_align(tilt_series, REFERENCE_TILT, coarse_alignment_mask)
    coarse_aligned = shift_2d(tilt_series, shifts=coarse_shifts)

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
              f" {tilt_axis_angle.std():.2f}")

        shifts = stretch_align(
            tilt_series, reference_tilt, coarse_alignment_mask, tilt_angles, tilt_axis_angle
        )

        coarse_aligned = shift_2d(tilt_series, shifts=shifts)

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
    coarse_aligned = shift_2d(tilt_series, shifts=shifts)
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
            shift_2d(tilt_series, shifts=shifts),
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

    final, aligned_ts = filtered_back_projection_3d(
        tilt_series,
        (RECON_Z, size, size),
        tilt_angles,
        tilt_axis_angle,
        shifts,
        weighting=WEIGHTING,
        object_diameter=OBJECT_DIAMETER,
    )

    mrcfile.write(
        IMAGE_FILE.with_name(IMAGE_FILE.stem + "_exact.mrc"),
        final.detach().numpy().astype(np.float32),
        voxel_size=ALIGNMENT_PIXEL_SIZE,
        overwrite=True,
    )

    # generate a proper fourier inverted reconstruction
    s0 = T(-tomogram_center)
    r0 = Ry(tilt_angles, zyx=True)
    r1 = Rz(tilt_axis_angle, zyx=True)
    s2 = T(F.pad(tilt_image_center, pad=(1, 0), value=0))
    M = s2 @ r1 @ r0 @ s0

    fine_reconstruction = backproject_fourier(
        images=shift_2d(tilt_series, shifts),
        rotation_matrices=torch.linalg.inv(M[:, :3, :3]),
        rotation_matrix_zyx=True,
        do_gridding_correction=False,
    )

    mrcfile.write(
        IMAGE_FILE.with_name(IMAGE_FILE.stem + "_fine.mrc"),
        fine_reconstruction.detach().numpy().astype(np.float32),
        voxel_size=ALIGNMENT_PIXEL_SIZE,
        overwrite=True,
    )

    viewer = napari.Viewer()
    viewer.add_image(initial_reconstruction.detach().numpy(), name="initial reconstruction")
    viewer.add_image(final.detach().numpy(), name="optimized reconstruction")
    viewer.add_image(fine_reconstruction.detach().numpy(), name="fourier inverted")
    viewer.add_image(aligned_ts.detach().numpy(), name="aligned_ts")
    napari.run()

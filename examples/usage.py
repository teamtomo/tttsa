"""Example of tttsa.tilt_series_alignment() usage.

TODO download example data from Zenodo.
"""

from pathlib import Path

import einops
import mrcfile
import numpy as np
import torch
from torch_fourier_rescale import fourier_rescale_2d
from torch_subpixel_crop import subpixel_crop_2d

from tttsa import tilt_series_alignment
from tttsa.back_projection import filtered_back_projection_3d
from tttsa.utils import dft_center

IMAGE_FILE = Path("data/tomo200528_107.st")
IMAGE_PIXEL_SIZE = 1.724
STAGE_TILT_ANGLE_PRIORS = torch.arange(-51, 54, 3)  # 107: 54, 100: 51
TILT_AXIS_ANGLE_PRIOR = -88.7  # -88.7 according to mdoc, but faulty to test
# this angle is assumed to be a clockwise forward rotation after projecting the sample
ALIGNMENT_PIXEL_SIZE = IMAGE_PIXEL_SIZE * 8
ALIGN_Z = int(1600 / ALIGNMENT_PIXEL_SIZE)  # number is in A
RECON_Z = int(2400 / ALIGNMENT_PIXEL_SIZE)
WEIGHTING = "hamming"  # weighting scheme for filtered back projection
# the object diameter in number of pixels
OBJECT_DIAMETER = 300 / ALIGNMENT_PIXEL_SIZE


tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))

tilt_series, _ = fourier_rescale_2d(  # should normalize beforehand
    image=tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
)  # Ensure normalization after fourier rescale
tilt_series -= einops.reduce(tilt_series, "tilt h w -> tilt 1 1", reduction="mean")
tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)

n_tilts, h, w = tilt_series.shape
center = dft_center((h, w), rfft=False, fftshifted=True)
center = einops.repeat(center, "yx -> b yx", b=n_tilts)
tilt_series = subpixel_crop_2d(  # torch-subpixel-crop
    image=tilt_series,
    positions=center,
    sidelength=min(h, w),
)[:, :464, :]  # TODO remove temp workaround for subpixel crop
_, h, w = tilt_series.shape
size = min(h, w)

tilt_angles, tilt_axis_angles, shifts = tilt_series_alignment(
    tilt_series, STAGE_TILT_ANGLE_PRIORS, TILT_AXIS_ANGLE_PRIOR, ALIGN_Z
)

final, aligned_ts = filtered_back_projection_3d(
    tilt_series,
    (RECON_Z, size, size),
    tilt_angles,
    tilt_axis_angles,
    shifts,
    weighting=WEIGHTING,
    object_diameter=OBJECT_DIAMETER,
)

mrcfile.write(
    IMAGE_FILE.with_suffix(".mrc"),
    final.detach().numpy().astype(np.float32),
    voxel_size=ALIGNMENT_PIXEL_SIZE,
    overwrite=True,
)

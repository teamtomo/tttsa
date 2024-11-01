import torch
import mrcfile
import napari
import numpy
import einops

from pathlib import Path
from tttsa import tilt_series_alignment
from tttsa.back_projection import filtered_back_projection_3d
from torch_fourier_rescale import fourier_rescale_2d


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


tilt_series = torch.as_tensor(mrcfile.read(IMAGE_FILE))

tilt_series, _ = fourier_rescale_2d(  # should normalize beforehand
    image=tilt_series,
    source_spacing=IMAGE_PIXEL_SIZE,
    target_spacing=ALIGNMENT_PIXEL_SIZE,
    maintain_center=True,
    # rescale_intensity
)

tilt_series -= einops.reduce(tilt_series, "tilt h w -> tilt 1 1", reduction="mean")
tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)
n_tilts, h, w = tilt_series.shape
center = dft_center((h, w), rfft=False, fftshifted=True)
center = einops.repeat(center, "yx -> b yx", b=len(tilt_series))
tilt_series = extract_squares(  # torch-subpixel-crop
    image=tilt_series,
    positions=center,
    sidelength=min(h, w),
)


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

viewer = napari.Viewer()
viewer.add_image(initial_reconstruction.detach().numpy(), name="initial reconstruction")
viewer.add_image(final.detach().numpy(), name="optimized reconstruction")
viewer.add_image(fine_reconstruction.detach().numpy(), name="fourier inverted")
viewer.add_image(aligned_ts.detach().numpy(), name="aligned_ts")
napari.run()
"""Run the projection matching algorithm."""

from typing import Tuple

import einops
import torch
import torch.nn.functional as F
from cryotypes.projectionmodel import ProjectionModel
from cryotypes.projectionmodel import ProjectionModelDataLabels as PMDL
from rich.progress import track
from torch_grid_utils import coordinate_grid

from .affine import affine_transform_2d
from .alignment import find_image_shift
from .transformations import R_2d, T_2d, projection_model_to_tsa_matrix
from .utils import array_to_grid_sample, homogenise_coordinates

# update shift
PMDL.SHIFT = [PMDL.SHIFT_Y, PMDL.SHIFT_X]


def get_lerp_corner_weights(
    coordinates: torch.Tensor,
    out_shape: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get lerp locations and weights for inserting in 1D."""
    # linearise data and coordinates
    coordinates = coordinates.view(-1).float()

    # only keep data and coordinates inside the image
    in_image_idx = (coordinates >= 0) & (
        coordinates <= torch.tensor(out_shape, device=coordinates.device) - 1
    )
    coordinates = coordinates[in_image_idx]

    # calculate and cache floor and ceil of coordinates for each value to be inserted
    corner_coordinates = torch.empty(
        size=(coordinates.shape[0], 2), dtype=torch.long, device=coordinates.device
    )
    corner_coordinates[:, 0] = torch.floor(coordinates)
    corner_coordinates[:, 1] = torch.ceil(coordinates)

    # calculate linear interpolation weights for each data point being inserted
    weights = torch.empty(
        size=(coordinates.shape[0], 2), device=coordinates.device
    )  # (b, 2,
    # yx)
    weights[:, 1] = coordinates - corner_coordinates[:, 0]  # upper corner weights
    weights[:, 0] = 1 - weights[:, 1]  # lower corner weights

    return corner_coordinates, weights, in_image_idx


def back_and_forth(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    forward_angle: float,
    align_z: int,
) -> torch.Tensor:
    """Tilt series contains tilts that are back and forward projected.

    The forward projection is at 0 degrees tilt, i.e. there could be 5 tilts at angles
    [-12, -8, -4, 4, 8] which means the tilt to where the re-projection is done is
    excluded.
    The tilt_series needs to pre-weighted.
    """
    device = tilt_series.device
    n_tilts, h, w = tilt_series.shape
    projection_dims = (h, w)
    zx_slice_dims = (align_z, w)
    center = torch.tensor(zx_slice_dims) // 2

    s0 = T_2d(-center)
    r0 = R_2d(tilt_angles)
    s1 = T_2d(center)
    M = einops.rearrange(s1 @ r0 @ s0, "... i j -> ... 1 1 i j").to(device)

    # create grid for xz-slice reconstruction
    grid = homogenise_coordinates(coordinate_grid(zx_slice_dims, device=device))
    grid = einops.rearrange(grid, "d w coords -> d w coords 1")
    grid = M @ grid
    grid = einops.rearrange(grid, "... d w coords 1 -> ... d w coords")[
        ..., :2
    ].contiguous()
    grid = array_to_grid_sample(grid, zx_slice_dims)

    # create the grid for projecting the xz-slice forward
    projection = torch.zeros(projection_dims, dtype=torch.float32, device=device)
    M_proj = (s1 @ R_2d(forward_angle) @ s0)[:, 1:2, :]
    M_proj = einops.rearrange(M_proj, "... i j -> ... 1 1 i j").to(device)

    proj_grid = homogenise_coordinates(coordinate_grid(zx_slice_dims, device=device))
    proj_grid = einops.rearrange(proj_grid, "d w coords -> d w coords 1")
    proj_grid = M_proj @ proj_grid
    proj_grid = einops.rearrange(proj_grid, "... d w coords 1 -> ... d w coords")
    corners, weights, valid_ids = get_lerp_corner_weights(proj_grid, w)

    def place_in_image(data: torch.Tensor, image: torch.Tensor) -> None:
        """Utility function for linear interpolation."""
        d = data[valid_ids]
        for x in (0, 1):  # loop over floor and ceil of the coordinates
            w = weights[:, x]
            xc = einops.rearrange(
                corners[
                    :,
                    [
                        x,
                    ],
                ],
                "b x -> x b",
            )
            image.index_put_(indices=(xc,), values=w * d, accumulate=True)

    for y_slice in range(h):
        zx_slice = einops.reduce(
            F.grid_sample(
                einops.rearrange(tilt_series[:, y_slice], "n w -> n 1 1 w"),
                grid,
                align_corners=True,
                mode="bicubic",
            ),
            "n c d w -> d w",
            "mean",
        )
        place_in_image(
            zx_slice.view(-1),  # data
            projection[y_slice],  # image
        )

    return projection


def predict_projection(
    tilt_series: torch.Tensor,
    projection_model: ProjectionModel,
    forward_projection: ProjectionModel,
    align_z: int,
) -> torch.Tensor:
    """Find the projection at the specified model point."""
    # initializes sizes
    device = tilt_series.device
    n_tilts, h, w = tilt_series.shape  # for simplicity assume square images
    tilt_image_dimensions = (h, w)
    filter_size = w

    # generate the 2d alignment affine matrix
    M = projection_model_to_tsa_matrix(
        projection_model,
        tilt_image_dimensions,
        tilt_image_dimensions,
    ).to(device)

    aligned_ts = affine_transform_2d(
        tilt_series,
        M,
        out_shape=tilt_image_dimensions,
    )

    # AreTomo3 code uses a modified hamming window
    # 2 * q * (0.55f + 0.45f * cosf(6.2831852f * q))  # with q from 0 to .5 (Ny)
    # https://github.com/czimaginginstitute/AreTomo3/blob/
    #   c39dcdad9525ee21d7308a95622f3d47fe7ab4b9/AreTomo/Recon/GRWeight.cu#L20
    q = (
        torch.arange(
            filter_size // 2 + filter_size % 2 + 1,
            dtype=torch.float32,
            device=device,
        )
        / filter_size
    )
    # regular hamming: q * (.54 + .46 * torch.cos(torch.pi * q))
    filters = 2 * q * (0.54 + 0.46 * torch.cos(2 * torch.pi * q))
    filters /= filters.max()  # 0-1 normalization
    filters = filters * (1 - 1 / n_tilts) + 1 / n_tilts  # start at 1 / N

    weighted = torch.fft.irfftn(
        torch.fft.rfftn(aligned_ts, dim=(-2, -1)) * filters, dim=(-2, -1)
    )
    if len(weighted.shape) == 2:  # rfftn gets rid of batch dimension: add it back
        weighted = einops.rearrange(weighted, "h w -> 1 h w")

    projection = back_and_forth(
        weighted,
        torch.tensor(projection_model[PMDL.ROTATION_Y].to_numpy()),
        float(forward_projection[PMDL.ROTATION_Y].iloc[0]),
        align_z,
    )

    # generate the 2d alignment affine matrix
    M = torch.linalg.inv(
        projection_model_to_tsa_matrix(
            forward_projection,
            tilt_image_dimensions,
            tilt_image_dimensions,
        )
    ).to(device)

    aligned_projection = affine_transform_2d(
        projection,
        M,
        out_shape=tilt_image_dimensions,
    )
    return aligned_projection


def projection_matching(
    tilt_series: torch.Tensor,
    projection_model_in: ProjectionModel,
    reference_tilt_id: int,
    alignment_mask: torch.Tensor,
    tomogram_dimensions: Tuple[int, int, int],
    reconstruction_weighting: str = "hamming",
    exact_weighting_object_diameter: float | None = None,
) -> Tuple[ProjectionModel, torch.Tensor]:
    """Run projection matching."""
    device = tilt_series.device
    n_tilts, size, _ = tilt_series.shape
    aligned_set = [reference_tilt_id]
    # copy the model to update with new shifts
    projection_model_out = projection_model_in.copy(deep=True)
    tilt_angles = torch.tensor(  # to tensor as we need it to calculate weights
        projection_model_out[PMDL.ROTATION_Y].to_numpy(), dtype=tilt_series.dtype
    )

    # generate indices by alternating postive/negative tilts
    max_offset = max(reference_tilt_id, n_tilts - reference_tilt_id - 1)
    index_sequence = []
    for i in range(1, max_offset + 1):  # skip reference
        if reference_tilt_id + i < n_tilts:
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
        ).to(device)
        projection = predict_projection(
            tilt_series[aligned_set,] * weights[aligned_set,],
            projection_model_out.iloc[aligned_set,],
            projection_model_out.iloc[[i],],
            tomogram_dimensions[0],  # only align z
        )
        # intermediate_recon = filtered_back_projection_3d(
        #     tilt_series[aligned_set,] * weights[aligned_set,],
        #     tomogram_dimensions,
        #     projection_model_out.iloc[aligned_set,],
        #     weighting=reconstruction_weighting,
        #     object_diameter=exact_weighting_object_diameter,
        # )
        # projection, projection_weights = tomogram_reprojection(
        #     intermediate_recon,
        #     (size, size),
        #     projection_model_out.iloc[[i],],
        # )

        # ensure correlation in relevant area
        # projection_weights = projection_weights / projection_weights.max()
        projection_weights = alignment_mask
        projection *= projection_weights
        projection = (projection - projection.mean()) / projection.std()
        raw = tilt_series[i] * projection_weights
        raw = (raw - raw.mean()) / raw.std()
        shift = find_image_shift(
            raw,
            projection,
        )
        projection_model_out.loc[i, PMDL.SHIFT] += shift.numpy()
        aligned_set.append(i)

        # for debugging:
        projections[i] = projection.detach().cpu()

    return projection_model_out, projections

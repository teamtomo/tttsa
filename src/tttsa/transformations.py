"""4x4 matrices for rotations and translations.

Functions in this module generate matrices which left-multiply column vectors containing
`xyzw` or `zyxw` homogeneous coordinates.
"""

from typing import Tuple

import einops
import torch
import torch.nn.functional as F
from cryotypes.projectionmodel import ProjectionModel
from cryotypes.projectionmodel import ProjectionModelDataLabels as PMDL

# update shift
PMDL.SHIFT = [PMDL.SHIFT_Y, PMDL.SHIFT_X]


def Rx(angles_degrees: torch.Tensor, zyx: bool = False) -> torch.Tensor:
    """4x4 matrices for a rotation of homogeneous coordinates around the X-axis.

    Parameters
    ----------
    angles_degrees: torch.Tensor
        `(..., )` array of angles
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogeneous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    angles_degrees = torch.atleast_1d(torch.as_tensor(angles_degrees))
    angles_packed, ps = einops.pack([angles_degrees], pattern="*")  # to 1d
    n = angles_packed.shape[0]
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), "i j -> n i j", n=n).clone()
    matrices[:, 1, 1] = c
    matrices[:, 1, 2] = -s
    matrices[:, 2, 1] = s
    matrices[:, 2, 2] = c
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern="* i j")
    return matrices


def Ry(angles_degrees: torch.Tensor, zyx: bool = False) -> torch.Tensor:
    """4x4 matrices for a rotation of homogeneous coordinates around the Y-axis.

    Parameters
    ----------
    angles_degrees: torch.Tensor
        `(..., )` array of angles
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogeneous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    angles_degrees = torch.atleast_1d(torch.as_tensor(angles_degrees))
    angles_packed, ps = einops.pack([angles_degrees], pattern="*")  # to 1d
    n = angles_packed.shape[0]
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), "i j -> n i j", n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 2] = s
    matrices[:, 2, 0] = -s
    matrices[:, 2, 2] = c
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern="* i j")
    return matrices


def Rz(angles_degrees: torch.Tensor, zyx: bool = False) -> torch.Tensor:
    """4x4 matrices for a rotation of homogeneous coordinates around the Z-axis.

    Parameters
    ----------
    angles_degrees: torch.Tensor
        `(..., )` array of angles
    zyx: bool
        Whether output should be compatible with `zyxw` (`True`) or `xyzw`
        (`False`) homogeneous coordinates.

    Returns
    -------
    matrices: `(..., 4, 4)` array of 4x4 rotation matrices.
    """
    angles_degrees = torch.atleast_1d(torch.as_tensor(angles_degrees))
    angles_packed, ps = einops.pack([angles_degrees], pattern="*")  # to 1d
    n = angles_packed.shape[0]
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(4), "i j -> n i j", n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    if zyx is True:
        matrices[:, :3, :3] = torch.flip(matrices[:, :3, :3], dims=(-2, -1))
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern="* i j")
    return matrices


def T(shifts: torch.Tensor) -> torch.Tensor:
    """4x4 matrices for translations.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 3)` array of shifts.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 4, 4)` array of 4x4 shift matrices.
    """
    shifts = torch.atleast_1d(torch.as_tensor(shifts))
    shifts, ps = einops.pack([shifts], pattern="* coords")  # to 2d
    n = shifts.shape[0]
    matrices = einops.repeat(torch.eye(4), "i j -> n i j", n=n).clone()
    matrices[:, :3, 3] = shifts
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern="* i j")
    return matrices


def S(scale_factors: torch.Tensor) -> torch.Tensor:
    """4x4 matrices for scaling.

    Parameters
    ----------
    scale_factors: torch.Tensor
        `(..., 3)` array of scale factors.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 4, 4)` array of 4x4 shift matrices.
    """
    scale_factors = torch.atleast_1d(torch.as_tensor(scale_factors))
    scale_factors, ps = einops.pack([scale_factors], pattern="* coords")  # to 2d
    n = scale_factors.shape[0]
    matrices = einops.repeat(torch.eye(4), "i j -> n i j", n=n).clone()
    matrices[:, [0, 1, 2], [0, 1, 2]] = scale_factors
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern="* i j")
    return matrices


# Matrices for 2D transformations


def R_2d(angles_degrees: torch.Tensor, yx: bool = False) -> torch.Tensor:
    """3x3 matrices for a rotation of homogeneous coordinates around the X-axis.

    Parameters
    ----------
    angles_degrees: torch.Tensor
        `(..., )` array of angles
    yx: bool
        Whether output should be compatible with `yxw` (`True`) or `xyw`
        (`False`) homogeneous coordinates.

    Returns
    -------
    matrices: `(..., 3, 3)` array of 3x3 rotation matrices.
    """
    angles_degrees = torch.atleast_1d(torch.as_tensor(angles_degrees))
    angles_packed, ps = einops.pack([angles_degrees], pattern="*")  # to 1d
    n = angles_packed.shape[0]
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)
    matrices = einops.repeat(torch.eye(3), "i j -> n i j", n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c
    if yx is True:
        matrices[:, :2, :2] = torch.flip(matrices[:, :2, :2], dims=(-2, -1))
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern="* i j")
    return matrices


def T_2d(shifts: torch.Tensor) -> torch.Tensor:
    """3x3 matrices for translations.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 2)` array of shifts.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 3, 3)` array of 3x3 shift matrices.
    """
    shifts = torch.atleast_1d(torch.as_tensor(shifts))
    shifts, ps = einops.pack([shifts], pattern="* coords")  # to 2d
    n = shifts.shape[0]
    matrices = einops.repeat(torch.eye(3), "i j -> n i j", n=n).clone()
    matrices[:, :2, 2] = shifts
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern="* i j")
    return matrices


def S_2d(scale_factors: torch.Tensor) -> torch.Tensor:
    """3x3 matrices for scaling.

    Parameters
    ----------
    scale_factors: torch.Tensor
        `(..., 2)` array of scale factors.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 3, 3)` array of 3x3 shift matrices.
    """
    scale_factors = torch.atleast_1d(torch.as_tensor(scale_factors))
    scale_factors, ps = einops.pack([scale_factors], pattern="* coords")  # to 2d
    n = scale_factors.shape[0]
    matrices = einops.repeat(torch.eye(3), "i j -> n i j", n=n).clone()
    matrices[:, [0, 1], [0, 1]] = scale_factors
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern="* i j")
    return matrices


def stretch_matrix(
    tilt_image_dimensions: Tuple[int, int],
    tilt_axis_angle: torch.Tensor,
    scale_factor: torch.Tensor,
) -> torch.Tensor:
    """Calculate a tilt-image stretch matrix for coarse alignment."""
    image_center = torch.tensor(tilt_image_dimensions) // 2
    s0 = T_2d(-image_center)
    r_forward = R_2d(tilt_axis_angle, yx=True)
    r_backward = torch.linalg.inv(r_forward)
    m_stretch = torch.eye(3)
    m_stretch[1, 1] = scale_factor  # this is a shear matrix
    s1 = T_2d(image_center)
    return s1 @ r_forward @ m_stretch @ r_backward @ s0


def projection_model_to_projection_matrix(
    projection_model: ProjectionModel,
    tilt_image_dimensions: Tuple[int, int],
    tomogram_dimensions: Tuple[int, int, int],
) -> torch.Tensor:
    """Convert a cryotypes ProjectionModel to a projection matrix."""
    tilt_image_center = (
        torch.tensor((int(tomogram_dimensions[0]), *tilt_image_dimensions)) // 2
    )
    tomogram_center = torch.tensor(tomogram_dimensions) // 2
    s0 = T(-tilt_image_center)
    r0 = Rx(torch.tensor(projection_model[PMDL.ROTATION_X].to_numpy()), zyx=True)
    r1 = Ry(torch.tensor(projection_model[PMDL.ROTATION_Y].to_numpy()), zyx=True)
    r2 = Rz(torch.tensor(projection_model[PMDL.ROTATION_Z].to_numpy()), zyx=True)
    s1 = T(
        F.pad(
            torch.tensor(projection_model[PMDL.SHIFT].to_numpy()), pad=(1, 0), value=0
        )
    )
    s2 = T(tomogram_center)
    return s2 @ s1 @ r2 @ r1 @ r0 @ s0


def projection_model_to_tsa_matrix(
    projection_model: ProjectionModel,
    tilt_image_dimensions: Tuple[int, int],
    projected_tomogram_dimensions: Tuple[int, int],
) -> torch.Tensor:
    """Convert cryotypes ProjectionModel to a 2D tilt-series alignment matrix."""
    tilt_image_center = torch.tensor(tilt_image_dimensions) // 2
    projected_tomogram_center = torch.tensor(projected_tomogram_dimensions) // 2
    s0 = T_2d(-tilt_image_center)
    r0 = R_2d(torch.tensor(projection_model[PMDL.ROTATION_Z].to_numpy()), yx=True)
    s1 = T_2d(torch.tensor(projection_model[PMDL.SHIFT].to_numpy()))
    s2 = T_2d(projected_tomogram_center)
    # invert for forward alignment and reconstruction
    return torch.linalg.inv(s2 @ s1 @ r0 @ s0)


def projection_model_to_backproject_matrix(
    projection_model: ProjectionModel,
    tomogram_dimensions: Tuple[int, int, int],
) -> torch.Tensor:
    """Convert cryotypes ProjectionModel to a backprojection matrix."""
    tomogram_center = torch.tensor(tomogram_dimensions) // 2
    s0 = T(-tomogram_center)
    r0 = Rx(torch.tensor(projection_model[PMDL.ROTATION_X].to_numpy()), zyx=True)
    r1 = Ry(torch.tensor(projection_model[PMDL.ROTATION_Y].to_numpy()), zyx=True)
    s1 = T(tomogram_center)
    # invert for forward alignment and reconstruction
    return torch.linalg.inv(s1 @ r1 @ r0 @ s0)

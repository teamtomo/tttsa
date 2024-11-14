import pytest
import torch

from tttsa.affine import affine_transform_2d, affine_transform_3d
from tttsa.transformations import R_2d, Rz


def test_affine_transform_2d():
    m1 = R_2d(torch.tensor(45.0))
    m2 = R_2d(torch.randn(3))

    # with a single image
    a = torch.zeros((4, 5))
    b = affine_transform_2d(a, m1)
    assert a.shape == b.shape
    b = affine_transform_2d(a, m1, (5, 4))
    assert b.shape == (5, 4)
    b = affine_transform_2d(a, m2)
    assert b.shape == (3, 4, 5)
    b = affine_transform_2d(a, m2, (5, 4))
    assert b.shape == (3, 5, 4)

    # with a batch of images
    a = torch.zeros((3, 4, 5))
    b = affine_transform_2d(a, m2)
    assert a.shape == b.shape
    b = affine_transform_2d(a, m2, (5, 4))
    assert b.shape == (3, 5, 4)
    a = torch.zeros((2, 4, 5))
    with pytest.raises(ValueError):
        affine_transform_2d(a, m1)
    with pytest.raises(ValueError):
        affine_transform_2d(a, m2)


def test_affine_transform_3d():
    m1 = Rz(torch.tensor(45.0))
    m2 = Rz(torch.randn(3))

    # with a single image
    a = torch.zeros((3, 4, 5))
    b = affine_transform_3d(a, m1)
    assert a.shape == b.shape
    b = affine_transform_3d(a, m1, (5, 4, 3))
    assert b.shape == (5, 4, 3)
    b = affine_transform_3d(a, m2)
    assert b.shape == (3, 3, 4, 5)
    b = affine_transform_3d(a, m2, (5, 4, 3))
    assert b.shape == (3, 5, 4, 3)

    # with a batch of images
    a = torch.zeros((3, 3, 4, 5))
    b = affine_transform_3d(a, m2)
    assert a.shape == b.shape
    b = affine_transform_3d(a, m2, (5, 4, 3))
    assert b.shape == (3, 5, 4, 3)
    a = torch.zeros((2, 3, 4, 5))
    with pytest.raises(ValueError):
        affine_transform_3d(a, m1)
    with pytest.raises(ValueError):
        affine_transform_3d(a, m2)

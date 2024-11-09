import pytest
import torch

from tttsa.affine import affine_transform_2d, stretch_image
from tttsa.transformations import R_2d


def test_stretch_image():
    a = torch.zeros((5, 5))
    b = stretch_image(a, 1.1, -85)
    assert a.shape == b.shape


def test_affine_transform_2d():
    a = torch.zeros((4, 5))
    m1 = R_2d(torch.tensor(45.0))
    b = affine_transform_2d(a, m1)
    assert a.shape == b.shape
    b = affine_transform_2d(a, m1, (5, 4))
    assert b.shape == (5, 4)
    m2 = R_2d(torch.randn(3))
    b = affine_transform_2d(a, m2)
    assert b.shape == (3, 4, 5)
    b = affine_transform_2d(a, m2, (5, 4))
    assert b.shape == (3, 5, 4)
    a = torch.zeros((3, 4, 5))
    b = affine_transform_2d(a, m2)
    assert a.shape == b.shape
    a = torch.zeros((2, 4, 5))
    b = affine_transform_2d(a, m1)
    assert a.shape == b.shape
    with pytest.raises(RuntimeError):
        affine_transform_2d(a, m2)


def test_affine_transform_3d():
    pass

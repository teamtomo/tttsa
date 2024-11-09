import pytest
import torch

from tttsa.alignment import find_image_shift


def test_find_image_shift():
    a = torch.zeros((4, 4))
    a[0, 0] = 1
    b = torch.zeros((4, 4))
    b[2, 2] = 0.7
    b[2, 3] = 0.3
    shift = find_image_shift(a, b)
    print(shift)
    assert shift.dtype == torch.float32
    assert torch.all(shift == -2.0), (
        "Interpolating a shift too close to a border is "
        "not possible, so an integer shift should be "
        "returned."
    )
    a = torch.zeros((8, 8))
    a[3, 3] = 1
    b = torch.zeros((8, 8))
    b[4, 4] = 0.7
    b[4, 5] = 0.3
    shift = find_image_shift(a, b)
    # values should interpolated with floating point precision
    assert shift.dtype == torch.float32
    assert shift[0] == pytest.approx(-1.1, 0.1)
    assert shift[1] == pytest.approx(-1.2, 0.1)

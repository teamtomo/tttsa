import torch

from tttsa.correlation import correlate_2d


def test_correlate_2d():
    a = torch.zeros((10, 10))
    a[5, 5] = 1
    b = torch.zeros((10, 10))
    b[6, 6] = 1
    cross_correlation = correlate_2d(a, b)
    peak_position = torch.unravel_index(
        indices=torch.argmax(cross_correlation), shape=cross_correlation.shape
    )
    shift = torch.as_tensor(peak_position) - torch.tensor([5, 5])
    assert peak_position == (4, 4)
    assert torch.allclose(shift, torch.tensor([-1, -1]))
    assert cross_correlation.shape == a.shape


def test_correlate_2d_stacks():
    # test for stacks of images
    a = torch.zeros((3, 10, 11))
    a[:, 5, 5] = 1
    b = torch.zeros((1, 10, 11))
    b[:, 6, 6] = 1
    cross_correlation = correlate_2d(a, b)
    assert cross_correlation.shape == a.shape
    cross_correlation = correlate_2d(b, a)
    assert cross_correlation.shape == a.shape
    cross_correlation = correlate_2d(a, a)
    assert cross_correlation.shape == a.shape

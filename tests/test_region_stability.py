import torch

from sparse2unseen.confidence.region_stability import region_counts, stability_weights


def test_region_counts_preserve_total_count():
    density = torch.ones(2, 1, 8, 8)
    counts = region_counts(density, (4, 4))
    assert torch.allclose(counts.sum(dim=(-2, -1)), density.sum(dim=(-2, -1, -3)))


def test_identical_views_have_unit_weight():
    pred = torch.ones(4, 2, 1, 8, 8)
    weights = stability_weights(pred, grid=(4, 4), beta=4.0)
    assert weights.shape == (2, 1, 8, 8)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_unstable_region_is_downweighted():
    pred = torch.ones(4, 1, 1, 8, 8)
    pred[0, :, :, :4, :4] *= 3
    weights = stability_weights(pred, grid=(2, 2), beta=0.1)
    assert weights[0, 0, 1, 1] < weights[0, 0, 6, 6]

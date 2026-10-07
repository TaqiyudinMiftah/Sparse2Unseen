from __future__ import annotations

from collections import Counter
import copy
import random

import numpy as np
import pytest
import torch

from sparse2unseen.experiments.sampling import BalancedEpochSampler
from sparse2unseen.experiments.training import capture_rng, restore_rng, supervised_loss, train_epoch, ema_decay_for_step
from sparse2unseen.confidence.region_stability import region_counts
from sparse2unseen.losses import resize_density_preserve_count
from sparse2unseen.ssl.ema import make_teacher


def test_uneven_exposure_budget_covers_every_image_without_skew():
    generator = torch.Generator().manual_seed(3)
    sampler = BalancedEpochSampler(128, 320, generator)
    indices = list(sampler)
    assert len(indices) == 320
    counts = Counter(indices)
    assert set(counts) == set(range(128))
    assert set(counts.values()) == {2, 3}


def test_sampler_and_all_rng_states_resume_the_same_epoch():
    generators = {"labeled": torch.Generator().manual_seed(1)}
    sampler = BalancedEpochSampler(7, 16, generators["labeled"])
    list(sampler)
    state = capture_rng(generators)
    expected = (list(sampler), random.random(), np.random.rand(), torch.rand(4))
    restore_rng(state, generators)
    actual = (list(sampler), random.random(), np.random.rand(), torch.rand(4))
    assert actual[:3] == expected[:3]
    assert torch.equal(actual[3], expected[3])


def test_reduction_preserves_single_point_at_a_missed_bilinear_location():
    target = torch.zeros(1, 1, 320, 320)
    target[0, 0, 0, 0] = 1
    reduced = resize_density_preserve_count(target, (20, 20))
    assert reduced.sum() == pytest.approx(1.0)


def test_region_counts_are_disjoint_at_nondivisible_resolution():
    density = torch.arange(35).reshape(1, 1, 5, 7).float()
    assert region_counts(density, (3, 4)).sum() == density.sum()


def test_stability_weight_batch_axis_is_not_the_number_of_views():
    from sparse2unseen.confidence.region_stability import stability_weights
    predictions = torch.ones(4, 3, 1, 5, 7)
    weights = stability_weights(predictions, grid=(3, 4))
    assert weights.shape == (3, 1, 5, 7)
    assert torch.equal(weights, torch.ones_like(weights))


class ToyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, image):
        return self.weight * image


class ToyFinal(ToyBase):
    def forward_train(self, image1, image2, regions):
        probability = torch.sigmoid(self.weight) * torch.ones_like(regions)
        return (image1*self.weight, image2*self.weight, probability, probability,
                None, 0.2*self.weight.square(), 0)

    def forward(self, image):
        return super().forward(image), torch.sigmoid(image*self.weight)


def batch(value):
    image = torch.full((2, 1, 2, 2), value)
    return image, image + 0.5, ([], torch.ones_like(image), torch.ones_like(image))


def spec(method, samples=8, updates=1):
    return {"method": method, "samples_per_epoch": samples, "updates_per_epoch": updates,
            "settings": {"warmup_epochs": 0, "accumulation_steps": 4, "log_para": 1,
                         "ramp_epochs": 1, "unsupervised_weight": 1, "ema_decay": 0.9,
                         "teacher_views": 4, "region_grid": [2, 2], "stability_beta": 4}}


def test_label_only_accumulation_matches_an_explicit_average():
    model, reference = ToyBase(), ToyBase()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    batches = [batch(float(value)) for value in (1, 2, 3, 4)]
    reference_optimizer.zero_grad()
    sum(supervised_loss(reference, "label_only", item, "cpu", 1) for item in batches).div(4).backward()
    reference_optimizer.step()
    metrics = train_epoch(model, None, optimizer, batches, None, spec("label_only"), 0, "cpu")
    assert torch.allclose(model.weight, reference.weight)
    assert metrics["optimizer_updates"] == 1
    assert metrics["unlabeled_samples"] == 0


def test_mpcount_objective_matches_the_actual_upstream_training_step():
    from sparse2unseen.experiments.inference import upstream
    upstream()
    from trainers.dgtrainer import DGTrainer
    native = object.__new__(DGTrainer)
    native.device, native.mode, native.log_para = "cpu", "final", 1
    model, reference = ToyFinal(), ToyFinal()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    item = batch(1.0)
    loss = supervised_loss(model, "mpcount", item, "cpu", 1)
    loss.backward()
    optimizer.step()
    native_loss = native.train_step(reference, torch.nn.MSELoss(), reference_optimizer, item, 0)
    assert float(loss) == pytest.approx(native_loss)
    assert torch.equal(model.weight, reference.weight)


@pytest.mark.parametrize("method,expected_calls", [("ssl_dg", 0), ("domain_stable", 1)])
def test_naive_ssl_and_stability_method_are_distinct(monkeypatch, method, expected_calls):
    import sparse2unseen.experiments.training as training
    calls = []

    def weights(views, grid, beta):
        calls.append(views.shape)
        return torch.ones_like(views[0])

    monkeypatch.setattr(training, "stability_weights", weights)
    monkeypatch.setattr(training, "DomainDiversifier", lambda cfg: lambda images: images)
    model = ToyFinal()
    teacher = make_teacher(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    unlabeled = [(torch.ones(2, 1, 2, 2), torch.ones(2, 1, 2, 2)*2, [])]
    train_epoch(model, teacher, optimizer, [batch(1.0)], unlabeled, spec(method, samples=2), 0, "cpu")
    assert len(calls) == expected_calls
    if calls:
        assert calls[0] == (4, 2, 1, 2, 2)


def test_ssl_has_explicit_unlabeled_budget_and_teacher_has_no_gradients():
    model = ToyBase()
    teacher = make_teacher(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batches = [batch(float(value)) for value in (1, 2, 3, 4)]
    unlabeled = [(torch.ones(2, 1, 2, 2), torch.ones(2, 1, 2, 2)*2, ["u1", "u2"])] * 4
    metrics = train_epoch(model, teacher, optimizer, batches, unlabeled, spec("mean_teacher"), 0, "cpu")
    assert metrics["unlabeled_samples"] == 8
    assert metrics["optimizer_updates"] == 1
    assert metrics["unsupervised_loss"] > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert torch.allclose(teacher.weight, torch.tensor(0.4)*0.5 + model.weight*0.5)


def test_ema_startup_matches_the_original_mean_teacher_rule():
    assert ema_decay_for_step(0.999, 1) == 0.5
    assert ema_decay_for_step(0.999, 20) == pytest.approx(20/21)
    assert ema_decay_for_step(0.999, 1000) == 0.999


def test_warmup_does_not_fetch_unlabeled_annotations_or_images():
    class Unused:
        def __iter__(self):
            pytest.fail("Unlabeled loader must remain unused during warmup")

    model = ToyBase()
    teacher = make_teacher(model)
    recipe = spec("mean_teacher", samples=2)
    recipe["settings"]["warmup_epochs"] = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_epoch(model, teacher, optimizer, [batch(1.0)], Unused(), recipe, 0, "cpu")
    assert result["unlabeled_samples"] == 0


def test_interrupted_optimizer_teacher_and_sampler_resume_match_continuous_training(tmp_path):
    model = ToyBase()
    teacher = make_teacher(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generators = {"labeled": torch.Generator().manual_seed(3)}
    sampler = BalancedEpochSampler(4, 8, generators["labeled"])

    def one_epoch():
        values = list(sampler)
        batches = [batch(float(value)+1) for value in values[::2]]
        unlabeled = [(torch.rand(2, 1, 2, 2), torch.rand(2, 1, 2, 2), []) for _ in batches]
        train_epoch(model, teacher, optimizer, batches, unlabeled, spec("mean_teacher"), 0, "cpu")

    one_epoch()
    state = copy.deepcopy({"model": model.state_dict(), "teacher": teacher.state_dict(),
                           "optimizer": optimizer.state_dict(), "rng": capture_rng(generators)})
    one_epoch()
    expected_model, expected_teacher = model.weight.clone(), teacher.weight.clone()
    model.load_state_dict(state["model"])
    teacher.load_state_dict(state["teacher"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], generators)
    one_epoch()
    assert torch.equal(model.weight, expected_model)
    assert torch.equal(teacher.weight, expected_teacher)

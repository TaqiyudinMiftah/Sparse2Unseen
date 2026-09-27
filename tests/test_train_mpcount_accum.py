from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_mpcount_accum.py"
spec = importlib.util.spec_from_file_location("train_mpcount_accum", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ToyFinal(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))

    def forward_train(self, img1, img2, gt_cmaps):
        cmap = torch.sigmoid(torch.ones_like(gt_cmaps) * self.weight)
        loss_con = 0.2 * self.weight.square()
        return img1 * self.weight, img2 * self.weight, cmap, cmap, None, loss_con, 0

    def forward(self, img):
        return img * self.weight, torch.sigmoid(img * self.weight)


def make_batch(value: float):
    img1 = torch.full((2, 1, 2, 2), value)
    img2 = torch.full((2, 1, 2, 2), value + 0.5)
    dmap = torch.zeros_like(img1)
    cmap = torch.zeros_like(img1)
    return img1, img2, ([], dmap, cmap)


def new_trainer(tmp_path: Path, monkeypatch, name: str, steps: int):
    monkeypatch.chdir(tmp_path)
    return module.AccumDGTrainer(
        seed=1, version=name, device="cpu", log_para=1,
        patch_size=10000, mode="final", accumulation_steps=steps
    )


def run_accumulated(trainer, batches):
    model = ToyFinal()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    trainer.configure_epoch(len(batches))
    initial = model.weight.item()
    for index, batch in enumerate(batches):
        trainer.train_step(model, torch.nn.MSELoss(), optimizer, batch, epoch=0)
        if index < trainer.accumulation_steps - 1:
            assert model.weight.item() == initial
    return model, trainer.optimizer_steps


def run_full_batches(batches, group_size: int):
    model = ToyFinal()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for start in range(0, len(batches), group_size):
        group = batches[start : start + group_size]
        img1 = torch.cat([batch[0] for batch in group])
        img2 = torch.cat([batch[1] for batch in group])
        dmap = torch.cat([batch[2][1] for batch in group])
        cmap = torch.cat([batch[2][2] for batch in group])
        pred1, pred2, cls1, cls2, _, loss_con, _ = model.forward_train(
            img1, img2, cmap
        )
        raw_loss = (
            torch.nn.functional.mse_loss(pred1, dmap)
            + torch.nn.functional.mse_loss(pred2, dmap)
            + 10 * torch.nn.functional.binary_cross_entropy(cls1, cmap)
            + 10 * torch.nn.functional.binary_cross_entropy(cls2, cmap)
            + 10 * loss_con
        )
        optimizer.zero_grad()
        raw_loss.backward()
        optimizer.step()
    return model


def test_accumulation_matches_full_batch_without_batchnorm(tmp_path, monkeypatch):
    batches = [make_batch(value) for value in (1.0, 2.0, 3.0, 4.0)]
    trainer = new_trainer(tmp_path, monkeypatch, "accum4", steps=4)
    actual, steps = run_accumulated(trainer, batches)
    expected = run_full_batches(batches, 4)
    assert steps == 1
    assert torch.allclose(actual.weight, expected.weight, atol=1e-6)


def test_partial_group_is_scaled_by_its_actual_size(tmp_path, monkeypatch):
    batches = [make_batch(float(value)) for value in range(1, 6)]
    trainer = new_trainer(tmp_path, monkeypatch, "remainder", steps=4)
    actual, steps = run_accumulated(trainer, batches)
    expected = run_full_batches(batches, 4)
    assert steps == 2
    assert torch.allclose(actual.weight, expected.weight, atol=1e-6)


def test_one_step_matches_unmodified_upstream_final_mode(tmp_path, monkeypatch):
    batch = make_batch(2.0)
    adapted = new_trainer(tmp_path, monkeypatch, "adapted", steps=1)
    upstream = module.DGTrainer(
        seed=1, version="upstream", device="cpu", log_para=1,
        patch_size=10000, mode="final"
    )
    adapted_model = ToyFinal()
    upstream_model = ToyFinal()
    adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.05)
    upstream_optimizer = torch.optim.SGD(upstream_model.parameters(), lr=0.05)
    adapted.configure_epoch(1)
    adapted_loss = adapted.train_step(
        adapted_model, torch.nn.MSELoss(), adapted_optimizer, batch, epoch=0
    )
    upstream_loss = upstream.train_step(
        upstream_model, torch.nn.MSELoss(), upstream_optimizer, batch, epoch=0
    )
    assert abs(adapted_loss - upstream_loss) < 1e-6
    assert torch.allclose(adapted_model.weight, upstream_model.weight, atol=1e-6)


def test_primary_config_is_source_only_and_twenty_updates():
    config = yaml.safe_load(
        (SCRIPT.parents[1] / "configs/mpcount/stb_100_train_accum4.yml").read_text()
    )
    assert config["train_dataset"]["params"]["root"] == "data/stb"
    assert config["val_dataset"]["params"]["root"] == "data/stb"
    assert config["model"]["params"]["deterministic"] is True
    assert config["train_loader"]["batch_size"] == 4
    assert config["accumulation_steps"] == 4
    assert 320 // (config["train_loader"]["batch_size"] * config["accumulation_steps"]) == 20


def test_balanced_repeats_keep_twenty_updates_and_all_labeled_images():
    dataset = torch.utils.data.TensorDataset(torch.arange(32))
    options = {"batch_size": 4, "num_workers": 0, "shuffle": True, "pin_memory": False}
    original = torch.utils.data.DataLoader(dataset, **options)
    repeated = module.balanced_repeat_loader(
        original, samples_per_epoch=320, seed=1, options=options
    )
    assert len(repeated) == 80
    assert len(repeated) // 4 == 20
    sampled = Counter(index for batch in repeated for index in batch[0].tolist())
    assert sampled == {index: 10 for index in range(32)}


def test_balanced_repeats_require_an_exact_multiple():
    dataset = torch.utils.data.TensorDataset(torch.arange(32))
    options = {"batch_size": 4, "num_workers": 0, "shuffle": True}
    original = torch.utils.data.DataLoader(dataset, **options)
    with pytest.raises(ValueError, match="multiple"):
        module.balanced_repeat_loader(
            original, samples_per_epoch=319, seed=1, options=options
        )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sparse_config_has_only_source_data_and_matched_update_budget(seed):
    path = SCRIPT.parents[1] / f"configs/mpcount/stb_10_train_accum4_seed{seed}.yml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["seed"] == seed
    assert config["expected_train_images"] == 32
    assert config["expected_val_images"] == 80
    assert config["train_samples_per_epoch"] == 320
    assert config["expected_updates_per_epoch"] == 20
    assert config["train_dataset"]["params"]["root"] == f"data/stb_sparse10_seed{seed}"
    assert config["val_dataset"]["params"]["root"] == f"data/stb_sparse10_seed{seed}"
    assert config["accumulation_steps"] == 4
    assert config["train_loader"]["batch_size"] == 4
    assert config["model"]["params"]["deterministic"] is True


def test_wandb_logs_source_validation_after_epoch(tmp_path, monkeypatch):
    class FakeRun:
        def __init__(self):
            self.entries = []
            self.summary = {}

        def log(self, values, *, step, commit):
            self.entries.append((values, step, commit))

    fake_run = FakeRun()
    trainer = new_trainer(tmp_path, monkeypatch, "wandb_test", steps=4)
    trainer.wandb_run = fake_run
    model = ToyFinal()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    val_img = torch.ones((1, 1, 2, 2))
    val_batch = (val_img, val_img, torch.zeros((1, 0, 2)), "image", (0, 0, 0, 0))
    trainer.train_epoch(
        model, torch.nn.MSELoss(), [make_batch(1.0)], [val_batch],
        optimizer, None, epoch=0, best_criterion=1e10, best_epoch=-1
    )
    assert len(fake_run.entries) == 1
    values, step, commit = fake_run.entries[0]
    assert step == 0 and commit is True
    assert values["optimizer/updates_per_epoch"] == 1
    assert values["train/mean_loss"] == pytest.approx(
        values["train/last_batch_loss"], abs=1e-4
    )
    assert values["source_val/mae"] >= 0
    assert values["source_val/rmse"] + 1e-4 >= values["source_val/mae"]
    assert fake_run.summary["best_source_val_epoch"] == 0

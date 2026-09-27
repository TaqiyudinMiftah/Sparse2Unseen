from __future__ import annotations

import pytest

from sparse2unseen.monitoring.mpcount_log import (
    latest_training_section,
    parse_completed_epochs,
)


def test_only_completed_current_run_epochs_are_streamed() -> None:
    text = (
        "Start training at old\n"
        "Epoch 0: Training loss: 99.0000 Version: old\n"
        "Epoch 0: Val criterion: 99.0000 mse: 9999.0000 best: 100.0000, time: 1.0\n"
        "Epoch 0: accumulation=4, microbatches=80, optimizer steps=20\n"
        "End training at old\n"
        "Start training at new\n"
        "Epoch 0: Training loss: 8.0000 Version: new\n"
        "Epoch 0: Val criterion: 7.5000 mse: 144.0000 best: 10.0000, time: 1.0\n"
        "Epoch 0: accumulation=4, microbatches=80, optimizer steps=20\n"
        "Epoch 1: Training loss: 6.0000 Version: new\n"
        "Epoch 1: Val criterion: 7.0000 mse: 121.0000 best: 7.5000, time: 1.0\n"
    )
    records = parse_completed_epochs(text)
    assert len(records) == 1
    assert records[0].epoch == 0
    assert records[0].source_val_mae == 7.5
    assert records[0].wandb_values()["source_val/rmse"] == 12.0
    assert records[0].wandb_values()["optimizer/updates_per_epoch"] == 20


def test_nonfinite_completed_epoch_is_rejected() -> None:
    text = (
        "Start training at now\n"
        "Epoch 0: Training loss: 8.0 Version: run\n"
        "Epoch 0: Val criterion: 7.0 mse: 1e309 best: 8.0, time: 1.0\n"
        "Epoch 0: accumulation=4, microbatches=80, optimizer steps=20\n"
    )
    with pytest.raises(ValueError, match="Non-finite"):
        parse_completed_epochs(text)


def test_previous_run_completion_does_not_end_current_run() -> None:
    log = "Start training at old\nEnd training at old\nStart training at new\n"
    assert "End training at " not in latest_training_section(log)

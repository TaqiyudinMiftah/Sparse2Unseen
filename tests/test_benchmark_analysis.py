from __future__ import annotations

import copy
import statistics

import pytest

from sparse2unseen.benchmark_analysis import analyze, expected_keys


SETTINGS = {"sources": ["stb", "sta", "qnrf"], "fractions": [0.1], "seeds": [1, 2, 3],
            "methods": ["label_only", "mean_teacher", "mpcount", "ssl_dg", "domain_stable"],
            "full_label_methods": ["label_only", "mpcount"], "full_label_seed": 2023}


def result(method, seed, *, fraction=0.1, values=(10, 100, 200), source="stb"):
    return {"source": source, "fraction": fraction, "method": method, "seed": seed,
            "selection_domain": source + "_val",
            "domains": {domain: {"mae": value, "rmse": 2 * value}
                        for domain, value in zip(SETTINGS["sources"], values)}}


def test_full_matrix_size_matches_primary_protocol():
    settings = {**SETTINGS, "fractions": [0.05, 0.1, 0.4]}
    assert len(expected_keys(settings)) == 141


def test_partial_seeds_never_enter_statistics_or_contrasts():
    data = analyze([result("label_only", 1), result("mean_teacher", 1)], SETTINGS)
    assert not data["complete"]
    assert not data["contrasts"]
    assert not data["degradation"]
    assert all(not group["domains"] for group in data["groups"])
    with pytest.raises(ValueError, match="Conclusion gate"):
        analyze([result("label_only", 1)], SETTINGS, require_complete=True)


def test_sample_sd_and_rmse_are_seed_level_not_pooled():
    rows = [result("mpcount", seed, values=(value, value * 10, value * 20))
            for seed, value in enumerate([8, 10, 12], start=1)]
    data = analyze(rows, SETTINGS)
    group = next(row for row in data["groups"] if row["source"] == "stb" and row["method"] == "mpcount" and row["fraction"] < 1)
    assert group["domains"]["stb"]["mae"] == {"mean": 10, "sample_sd": 2, "n_seeds": 3}
    assert group["domains"]["stb"]["rmse"]["mean"] == 20
    assert group["domains"]["stb"]["rmse"]["sample_sd"] == 4


def test_relative_method_effects_are_computed_per_matched_seed():
    rows = [result("label_only", seed, values=(value, 100, 200))
            for seed, value in enumerate([10, 20, 40], start=1)]
    rows += [result("mean_teacher", seed, values=(5, 100, 200)) for seed in [1, 2, 3]]
    data = analyze(rows, SETTINGS)
    contrast = data["contrasts"][0]
    summary = contrast["domains"]["stb"]["mae"]["relative_reduction_pct"]
    assert summary["mean"] == pytest.approx(statistics.mean([50, 75, 87.5]))
    assert summary["sample_sd"] == pytest.approx(statistics.stdev([50, 75, 87.5]))
    assert summary["mean"] != pytest.approx((1 - 5 / statistics.mean([10, 20, 40])) * 100)
    assert contrast["seeds"] == [1, 2, 3]


def test_disproportionate_degradation_can_contradict_hypothesis():
    rows = [result("mpcount", 2023, fraction=1, values=(10, 100, 200))]
    rows += [result("mpcount", seed, values=(12, 105, 210)) for seed in [1, 2, 3]]
    data = analyze(rows, SETTINGS)
    degradation = data["degradation"][0]
    assert degradation["domains"]["stb"]["mae"]["relative_increase_pct"]["mean"] == 20
    for row in data["disproportion"]:
        assert row["target_minus_source_degradation_pp"]["mae"]["mean"] == -15


def test_zero_reference_percentage_is_undefined_not_silently_dropped():
    rows = [result("label_only", seed, values=(0, 100, 200)) for seed in [1, 2, 3]]
    rows += [result("mean_teacher", seed, values=(1, 100, 200)) for seed in [1, 2, 3]]
    contrast = analyze(rows, SETTINGS)["contrasts"][0]
    assert contrast["domains"]["stb"]["mae"]["relative_reduction_pct"] is None
    assert contrast["domains"]["stb"]["mae"]["absolute_reduction"]["mean"] == -1


def test_duplicate_invalid_or_target_selected_results_are_rejected():
    row = result("label_only", 1)
    with pytest.raises(ValueError, match="Duplicate"):
        analyze([row, copy.deepcopy(row)], SETTINGS)
    for field, value, message in [("selection_domain", "sta_val", "source validation"),
                                  ("seed", 9, "outside")]:
        bad = {**row, field: value}
        with pytest.raises(ValueError, match=message):
            analyze([bad], SETTINGS)
    bad = copy.deepcopy(row)
    bad["domains"]["stb"]["mae"] = float("nan")
    with pytest.raises(ValueError, match="Invalid count"):
        analyze([bad], SETTINGS)
    bad["domains"]["stb"]["mae"] = 100
    with pytest.raises(ValueError, match="RMSE cannot"):
        analyze([bad], SETTINGS)


def test_complete_requires_every_source_fraction_method_and_seed():
    rows = [result(method, seed, fraction=fraction, source=source)
            for source, fraction, method, seed in expected_keys(SETTINGS)]
    data = analyze(rows, SETTINGS, require_complete=True)
    assert data["complete"] and data["remaining_runs"] == 0
    assert data["completed_runs"] == 51
    assert len(data["contrasts"]) == 12

"""Post-evaluation statistics; never used by training or checkpoint selection."""
from __future__ import annotations

from collections import defaultdict
import math
import statistics


METRICS = ("mae", "rmse")
FULL_REFERENCE = {"label_only": "label_only", "mean_teacher": "label_only",
                  "mpcount": "mpcount", "ssl_dg": "mpcount", "domain_stable": "mpcount"}
CONTRASTS = (("mean_teacher", "label_only"), ("mpcount", "label_only"),
             ("ssl_dg", "mpcount"), ("domain_stable", "ssl_dg"))


def expected_keys(settings):
    sparse = {(source, fraction, method, seed)
              for source in settings["sources"] for fraction in settings["fractions"]
              for method in settings["methods"] for seed in settings["seeds"]}
    full = {(source, 1.0, method, settings["full_label_seed"])
            for source in settings["sources"] for method in settings["full_label_methods"]}
    return sparse | full


def result_key(row):
    return row["source"], row["fraction"], row["method"], row["seed"]


def summary(values):
    """SD across seed-level values, not pooled images or standard error."""
    values = list(values)
    if not values or any(value is None for value in values):
        return None
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Non-finite statistic")
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n_seeds": len(values)}


def relative_change(value, reference):
    # A zero-error reference has no defined percentage change.
    return 100.0 * (value - reference) / reference if reference > 0 else None


def analyze(results, settings, *, require_complete=False):
    expected = expected_keys(settings)
    indexed = {}
    for row in results:
        key = result_key(row)
        if key not in expected:
            raise ValueError(f"Result is outside the declared matrix: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate experiment: {key}")
        if row["selection_domain"] != row["source"] + "_val":
            raise ValueError("Result was not selected on source validation")
        if set(row["domains"]) != set(settings["sources"]):
            raise ValueError("A result is missing a fixed evaluation domain")
        for values in row["domains"].values():
            for metric in METRICS:
                if not math.isfinite(values[metric]) or values[metric] < 0:
                    raise ValueError("Invalid count error")
            if values["rmse"] + 1e-8 < values["mae"]:
                raise ValueError("RMSE cannot be lower than MAE for the same errors")
        indexed[key] = row
    missing = expected - indexed.keys()
    if require_complete and missing:
        raise ValueError(f"Conclusion gate: {len(missing)} experiments remain incomplete")

    expected_groups = defaultdict(set)
    for source, fraction, method, seed in expected:
        expected_groups[(source, fraction, method)].add(seed)
    groups, ready = [], {}
    for (source, fraction, method), seeds in sorted(expected_groups.items()):
        entries = {seed: indexed[(source, fraction, method, seed)]
                   for seed in sorted(seeds) if (source, fraction, method, seed) in indexed}
        group = {"source": source, "fraction": fraction, "method": method,
                 "completed_seeds": sorted(entries), "expected_seeds": sorted(seeds),
                 "complete": set(entries) == seeds, "domains": {}}
        if group["complete"]:
            ready[(source, fraction, method)] = entries
            group["domains"] = {
                domain: {metric: summary(row["domains"][domain][metric] for row in entries.values())
                         for metric in METRICS} for domain in settings["sources"]}
        groups.append(group)

    degradation, contrasts, disproportion = [], [], []
    for (source, fraction, method), entries in sorted(ready.items()):
        if fraction == 1:
            continue
        reference_method = FULL_REFERENCE[method]
        full = ready.get((source, 1.0, reference_method))
        if full is not None:
            reference = full[settings["full_label_seed"]]
            domains = {}
            for domain in settings["sources"]:
                domains[domain] = {}
                for metric in METRICS:
                    base = reference["domains"][domain][metric]
                    domains[domain][metric] = {
                        "full_reference": base,
                        "absolute_increase": summary(row["domains"][domain][metric] - base
                                                     for row in entries.values()),
                        "relative_increase_pct": summary(relative_change(row["domains"][domain][metric], base)
                                                         for row in entries.values())}
            degradation.append({"source": source, "fraction": fraction, "method": method,
                                "reference_method": reference_method,
                                "reference_seed": settings["full_label_seed"], "domains": domains})
            for target in settings["sources"]:
                if target == source:
                    continue
                differences = {}
                for metric in METRICS:
                    values = []
                    for row in entries.values():
                        in_domain = relative_change(row["domains"][source][metric],
                                                    reference["domains"][source][metric])
                        unseen = relative_change(row["domains"][target][metric],
                                                 reference["domains"][target][metric])
                        values.append(unseen - in_domain if in_domain is not None and unseen is not None else None)
                    differences[metric] = summary(values)
                disproportion.append({"source": source, "target": target, "fraction": fraction,
                                       "method": method, "reference_method": reference_method,
                                       "target_minus_source_degradation_pp": differences})

        for candidate, baseline in CONTRASTS:
            if method != candidate:
                continue
            other = ready.get((source, fraction, baseline))
            if other is None:
                continue
            if set(other) != set(entries):
                raise ValueError("A method contrast does not have matched sparse seeds")
            domains = {}
            for domain in settings["sources"]:
                domains[domain] = {}
                for metric in METRICS:
                    differences, percentages = [], []
                    for seed, row in entries.items():
                        base = other[seed]["domains"][domain][metric]
                        value = row["domains"][domain][metric]
                        differences.append(base - value)
                        change = relative_change(value, base)
                        percentages.append(-change if change is not None else None)
                    domains[domain][metric] = {"absolute_reduction": summary(differences),
                                               "relative_reduction_pct": summary(percentages)}
            contrasts.append({"source": source, "fraction": fraction, "candidate": candidate,
                              "baseline": baseline, "seeds": sorted(entries), "domains": domains,
                              "comparison": "matched split/model-seed identifiers; not a claim of identical stochastic trajectories"})
    return {"analysis_version": 1, "complete": not missing, "expected_runs": len(expected),
            "completed_runs": len(indexed), "remaining_runs": len(missing), "groups": groups,
            "degradation": degradation, "contrasts": contrasts, "disproportion": disproportion}

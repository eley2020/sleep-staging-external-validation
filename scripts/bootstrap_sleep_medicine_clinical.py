"""Subject-level bootstrap uncertainty for Sleep Medicine clinical endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ANALYSIS = Path("outputs/external_validation/bspc_n1_clinical_analysis.json")
OUT = Path("outputs/external_validation/sleep_medicine_clinical_bootstrap.json")
CLINICAL = ("tst_hours", "sleep_efficiency_pct", "waso_minutes", "rem_latency_minutes", "wake_minutes", "n1_minutes", "n2_minutes", "n3_minutes", "rem_minutes")


def point(rows: list[dict], name: str) -> dict[str, float | int | None]:
    truth = np.asarray([row["truth"][name] for row in rows], dtype=float)
    pred = np.asarray([row["prediction"][name] for row in rows], dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    delta = pred[valid] - truth[valid]
    return {
        "n_subjects": int(valid.sum()),
        "bias": float(delta.mean()) if len(delta) else None,
        "mae": float(np.abs(delta).mean()) if len(delta) else None,
        "median_absolute_error": float(np.median(np.abs(delta))) if len(delta) else None,
    }


def bootstrap(rows: list[dict], name: str, rng: np.random.Generator, draws: int) -> dict:
    truth = np.asarray([row["truth"][name] for row in rows], dtype=float)
    pred = np.asarray([row["prediction"][name] for row in rows], dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[valid], pred[valid]
    if not len(truth):
        return {"n_subjects": 0, "point": {"bias": None, "mae": None, "median_absolute_error": None}, "ci95": {}}
    indices = rng.integers(0, len(truth), size=(draws, len(truth)))
    delta = pred[indices] - truth[indices]
    values = {
        "bias": delta.mean(axis=1),
        "mae": np.abs(delta).mean(axis=1),
        "median_absolute_error": np.median(np.abs(delta), axis=1),
    }
    return {
        "n_subjects": int(len(truth)),
        "point": point([{"truth": {name: t}, "prediction": {name: p}} for t, p in zip(truth, pred)], name),
        "ci95": {key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))] for key, value in values.items()},
    }


def paired_delta(rows_a: list[dict], rows_b: list[dict], name: str, rng: np.random.Generator, draws: int) -> dict:
    if [row["subject"] for row in rows_a] != [row["subject"] for row in rows_b]:
        raise ValueError(f"Unmatched subjects for {name}")
    truth = np.asarray([row["truth"][name] for row in rows_a], dtype=float)
    pred_a = np.asarray([row["prediction"][name] for row in rows_a], dtype=float)
    pred_b = np.asarray([row["prediction"][name] for row in rows_b], dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred_a) & np.isfinite(pred_b)
    truth, pred_a, pred_b = truth[valid], pred_a[valid], pred_b[valid]
    if not len(truth):
        return {"n_subjects": 0, "point": {}, "ci95": {}}
    idx = rng.integers(0, len(truth), size=(draws, len(truth)))
    err_a, err_b = pred_a[idx] - truth[idx], pred_b[idx] - truth[idx]
    values = {
        "bias_adapted_minus_frozen": err_b.mean(axis=1) - err_a.mean(axis=1),
        "mae_adapted_minus_frozen": np.abs(err_b).mean(axis=1) - np.abs(err_a).mean(axis=1),
    }
    point_a = point([{"truth": {name: t}, "prediction": {name: p}} for t, p in zip(truth, pred_a)], name)
    point_b = point([{"truth": {name: t}, "prediction": {name: p}} for t, p in zip(truth, pred_b)], name)
    return {
        "n_subjects": int(len(truth)),
        "point": {"bias_adapted_minus_frozen": point_b["bias"] - point_a["bias"], "mae_adapted_minus_frozen": point_b["mae"] - point_a["mae"]},
        "ci95": {key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))] for key, value in values.items()},
    }


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260828)
    draws = 5000
    payload = {"protocol": "Subject-level paired bootstrap (5000 draws) for retrospective sleep-structure endpoints; no labels are used for model or parameter selection.", "draws": draws, "cohorts": {}}
    for cohort in ("shhs1", "shhs2"):
        payload["cohorts"][cohort] = {}
        for seed, run in analysis[cohort].items():
            payload["cohorts"][cohort][seed] = {}
            for mode in ("frozen", "adapted"):
                rows = run[mode]["subject_rows"]
                payload["cohorts"][cohort][seed][mode] = {name: bootstrap(rows, name, rng, draws) for name in CLINICAL}
            payload["cohorts"][cohort][seed]["paired_adapted_minus_frozen"] = {name: paired_delta(run["frozen"]["subject_rows"], run["adapted"]["subject_rows"], name, rng, draws) for name in CLINICAL}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

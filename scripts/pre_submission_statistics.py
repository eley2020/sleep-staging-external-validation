"""Generate locked, subject-level statistics for the submission package.

This script only consumes saved predictions and reports.  It never fits a
model, selects a checkpoint, or reads target labels for any purpose other
than final scoring.  Bootstrap resampling is performed at the subject/record
level so epochs are not treated as independent replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score


STAGES = ["W", "N1", "N2", "N3", "REM"]
METRICS = ("accuracy", "macro_f1", "weighted_f1", "kappa")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(5),
        target_names=STAGES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(5)).tolist(),
    }


def lightweight_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    matrix = np.bincount(y_true * 5 + y_pred, minlength=25).reshape(5, 5).astype(float)
    return metrics_from_matrix(matrix)


def metrics_from_matrix(matrix: np.ndarray) -> dict[str, float]:
    """Compute aggregate scores from a 5x5 confusion matrix."""
    matrix = np.asarray(matrix, dtype=float)
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    f1 = []
    support = matrix.sum(axis=1)
    for i in range(5):
        tp = matrix[i, i]
        precision = tp / matrix[:, i].sum() if matrix[:, i].sum() else 0.0
        recall = tp / support[i] if support[i] else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    expected = float((matrix.sum(axis=0) * support).sum() / total**2) if total else 0.0
    kappa = (accuracy - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if total else 0.0,
        "kappa": float(kappa),
    }


def load_npz_records(directory: Path, pattern: str, record_key: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob(pattern)):
        with np.load(path, allow_pickle=True) as data:
            if "y_true" not in data or "y_pred" not in data:
                continue
            y_true = np.asarray(data["y_true"], dtype=np.int64)
            y_pred = np.asarray(data["y_pred"], dtype=np.int64)
        if y_true.shape != y_pred.shape or y_true.ndim != 1:
            raise ValueError(f"Invalid prediction archive: {path}")
        record = path.stem
        if record.endswith("_predictions"):
            record = record[: -len("_predictions")]
        subject = record
        rows.append({"record": record, "subject": subject, "y_true": y_true, "y_pred": y_pred})
    if not rows:
        raise FileNotFoundError(f"No prediction archives found in {directory} ({pattern})")
    return rows


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attach_metadata(rows: list[dict[str, Any]], path: Path, record_field: str) -> None:
    metadata = {str(r[record_field]): r for r in load_csv_rows(path)}
    for row in rows:
        key = row["record"]
        match = metadata.get(key)
        if match is None:
            # ISRUC archives use numeric stems while CSV records may carry .rec.
            match = metadata.get(key.replace(".rec", "")) or metadata.get(key + ".rec")
        if match:
            row["subject"] = str(match.get("subject", match.get("record", key)))
            row["metadata"] = match


def score_rows(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    y_true = np.concatenate([r["y_true"] for r in rows])
    y_pred = np.concatenate([r["y_pred"] for r in rows])
    point = metric_dict(y_true, y_pred)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["subject"], []).append(row)
    subjects = sorted(grouped)
    # Resampling subject-level confusion matrices preserves the requested
    # bootstrap unit without repeatedly concatenating all epoch arrays.
    subject_matrices = {
        subject: sum(
            (
                np.bincount(r["y_true"] * 5 + r["y_pred"], minlength=25)
                .reshape(5, 5)
                .astype(np.int64)
                for r in grouped[subject]
            ),
            start=np.zeros((5, 5), dtype=np.int64),
        )
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(bootstrap_samples, dtype=float) for name in METRICS}
    for i in range(bootstrap_samples):
        sampled_subjects = rng.choice(subjects, len(subjects), replace=True)
        matrix = sum((subject_matrices[s] for s in sampled_subjects), start=np.zeros((5, 5), dtype=np.int64))
        metrics = metrics_from_matrix(matrix)
        for name in METRICS:
            draws[name][i] = metrics[name]
    ci = {name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for name, values in draws.items()}
    per_subject = []
    for subject in subjects:
        yt = np.concatenate([r["y_true"] for r in grouped[subject]])
        yp = np.concatenate([r["y_pred"] for r in grouped[subject]])
        m = metric_dict(yt, yp)
        per_subject.append({"subject": subject, "n_epochs": int(len(yt)), **{k: m[k] for k in METRICS}})
    return {
        "n_subjects": len(subjects),
        "n_records": len(rows),
        "n_epochs": int(len(y_true)),
        "point_estimate": {k: point[k] for k in METRICS},
        "subject_bootstrap_95ci": ci,
        "classification_report": point["classification_report"],
        "confusion_matrix": point["confusion_matrix"],
        "per_subject": per_subject,
    }


def stage_sensitivity_specificity(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    matrix = np.asarray(report["confusion_matrix"], dtype=float)
    total = matrix.sum()
    out = {}
    for i, stage in enumerate(STAGES):
        tp = matrix[i, i]
        fn = matrix[i].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = total - tp - fn - fp
        out[stage] = {
            "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        }
    return out


def paired_adaptation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = []
    p_values = []
    for split in payload.get("splits", []):
        frozen = split.get("frozen_source_metrics", {})
        adapted = split.get("adapted_metrics", {})
        delta = {k: float(adapted[k] - frozen[k]) for k in METRICS if k in frozen and k in adapted}
        raw.append({"seed": split.get("seed"), "delta": delta, "frozen": frozen, "adapted": adapted})
    for name in METRICS:
        values = np.asarray([r["delta"][name] for r in raw if name in r["delta"]], dtype=float)
        p_values.append(float(wilcoxon(values, alternative="two-sided", method="auto").pvalue) if len(values) > 0 else 1.0)
    order = np.argsort(p_values)
    holm = np.ones(len(p_values), dtype=float)
    for rank, idx in enumerate(order):
        holm[idx] = min(1.0, (len(p_values) - rank) * p_values[idx])
    return {
        "protocol": payload.get("protocol"),
        "splits": raw,
        "across_split_summary": payload.get("across_split_summary", {}),
        "wilcoxon_p_raw": dict(zip(METRICS, p_values)),
        "wilcoxon_p_holm": dict(zip(METRICS, holm.tolist())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/external_validation/pre_submission"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cohorts: dict[str, dict[str, Any]] = {}
    shhs1 = Path("outputs/external_validation/shhs/frozen_ensemble_identity_shhs1")
    shhs2 = Path("outputs/external_validation/shhs/frozen_ensemble_identity_shhs2")
    isruc = Path("outputs/external_validation/isruc_cohort1_zero_shot_x1_final")
    st = Path("outputs/external_validation/sleepedf_st/quality_gated_attnsleep_20fold_frozen_ensemble_no_tta")
    for name, directory, pattern, csv_path, field in [
        ("SHHS1", shhs1 / "record_predictions", "*.npz", shhs1 / "record_metrics.csv", "record"),
        ("SHHS2", shhs2 / "record_predictions", "*.npz", shhs2 / "record_metrics.csv", "record"),
        ("ISRUC_Cohort_I", isruc, "*.npz", isruc / "record_metrics.csv", "record"),
        ("SleepEDF_ST", st, "*_predictions.npz", st / "record_metrics.csv", "record"),
    ]:
        rows = load_npz_records(directory, pattern, field)
        if csv_path.exists():
            attach_metadata(rows, csv_path, field)
        report = score_rows(rows, args.bootstrap_samples, args.seed)
        report["stage_sensitivity_specificity"] = stage_sensitivity_specificity(report)
        report["source"] = str(directory)
        report["target_labels_used"] = "final scoring only"
        cohorts[name] = report
        (out / f"{name.lower()}_per_subject.csv").write_text(
            "subject,n_epochs,accuracy,macro_f1,weighted_f1,kappa\n" +
            "\n".join(",".join(str(r[k]) for k in ("subject", "n_epochs", *METRICS)) for r in report["per_subject"]) + "\n",
            encoding="utf-8",
        )

    adaptation = {}
    for name, path in [("SHHS1_adaptation", Path("outputs/external_validation/shhs/adaptation/full_multisplit_summary.json")), ("SleepEDF_ST_adaptation", Path("outputs/external_validation/sleepedf_st/adaptation/full_multisplit_summary.json"))]:
        if path.exists():
            adaptation[name] = paired_adaptation(path)

    payload = {
        "protocol": "Pre-submission locked statistics; subject/record-level bootstrap; no target-label model selection",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "cohorts": cohorts,
        "adaptation": adaptation,
        "independent_cohort_definition": "ISRUC Cohort I is the primary independent cohort; Sleep-EDF ST is an independent Sleep-EDF telemetry subset; SHHS2 is longitudinal follow-up, not a new population.",
        "external_data_gap": "MASS/MESA raw PSG was not present locally; no MASS/MESA result is claimed.",
    }
    (out / "pre_submission_statistics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary_rows = []
    for name, report in cohorts.items():
        row = {"cohort": name, "n_subjects": report["n_subjects"], "n_records": report["n_records"], "n_epochs": report["n_epochs"]}
        for metric in METRICS:
            row[metric] = report["point_estimate"][metric]
            row[f"{metric}_ci_low"] = report["subject_bootstrap_95ci"][metric][0]
            row[f"{metric}_ci_high"] = report["subject_bootstrap_95ci"][metric][1]
        summary_rows.append(row)
    with (out / "cohort_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)
    print(out / "pre_submission_statistics.json")


if __name__ == "__main__":
    main()

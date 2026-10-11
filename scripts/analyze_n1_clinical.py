"""Unified N1 and clinical hypnogram analysis for BSPC submission."""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

STAGES = ("W", "N1", "N2", "N3", "REM")
CLINICAL = ("tst_hours", "sleep_efficiency_pct", "waso_minutes", "rem_latency_minutes", "wake_minutes", "n1_minutes", "n2_minutes", "n3_minutes", "rem_minutes")


def json_clean(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    return value


def clinical(y: np.ndarray, epoch_minutes: float = 0.5, n_signal_epochs: int | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    if n_signal_epochs is None:
        n_signal_epochs = len(y)
    sleep = y != 0
    idx = np.flatnonzero(sleep)
    first, last = (int(idx[0]), int(idx[-1])) if len(idx) else (0, -1)
    rem = np.flatnonzero(y == 4)
    return {
        "tst_hours": float(sleep.sum() * epoch_minutes / 60.0),
        "sleep_efficiency_pct": float(sleep.sum() / max(n_signal_epochs, 1) * 100.0),
        "waso_minutes": float((y[first:last + 1] == 0).sum() * epoch_minutes) if len(idx) else 0.0,
        "rem_latency_minutes": float((int(rem[0]) - first) * epoch_minutes) if len(idx) and len(rem) else float("nan"),
        "wake_minutes": float((y == 0).sum() * epoch_minutes),
        "n1_minutes": float((y == 1).sum() * epoch_minutes),
        "n2_minutes": float((y == 2).sum() * epoch_minutes),
        "n3_minutes": float((y == 3).sum() * epoch_minutes),
        "rem_minutes": float((y == 4).sum() * epoch_minutes),
    }


def classification(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion_matrix(y, pred, labels=range(5)).astype(np.int64)
    report = classification_report(y, pred, labels=range(5), target_names=STAGES, output_dict=True, zero_division=0)
    per_stage = {}
    total = cm.sum()
    for i, stage in enumerate(STAGES):
        tp = float(cm[i, i]); fn = float(cm[i, :].sum() - cm[i, i]); fp = float(cm[:, i].sum() - cm[i, i]); tn = float(total - tp - fn - fp)
        per_stage[stage] = {"precision": float(report[stage]["precision"]), "recall_sensitivity": float(report[stage]["recall"]), "f1": float(report[stage]["f1-score"]), "support": int(report[stage]["support"]), "specificity": tn / max(tn + fp, 1.0)}
    return {"accuracy": float(report["accuracy"]), "macro_f1": float(report["macro avg"]["f1-score"]), "weighted_f1": float(report["weighted avg"]["f1-score"]), "kappa": float((np.trace(cm) / max(total, 1) - np.dot(cm.sum(1), cm.sum(0)) / max(total * total, 1)) / max(1.0 - np.dot(cm.sum(1), cm.sum(0)) / max(total * total, 1), 1e-12)), "per_stage": per_stage, "confusion_matrix": cm.tolist()}


def clinical_summary(rows: list[dict]) -> dict:
    out = {"n_subjects": len(rows), "metrics": {}}
    for name in CLINICAL:
        truth = np.asarray([r["truth"][name] for r in rows], dtype=np.float64)
        pred = np.asarray([r["prediction"][name] for r in rows], dtype=np.float64)
        valid = np.isfinite(truth) & np.isfinite(pred)
        delta = pred[valid] - truth[valid]
        sd = float(np.std(delta, ddof=1)) if valid.sum() > 1 else 0.0
        out["metrics"][name] = {"n_valid": int(valid.sum()), "truth_mean": float(np.mean(truth[valid])) if valid.any() else None, "prediction_mean": float(np.mean(pred[valid])) if valid.any() else None, "bias": float(np.mean(delta)) if valid.any() else None, "mae": float(np.mean(np.abs(delta))) if valid.any() else None, "median_absolute_error": float(np.median(np.abs(delta))) if valid.any() else None, "limits_of_agreement_95": [float(np.mean(delta) - 1.96 * sd), float(np.mean(delta) + 1.96 * sd)] if valid.any() else None}
    return out


def summarize_sequences(sequences: list[tuple[str, np.ndarray, np.ndarray, int | None]]) -> dict:
    y = np.concatenate([item[1] for item in sequences]); pred = np.concatenate([item[2] for item in sequences])
    rows = [{"subject": subject, "truth": clinical(yt, n_signal_epochs=n_signal), "prediction": clinical(yp, n_signal_epochs=n_signal)} for subject, yt, yp, n_signal in sequences]
    return {"classification": classification(y, pred), "clinical": clinical_summary(rows), "subject_rows": rows}


def load_shhs_prediction(path: Path, signal_csv: Path) -> list[tuple[str, np.ndarray, np.ndarray, int | None]]:
    with np.load(path, allow_pickle=True) as data:
        y_true, y_pred, spans = data["y_true"], data["y_pred"], data["spans"].tolist()
    signal_epochs = {}
    with signal_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "n_signal_epochs" in row: signal_epochs[row["record"]] = int(row["n_signal_epochs"])
    return [(str(subject), y_true[int(start):int(end)], y_pred[int(start):int(end)], signal_epochs.get(str(record))) for record, subject, start, end in spans]


def load_isruc_dir(path: Path, pred_key: str = "y_pred") -> list[tuple[str, np.ndarray, np.ndarray, int | None]]:
    rows = []
    for file in sorted(path.glob("*.npz"), key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else p.stem):
        with np.load(file, allow_pickle=True) as data:
            if "y_true" not in data or pred_key not in data: continue
            rows.append((file.stem, np.asarray(data["y_true"], dtype=np.int64), np.asarray(data[pred_key], dtype=np.int64), None))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shhs-root", type=Path, default=Path("outputs/external_validation/shhs"))
    parser.add_argument("--isruc-zero-shot", type=Path, default=Path("outputs/external_validation/isruc_cohort1_zero_shot_logits_cache"))
    args = parser.parse_args()
    payload = {"protocol": "Unified N1 sensitivity and clinical hypnogram analysis; metrics computed from saved subject-disjoint predictions", "shhs1": {}, "shhs2": {}, "isruc": {}}
    for cohort, seeds in (("shhs1", (20260813, 20260814, 20260815)), ("shhs2", (20260813, 20260814))):
        for seed in seeds:
            if cohort == "shhs1":
                run = args.shhs_root / "adaptation" / f"seed{seed}_full"; base = args.shhs_root / "frozen_ensemble_identity_shhs1" / "record_metrics.csv"
                adapted = load_shhs_prediction(run / "adapted_test_predictions.npz", base); frozen = load_shhs_prediction(run / "frozen_test_predictions.npz", base)
            else:
                run = args.shhs_root / "adaptation" / f"shhs2_seed{seed}_locked_followup"; base = args.shhs_root / "frozen_ensemble_identity_shhs2" / "record_metrics.csv"
                adapted = load_shhs_prediction(run / "adapted_predictions.npz", base); frozen = load_shhs_prediction(run / "source_predictions.npz", base)
            payload[cohort][str(seed)] = {"frozen": summarize_sequences(frozen), "adapted": summarize_sequences(adapted)}
    payload["isruc"]["zero_shot_100"] = summarize_sequences(load_isruc_dir(args.isruc_zero_shot, "y_pred"))
    # Explicit matched ISRUC full fine-tuning summaries, one entry per seed.
    for seed in (20260819, 20260820, 20260821):
        files = sorted(Path("outputs/external_validation").glob(f"isruc_original_finetune_matched_seed{seed}_subject_*.npz"))
        seq = []
        for file in files:
            with np.load(file, allow_pickle=True) as data:
                seq.append((file.stem, np.asarray(data["y_true"], dtype=np.int64), np.asarray(data["finetuned_y_pred"], dtype=np.int64), None))
        if seq: payload["isruc"].setdefault("matched_full_finetune", {})[str(seed)] = summarize_sequences(seq)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_clean(payload), indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

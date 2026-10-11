"""Fair train-versus-subject-validation audit for saved AttnSleep checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/attnsleep_multichannel_baseline_train/full20_quality_gate_4ch_emg_30epoch"),
    )
    parser.add_argument(
        "--np-data-dir",
        type=Path,
        default=Path("data/sleepedf20_multichannel_npz/eeg2_eog_emg_physical_volts"),
    )
    parser.add_argument("--attnsleep-dir", type=Path, default=Path("external_baselines/AttnSleep"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attnsleep_generalization_gap_audit"))
    parser.add_argument("--fold-ids", default="all")
    parser.add_argument("--num-folds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=73021)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def parse_folds(value: str, num_folds: int) -> list[int]:
    if value == "all":
        return list(range(num_folds))
    folds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not folds or any(fold < 0 or fold >= num_folds for fold in folds):
        raise ValueError(f"Invalid fold ids: {value}")
    return folds


def subject_id(path: str | Path) -> str:
    return Path(path).name[3:5]


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    total = float(cm.sum())
    observed = float(tp.sum() / total) if total else float("nan")
    expected = float((support * predicted).sum() / (total * total)) if total else float("nan")
    kappa = (observed - expected) / (1 - expected) if total and expected < 1 else 0.0
    return {
        "accuracy": observed,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else float("nan"),
        "kappa": float(kappa),
        "n_epochs": int(total),
        "class_precision": precision.tolist(),
        "class_recall": recall.tolist(),
        "class_f1": f1.tolist(),
        "class_support": support.astype(int).tolist(),
    }


def collect_confusions(
    model,
    loader,
    sample_subjects: np.ndarray,
    class_weights,
    device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch
    import torch.nn.functional as F

    model.eval()
    all_y: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    offset = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, quality_prior = batch
                entropy = None
            elif len(batch) == 4:
                x, y, quality_prior, entropy = batch
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            x = x.float().to(device)
            y_dev = y.to(device)
            quality_prior = quality_prior.float().to(device)
            if entropy is not None:
                entropy = entropy.float().to(device)
            fusion = getattr(model, "channel_fusion", "early")
            if fusion in {"quality-prior-gated", "quality-hybrid-gated", "quality-piecewise-gated", "feature-attention-gated"}:
                logits = model(x, channel_prior=quality_prior, **({"entropy_features": entropy} if entropy is not None else {}))
            else:
                logits = model(x, **({"entropy_features": entropy} if entropy is not None else {}))
            losses = F.cross_entropy(logits, y_dev, weight=class_weights, reduction="none")
            batch_weights = class_weights[y_dev]
            weighted_loss_sum += float(losses.sum().cpu())
            weight_sum += float(batch_weights.sum().cpu())
            y_np = y.detach().cpu().numpy().astype(np.int64)
            pred_np = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            all_y.append(y_np)
            all_pred.append(pred_np)
            offset += len(y_np)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    subject_vector = sample_subjects[:offset]
    if len(subject_vector) != len(y_true):
        raise RuntimeError("Subject ids do not align with loader samples")
    cm_by_subject: dict[str, np.ndarray] = {}
    for sid in sorted(set(subject_vector.tolist())):
        mask = subject_vector == sid
        cm = np.zeros((5, 5), dtype=np.int64)
        np.add.at(cm, (y_true[mask], y_pred[mask]), 1)
        cm_by_subject[str(sid)] = cm
    pooled = np.sum(list(cm_by_subject.values()), axis=0)
    metrics = metrics_from_confusion(pooled)
    metrics["weighted_cross_entropy"] = weighted_loss_sum / max(weight_sum, 1e-12)
    metrics["n_subjects"] = len(cm_by_subject)
    metrics["per_subject"] = {
        sid: metrics_from_confusion(cm) for sid, cm in cm_by_subject.items()
    }
    return metrics, cm_by_subject


def bootstrap_gap(
    train_cms: dict[str, np.ndarray],
    val_cms: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    train_ids = sorted(train_cms)
    val_ids = sorted(val_cms)
    draws: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    for _ in range(replicates):
        train_sample = rng.choice(train_ids, size=len(train_ids), replace=True)
        val_sample = rng.choice(val_ids, size=len(val_ids), replace=True)
        train_cm = np.sum([train_cms[sid] for sid in train_sample], axis=0)
        val_cm = np.sum([val_cms[sid] for sid in val_sample], axis=0)
        train_metric = metrics_from_confusion(train_cm)
        val_metric = metrics_from_confusion(val_cm)
        for metric in draws:
            draws[metric].append(train_metric[metric] - val_metric[metric])
    return {
        metric: {
            "lower_95": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "upper_95": float(np.quantile(values, 0.975)),
            "probability_positive": float(np.mean(np.asarray(values) > 0)),
        }
        for metric, values in draws.items()
    }


def make_subject_vector(files: list[str]) -> np.ndarray:
    parts = []
    for path in sorted(files):
        with np.load(path, allow_pickle=True) as data:
            labels = np.asarray(data["y"], dtype=np.int64)
        parts.append(np.repeat(subject_id(path), len(labels)))
    return np.concatenate(parts)


def main() -> None:
    args = parse_args()
    import torch
    from sklearn.metrics import confusion_matrix

    sys.path.insert(0, str(args.attnsleep_dir.resolve()))
    from model.model import AttnSleep
    from utils.util import load_folds_data

    from scripts.train_attnsleep_multichannel_baseline import (
        make_loaders,
        parse_channel_indices,
        resolve_device,
    )

    device = resolve_device(args.device)
    if device.type == "cuda":
        print(f"GPU audit device: {torch.cuda.get_device_name(device.index or 0)}", flush=True)
    folds = load_folds_data(str(args.np_data_dir.resolve()), args.num_folds)
    channel_indices = [0, 1, 2, 3]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []

    for fold_id in parse_folds(args.fold_ids, args.num_folds):
        fold_dir = args.run_dir / f"fold_{fold_id:02d}"
        checkpoint_path = fold_dir / "checkpoint_best.pt"
        metadata_path = fold_dir / "metadata.json"
        if not checkpoint_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing checkpoint or metadata in {fold_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        outer_train_files, test_files = folds[fold_id]
        val_subjects = set(metadata["checkpoint_selection_subjects"])
        test_subjects = set(metadata["test_subjects"])
        model_train_files = sorted(
            path for path in outer_train_files if subject_id(path) not in val_subjects
        )
        val_files = sorted(
            path for path in outer_train_files if subject_id(path) in val_subjects
        )
        if {subject_id(path) for path in model_train_files} & val_subjects:
            raise RuntimeError(f"Train/validation subject overlap in fold {fold_id}")
        if {subject_id(path) for path in model_train_files + val_files} & test_subjects:
            raise RuntimeError(f"Test subject included in gap audit fold {fold_id}")
        if {subject_id(path) for path in test_files} != test_subjects:
            raise RuntimeError(f"Outer test split mismatch in fold {fold_id}")

        model = AttnSleep(
            input_channels=int(metadata["input_channels"]),
            channel_fusion=metadata["channel_fusion"],
            channel_dropout=float(metadata.get("channel_dropout", 0.0)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        train_loader, val_loader, _counts, _channels, _n_channels = make_loaders(
            model_train_files,
            val_files,
            args.batch_size,
            channel_indices,
            quality_stats=None,
            sampler="none",
            augment_noise_std=0.0,
            augment_scale_std=0.0,
            augment_time_shift=0,
            augment_time_mask_samples=0,
            augment_time_mask_prob=0.0,
        )
        # Rebuild the train loader over the same raw dataset without shuffling.
        train_eval_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        weights = torch.tensor(metadata["class_weights"], dtype=torch.float32, device=device)
        train_subject_vector = make_subject_vector(model_train_files)
        val_subject_vector = make_subject_vector(val_files)
        train_metrics, train_cms = collect_confusions(
            model, train_eval_loader, train_subject_vector, weights, device
        )
        val_metrics, val_cms = collect_confusions(
            model, val_loader, val_subject_vector, weights, device
        )
        gap = {
            "accuracy": train_metrics["accuracy"] - val_metrics["accuracy"],
            "macro_f1": train_metrics["macro_f1"] - val_metrics["macro_f1"],
            "weighted_f1": train_metrics["weighted_f1"] - val_metrics["weighted_f1"],
            "weighted_cross_entropy": val_metrics["weighted_cross_entropy"] - train_metrics["weighted_cross_entropy"],
            "N1_f1": train_metrics["class_f1"][1] - val_metrics["class_f1"][1],
            "N3_f1": train_metrics["class_f1"][3] - val_metrics["class_f1"][3],
        }
        bootstrap = bootstrap_gap(
            train_cms,
            val_cms,
            args.bootstrap_replicates,
            args.seed + fold_id,
        )
        result = {
            "fold": fold_id,
            "checkpoint": str(checkpoint_path),
            "selected_epoch": int(checkpoint.get("epoch", -1)),
            "train_subjects": sorted({subject_id(path) for path in model_train_files}),
            "validation_subjects": sorted(val_subjects),
            "test_subjects_excluded": sorted(test_subjects),
            "evaluation_mode": "model.eval; no_grad; raw non-overlapping epochs; stochastic waveform augmentation disabled",
            "train": train_metrics,
            "validation": val_metrics,
            "train_confusion_by_subject": {
                sid: cm.tolist() for sid, cm in train_cms.items()
            },
            "validation_confusion_by_subject": {
                sid: cm.tolist() for sid, cm in val_cms.items()
            },
            "gap_train_minus_validation": gap,
            "subject_bootstrap_gap_95ci": bootstrap,
        }
        out_path = args.output_dir / f"fold_{fold_id:02d}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        fold_rows.append({"fold": fold_id, **gap, "macro_f1_gap_ci_low": bootstrap["macro_f1"]["lower_95"], "macro_f1_gap_ci_high": bootstrap["macro_f1"]["upper_95"], "macro_f1_gap_bootstrap_p_positive": bootstrap["macro_f1"]["probability_positive"], "train_macro_f1": train_metrics["macro_f1"], "validation_macro_f1": val_metrics["macro_f1"], "train_accuracy": train_metrics["accuracy"], "validation_accuracy": val_metrics["accuracy"], "train_n_subjects": train_metrics["n_subjects"], "validation_n_subjects": val_metrics["n_subjects"]})
        print(
            f"fold {fold_id:02d}: train/val Macro-F1 "
            f"{train_metrics['macro_f1']:.4f}/{val_metrics['macro_f1']:.4f}; "
            f"gap={gap['macro_f1']:+.4f}, subject-bootstrap 95% CI "
            f"[{bootstrap['macro_f1']['lower_95']:+.4f}, {bootstrap['macro_f1']['upper_95']:+.4f}]",
            flush=True,
        )
        del model, checkpoint, train_loader, val_loader, train_eval_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.output_dir / f"fold_metrics_{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    macro_gaps = np.asarray([row["macro_f1"] for row in fold_rows])
    aggregate = {
        "run_dir": str(args.run_dir),
        "folds": [row["fold"] for row in fold_rows],
        "n_folds": len(fold_rows),
        "macro_f1_gap_mean": float(macro_gaps.mean()),
        "macro_f1_gap_sd": float(macro_gaps.std(ddof=1)) if len(macro_gaps) > 1 else 0.0,
        "macro_f1_gap_median": float(np.median(macro_gaps)),
        "macro_f1_gap_min": float(macro_gaps.min()),
        "macro_f1_gap_max": float(macro_gaps.max()),
        "macro_f1_gap_positive_folds": int(np.sum(macro_gaps > 0)),
        "fold_rows": fold_rows,
        "note": "Outer folds overlap in their training subjects; fold-level summaries are descriptive, not independent-sample inference.",
    }
    aggregate_path = args.output_dir / f"aggregate_{os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "fold_rows"}, indent=2))


if __name__ == "__main__":
    main()

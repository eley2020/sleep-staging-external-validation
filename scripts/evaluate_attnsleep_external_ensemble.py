"""Run a frozen AttnSleep checkpoint ensemble on an external PSG cohort.

The evaluator never uses external labels to select checkpoints, calibrate
thresholds, or fit preprocessing statistics.  It averages logits from all
source-CV checkpoints, so the result is an out-of-domain, source-only model
ensemble rather than a target-domain adaptation experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "outputs/attnsleep_multichannel_baseline_train/"
            "full20_quality_gate_4ch_emg_30epoch"
        ),
    )
    parser.add_argument(
        "--np-data-dir",
        type=Path,
        default=Path("data/sleepedf_st_multichannel_npz/eeg2_eog_emg_physical_volts"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/external_validation/sleepedf_st/"
            "quality_gated_attnsleep_20fold_frozen_ensemble"
        ),
    )
    parser.add_argument(
        "--attnsleep-dir", type=Path, default=Path("external_baselines/AttnSleep")
    )
    parser.add_argument("--num-folds", type=int, default=20)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional smoke-test cap; 0 evaluates every external record.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--tta-transforms",
        default="identity;shift:-10:0,1,2;shift:10:0,1,2;scale:0.95:3;scale:1.05:3",
        help="Fixed, source-defined deterministic transforms. Use 'none' for identity only.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def parse_tta_transforms(value: str) -> list[dict[str, Any]]:
    if value in {"", "none"}:
        return [{"kind": "identity"}]
    transforms: list[dict[str, Any]] = []
    for raw_item in value.split(";"):
        parts = raw_item.strip().split(":")
        if parts == ["identity"]:
            transforms.append({"kind": "identity"})
        elif len(parts) == 3 and parts[0] == "shift":
            transforms.append(
                {
                    "kind": "shift",
                    "shift": int(parts[1]),
                    "positions": [int(v) for v in parts[2].split(",")],
                }
            )
        elif len(parts) == 3 and parts[0] == "scale":
            transforms.append(
                {
                    "kind": "scale",
                    "scale": float(parts[1]),
                    "positions": [int(v) for v in parts[2].split(",")],
                }
            )
        else:
            raise ValueError(f"Unsupported TTA transform: {raw_item!r}")
    if not any(item["kind"] == "identity" for item in transforms):
        transforms.insert(0, {"kind": "identity"})
    return transforms


def apply_tta(x: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    if transform["kind"] == "identity":
        return x
    output = x.copy()
    positions = transform["positions"]
    if min(positions) < 0 or max(positions) >= x.shape[1]:
        raise ValueError(f"TTA positions {positions} are invalid for {x.shape}")
    if transform["kind"] == "shift":
        output[:, positions, :] = np.roll(output[:, positions, :], transform["shift"], axis=2)
    elif transform["kind"] == "scale":
        output[:, positions, :] *= transform["scale"]
    else:
        raise ValueError(f"Unknown transform kind: {transform['kind']}")
    return output


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(device_arg)


def load_record(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=True) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        normalization = str(data["normalization"].item())
    if x.ndim != 3:
        raise ValueError(f"Expected (epochs, channels, samples) in {path}, got {x.shape}")
    return x, y, normalization


def predict_ensemble_logits(
    base_model,
    params,
    buffers,
    x: np.ndarray,
    batch_size: int,
    transforms: list[dict[str, Any]],
    device,
) -> np.ndarray:
    """Average frozen checkpoint logits with a vectorized model dimension."""
    import torch
    from torch.func import functional_call, vmap

    def call_model(model_params, model_buffers, batch):
        return functional_call(base_model, (model_params, model_buffers), (batch,))

    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            transform_logits = []
            for transform in transforms:
                xb = torch.from_numpy(apply_tta(batch, transform)).to(device)
                # Shape is (n_checkpoints, batch, n_classes). Averaging here
                # is algebraically identical to the original serial ensemble.
                transform_logits.append(
                    vmap(call_model, in_dims=(0, 0, None))(params, buffers, xb)
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )
            parts.append(np.mean(transform_logits, axis=0))
    return np.vstack(parts)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score

    report = classification_report(
        y_true,
        y_pred,
        labels=range(5),
        target_names=STAGE_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "classification_report": report,
    }


def bootstrap_subject_ci(records: list[dict[str, Any]], samples: int, seed: int) -> dict[str, list[float]]:
    if samples <= 0:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["subject"], []).append(record)
    subjects = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in ("accuracy", "macro_f1", "weighted_f1", "kappa")}
    for _ in range(samples):
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        y_true = np.concatenate([item["y_true"] for subject in chosen for item in grouped[subject]])
        y_pred = np.concatenate([item["y_pred"] for subject in chosen for item in grouped[subject]])
        values = metric_dict(y_true, y_pred)
        for key in draws:
            draws[key].append(values[key])
    return {key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for key, values in draws.items()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.attnsleep_dir.resolve()))

    import torch
    from model.model import AttnSleep

    device = resolve_device(args.device)
    transforms = parse_tta_transforms(args.tta_transforms)
    record_paths = sorted(args.np_data_dir.glob("*.npz"))
    if not record_paths:
        raise FileNotFoundError(f"No NPZ records in {args.np_data_dir}")
    if args.max_records > 0:
        record_paths = record_paths[: args.max_records]

    metadata_paths = [args.run_dir / f"fold_{fold_id:02d}" / "metadata.json" for fold_id in range(args.num_folds)]
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    if any(item["input_channels"] != 4 or item["channel_fusion"] != "quality-gated" for item in metadata):
        raise ValueError("This evaluator is restricted to the frozen four-channel quality-gated run")

    source_models = []
    for fold_id in range(args.num_folds):
        checkpoint_path = args.run_dir / f"fold_{fold_id:02d}" / "checkpoint_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = AttnSleep(input_channels=4, channel_fusion="quality-gated", channel_dropout=0.15).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        source_models.append(model)
    from torch.func import stack_module_state

    ensemble_params, ensemble_buffers = stack_module_state(source_models)
    base_model = source_models[0]
    print(f"Loaded {len(source_models)} frozen source checkpoints on {device}")

    records: list[dict[str, Any]] = []
    for record_path in record_paths:
        x, y_true, normalization = load_record(record_path)
        if normalization != "physical-volts":
            raise ValueError(f"{record_path} has normalization={normalization!r}, expected physical-volts")
        if x.shape[1:] != (4, 3000):
            raise ValueError(f"{record_path} has incompatible input shape {x.shape}")
        mean_logits = predict_ensemble_logits(
            base_model,
            ensemble_params,
            ensemble_buffers,
            x,
            args.batch_size,
            transforms,
            device,
        )
        y_pred = mean_logits.argmax(axis=1).astype(np.int64)
        record_metrics = metric_dict(y_true, y_pred)
        name = record_path.stem
        np.savez_compressed(
            args.output_dir / f"{name}_predictions.npz",
            y_true=y_true,
            y_pred=y_pred,
            mean_logits=mean_logits.astype(np.float32),
        )
        records.append({"name": name, "subject": name[3:5], "y_true": y_true, "y_pred": y_pred, "metrics": record_metrics})
        print(f"{name}: epochs={len(y_true)} acc={record_metrics['accuracy']:.4f} macro_f1={record_metrics['macro_f1']:.4f}")

    from sklearn.metrics import confusion_matrix

    all_true = np.concatenate([item["y_true"] for item in records])
    all_pred = np.concatenate([item["y_pred"] for item in records])
    aggregate = metric_dict(all_true, all_pred)
    aggregate.update(
        {
            "protocol": "frozen-source-checkpoint-logit-ensemble",
            "source_run_dir": str(args.run_dir),
            "external_np_data_dir": str(args.np_data_dir),
            "n_source_checkpoints": len(source_models),
            "n_records": len(records),
            "n_subjects": len({item["subject"] for item in records}),
            "n_epochs": int(len(all_true)),
            "tta_transforms": transforms,
            "target_label_use": "metrics only; no target labels used during inference or selection",
            "subject_bootstrap_95ci": bootstrap_subject_ci(records, args.bootstrap_samples, args.seed),
        }
    )
    record_rows = []
    for item in records:
        row = {"record": item["name"], "subject": item["subject"], "n_epochs": len(item["y_true"])}
        row.update({key: item["metrics"][key] for key in ("accuracy", "macro_f1", "weighted_f1", "kappa")})
        record_rows.append(row)
    with (args.output_dir / "record_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record_rows[0]))
        writer.writeheader()
        writer.writerows(record_rows)
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    np.save(args.output_dir / "pooled_y_true.npy", all_true)
    np.save(args.output_dir / "pooled_y_pred.npy", all_pred)
    np.save(args.output_dir / "pooled_confusion_matrix.npy", confusion_matrix(all_true, all_pred, labels=range(5)))
    print(json.dumps({key: aggregate[key] for key in ("accuracy", "macro_f1", "weighted_f1", "kappa")}, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate the frozen Sleep-EDF ensemble on SHHS EDF/XML pairs without NPZ staging.

SHHS labels are read only after the source-only ensemble has been fixed.  Each
record is processed and discarded in memory, so raw EDFs are never duplicated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any

import numpy as np
import pyedflib

try:
    from scripts.evaluate_attnsleep_external_ensemble import (
        metric_dict,
        predict_ensemble_logits,
        resolve_device,
    )
    from scripts.prepare_sleepedf20_multichannel import (
        EPOCH_SECONDS,
        UNKNOWN,
        physical_unit_scale_to_volts,
        resample_epoch,
    )
except ModuleNotFoundError:
    from evaluate_attnsleep_external_ensemble import metric_dict, predict_ensemble_logits, resolve_device
    from prepare_sleepedf20_multichannel import EPOCH_SECONDS, UNKNOWN, physical_unit_scale_to_volts, resample_epoch


STAGE_MAP = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "Stage 4 sleep|4": 3,
    "REM sleep|5": 4,
}
CHANNELS = ("EEG(sec)", "EEG", "EOG(L)", "EMG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shhs-root",
        type=Path,
        default=Path("/path/to/datasets/SHHS/polysomnography"),
    )
    parser.add_argument("--cohort", choices=["shhs1", "shhs2"], required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/attnsleep_multichannel_baseline_train/full20_quality_gate_4ch_emg_30epoch"),
    )
    parser.add_argument("--attnsleep-dir", type=Path, default=Path("external_baselines/AttnSleep"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--w-edge-mins", type=int, default=30)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def record_id(path: Path) -> str:
    return path.stem


def subject_id(record: str) -> str:
    return record.split("-", maxsplit=1)[1]


def xml_for(edf_path: Path, annotation_dir: Path) -> Path:
    return annotation_dir / f"{edf_path.stem}-nsrr.xml"


def read_labels(xml_path: Path, n_total_epochs: int) -> np.ndarray:
    labels = np.full(n_total_epochs, UNKNOWN, dtype=np.int8)
    root = element_tree.parse(xml_path).getroot()
    for event in root.iterfind(".//ScoredEvent"):
        if (event.findtext("EventType") or "").strip() != "Stages|Stages":
            continue
        concept = (event.findtext("EventConcept") or "").strip()
        label = STAGE_MAP.get(concept)
        if label is None:
            continue
        start = float(event.findtext("Start") or "nan")
        duration = float(event.findtext("Duration") or "nan")
        if not np.isfinite(start) or not np.isfinite(duration):
            raise ValueError(f"Invalid stage span in {xml_path}")
        start_epochs = start / EPOCH_SECONDS
        duration_epochs = duration / EPOCH_SECONDS
        if not np.isclose(start_epochs, round(start_epochs), atol=1e-6) or not np.isclose(duration_epochs, round(duration_epochs), atol=1e-6):
            raise ValueError(f"Non-30s stage span in {xml_path}: start={start}, duration={duration}")
        begin = int(round(start_epochs))
        end = min(begin + int(round(duration_epochs)), n_total_epochs)
        if begin < n_total_epochs:
            labels[begin:end] = label
    if not np.any(labels != UNKNOWN):
        raise ValueError(f"No recognized sleep stages in {xml_path}")
    return labels


def read_channel_epochs(reader: pyedflib.EdfReader, channel: str, n_epochs: int) -> tuple[np.ndarray, dict[str, Any]]:
    labels = reader.getSignalLabels()
    if channel not in labels:
        raise KeyError(f"Missing {channel!r}; channels={labels}")
    index = labels.index(channel)
    sfreq = float(reader.getSampleFrequency(index))
    samples_per_epoch = int(round(sfreq * EPOCH_SECONDS))
    if samples_per_epoch <= 0:
        raise ValueError(f"Invalid sampling rate for {channel}: {sfreq}")
    values = reader.readSignal(index).astype(np.float32)
    usable = n_epochs * samples_per_epoch
    if len(values) < usable:
        raise ValueError(f"Insufficient {channel} samples: {len(values)} < {usable}")
    scale = physical_unit_scale_to_volts(str(reader.getPhysicalDimension(index)))
    epochs = values[:usable].reshape(n_epochs, samples_per_epoch) * scale
    if samples_per_epoch == 3000:
        output = epochs.astype(np.float32, copy=False)
    elif 3000 % samples_per_epoch == 0:
        output = np.repeat(epochs, 3000 // samples_per_epoch, axis=1).astype(np.float32, copy=False)
    else:
        output = np.stack([resample_epoch(epoch, 3000) for epoch in epochs]).astype(np.float32)
    return output, {
        "name": channel,
        "sfreq": sfreq,
        "physical_unit": str(reader.getPhysicalDimension(index)),
        "scale_to_volts": scale,
        "native_epoch_samples": samples_per_epoch,
    }


def load_record(edf_path: Path, xml_path: Path, w_edge_mins: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = reader.getSignalLabels()
        indices = [labels.index(channel) for channel in CHANNELS if channel in labels]
        if len(indices) != len(CHANNELS):
            missing = sorted(set(CHANNELS) - set(labels))
            raise KeyError(f"Missing required channels {missing}")
        n_epochs = min(
            int(reader.getNSamples()[index] // (reader.getSampleFrequency(index) * EPOCH_SECONDS))
            for index in indices
        )
        y_all = read_labels(xml_path, n_epochs)
        epoch_channels, channel_meta = [], []
        for channel in CHANNELS:
            values, meta = read_channel_epochs(reader, channel, n_epochs)
            epoch_channels.append(values)
            channel_meta.append(meta)
        duration = float(reader.getFileDuration())
    finally:
        reader.close()

    valid = np.flatnonzero(y_all != UNKNOWN)
    if not len(valid):
        raise ValueError("No valid labeled epochs")
    y = y_all[valid]
    x_channels = [values[valid] for values in epoch_channels]
    non_wake = np.flatnonzero(y != 0)
    if not len(non_wake):
        raise ValueError("No non-wake epochs")
    edge = int(w_edge_mins) * 2
    keep = np.arange(max(int(non_wake[0]) - edge, 0), min(int(non_wake[-1]) + edge + 1, len(y)))
    y = y[keep].astype(np.int64)
    x = np.stack([values[keep] for values in x_channels], axis=1).astype(np.float32)
    return x, y, {
        "n_signal_epochs": n_epochs,
        "n_valid_labeled_epochs": int(len(valid)),
        "n_epochs": int(len(y)),
        "file_duration_seconds": duration,
        "channels": channel_meta,
        "label_counts": np.bincount(y, minlength=5).astype(int).tolist(),
    }


def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["record"] for row in csv.DictReader(handle)}


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edf_dir = args.shhs_root / "edfs" / args.cohort
    annotation_dir = args.shhs_root / "annotations-events-nsrr" / args.cohort
    edf_paths = sorted(edf_dir.glob("*.edf"))
    if not edf_paths or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing EDFs or NSRR annotations for {args.cohort}")
    pairs = [(path, xml_for(path, annotation_dir)) for path in edf_paths]
    pairs = [(edf, xml) for edf, xml in pairs if xml.exists()]
    if args.max_records:
        pairs = pairs[: args.max_records]
    if not pairs:
        raise ValueError("No matched EDF/XML pairs")

    sys.path.insert(0, str(args.attnsleep_dir.resolve()))
    import torch
    from model.model import AttnSleep
    from torch.func import stack_module_state

    device = resolve_device(args.device)
    metadata = [json.loads((args.run_dir / f"fold_{fold_id:02d}" / "metadata.json").read_text()) for fold_id in range(args.num_folds)]
    if any(item["input_channels"] != 4 or item["channel_fusion"] != "quality-gated" for item in metadata):
        raise ValueError("Expected four-channel quality-gated source checkpoints")
    models = []
    for fold_id in range(args.num_folds):
        checkpoint = torch.load(args.run_dir / f"fold_{fold_id:02d}" / "checkpoint_best.pt", map_location=device, weights_only=False)
        model = AttnSleep(input_channels=4, channel_fusion="quality-gated", channel_dropout=0.15).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    params, buffers = stack_module_state(models)
    base_model = models[0]
    print(f"Loaded {len(models)} frozen source checkpoints on {device}; matched pairs={len(pairs)}")

    records_path = args.output_dir / "record_metrics.csv"
    errors_path = args.output_dir / "errors.csv"
    predictions_dir = args.output_dir / "record_predictions"
    predictions_dir.mkdir(exist_ok=True)
    done = load_completed(records_path) if args.resume else set()
    rows = []
    if done:
        with records_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        missing_predictions = [name for name in done if not (predictions_dir / f"{name}.npz").exists()]
        if missing_predictions:
            raise FileNotFoundError(
                f"--resume found metric rows without predictions, e.g. {missing_predictions[:3]}"
            )

    for number, (edf_path, xml_path) in enumerate(pairs, start=1):
        name = record_id(edf_path)
        if name in done:
            continue
        try:
            x, y_true, record_meta = load_record(edf_path, xml_path, args.w_edge_mins)
            logits = predict_ensemble_logits(base_model, params, buffers, x, args.batch_size, [{"kind": "identity"}], device)
            y_pred = logits.argmax(axis=1).astype(np.int64)
            metrics = scalar_metrics(y_true, y_pred)
            row = {
                "record": name,
                "subject": subject_id(name),
                "n_epochs": len(y_true),
                **metrics,
                "n_signal_epochs": record_meta["n_signal_epochs"],
                "n_valid_labeled_epochs": record_meta["n_valid_labeled_epochs"],
            }
            append_row(records_path, row)
            rows.append(row)
            np.savez_compressed(
                predictions_dir / f"{name}.npz",
                y_true=y_true.astype(np.int8),
                y_pred=y_pred.astype(np.int8),
            )
            print(f"{number}/{len(pairs)} {name}: n={len(y_true)} acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)
        except Exception as error:
            error_row = {"record": name, "xml": xml_path.name, "error": repr(error)}
            append_row(errors_path, error_row)
            print(f"{number}/{len(pairs)} {name}: ERROR {error!r}", flush=True)

    completed_names = [str(row["record"]) for row in rows]
    y_true_parts, y_pred_parts, spans = [], [], []
    offset = 0
    for name in completed_names:
        with np.load(predictions_dir / f"{name}.npz") as prediction:
            current_true = np.asarray(prediction["y_true"], dtype=np.int8)
            current_pred = np.asarray(prediction["y_pred"], dtype=np.int8)
        y_true_parts.append(current_true)
        y_pred_parts.append(current_pred)
        spans.append([name, subject_id(name), offset, offset + len(current_true)])
        offset += len(current_true)
    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        spans=np.asarray(spans, dtype=object),
    )
    aggregate = metric_dict(y_true, y_pred)
    aggregate.update(
        {
            "protocol": "frozen 20-fold source-checkpoint logits ensemble; identity-only inference; target labels metrics only",
            "cohort": args.cohort,
            "shhs_root": str(args.shhs_root),
            "edf_directory": str(edf_dir),
            "annotation_directory": str(annotation_dir),
            "n_edfs": len(edf_paths),
            "n_matched_edf_xml_pairs": len(pairs),
            "n_evaluated_records": len(rows),
            "n_epochs": int(len(y_true)),
            "n_subjects": len({str(row["subject"]) for row in rows}),
            "input_channel_contract": {
                "source_order": ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"],
                "shhs_order": ["C3/A2 (EEG(sec))", "C4/A1 (EEG)", "left EOG", "chin EMG"],
                "target_sfreq_hz": 100,
                "normalization": "physical-volts",
            },
            "label_mapping": {key: value for key, value in STAGE_MAP.items()},
            "w_edge_mins": args.w_edge_mins,
            "errors": 0 if not errors_path.exists() else sum(1 for _ in errors_path.open(encoding="utf-8")) - 1,
        }
    )
    (args.output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({key: aggregate[key] for key in ("accuracy", "macro_f1", "weighted_f1", "kappa")}, indent=2))


if __name__ == "__main__":
    main()

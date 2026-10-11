"""Zero-shot AttnSleep evaluation on ISRUC-SLEEP Cohort I raw EDF/REC files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyedflib

try:
    from scripts.evaluate_attnsleep_external_ensemble import metric_dict, predict_ensemble_logits, resolve_device
    from scripts.prepare_sleepedf20_multichannel import physical_unit_scale_to_volts, resample_epoch
except ModuleNotFoundError:
    from evaluate_attnsleep_external_ensemble import metric_dict, predict_ensemble_logits, resolve_device
    from prepare_sleepedf20_multichannel import physical_unit_scale_to_volts, resample_epoch


LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}
EPOCH_SECONDS = 30
TARGET_SAMPLES = 3000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isruc-root", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, default=Path("outputs/attnsleep_multichannel_baseline_train/full20_quality_gate_4ch_emg_30epoch"))
    parser.add_argument("--attnsleep-dir", type=Path, default=Path("external_baselines/AttnSleep"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-suffix", default="1")
    parser.add_argument("--fourth-channel", default="X1")
    parser.add_argument("--eog-channel", default="auto", help="auto or an explicit EDF EOG label such as LOC-A2/ROC-A1")
    parser.add_argument("--subjects", default="", help="comma-separated subject IDs; empty evaluates all")
    parser.add_argument("--normalization", choices=["physical-volts", "record-zscore"], default="physical-volts")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def labels_from_txt(path: Path, n_epochs: int) -> np.ndarray:
    values = [int(line.strip()) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    values = values[:n_epochs]
    mapped = np.asarray([LABEL_MAP.get(value, -1) for value in values], dtype=np.int64)
    if len(mapped) < n_epochs:
        n_epochs = len(mapped)
    mapped = mapped[:n_epochs]
    if np.any(mapped < 0):
        raise ValueError(f"Unsupported ISRUC labels in {path}: {sorted(set(values) - set(LABEL_MAP))}")
    return mapped


def resolve_channel(reader: pyedflib.EdfReader, role: str, requested_fourth: str, requested_eog: str = "auto") -> str:
    labels = reader.getSignalLabels()
    aliases = {
        "eeg_c3": ["C3-A2", "C3-M2", "C3"],
        "eeg_c4": ["C4-A1", "C4-M1", "C4"],
        "eog": ["LOC-A2", "E1-M2", "E2-M1", "LOC", "ROC"],
    }
    if role == "emg":
        if requested_fourth in labels:
            return requested_fourth
        for index, label in enumerate(labels):
            if str(reader.getTransducer(index)).strip().lower() == "chin_channel":
                return label
        raise KeyError(f"Missing chin EMG channel {requested_fourth!r}; available={labels}")
    if role == "eog" and requested_eog != "auto":
        if requested_eog not in labels:
            raise KeyError(f"Missing requested EOG channel {requested_eog!r}; available={labels}")
        return requested_eog
    for candidate in aliases[role]:
        if candidate in labels:
            return candidate
    raise KeyError(f"Missing {role} channel; available={labels}")


def read_signal(reader: pyedflib.EdfReader, name: str, n_epochs: int) -> np.ndarray:
    labels = reader.getSignalLabels()
    if name not in labels:
        raise KeyError(f"Missing channel {name!r}; available={labels}")
    index = labels.index(name)
    sfreq = float(reader.getSampleFrequency(index))
    samples = int(round(sfreq * EPOCH_SECONDS))
    values = reader.readSignal(index).astype(np.float32) * physical_unit_scale_to_volts(str(reader.getPhysicalDimension(index)))
    usable = min(n_epochs * samples, len(values))
    n_epochs = usable // samples
    epochs = values[: n_epochs * samples].reshape(n_epochs, samples)
    if samples == TARGET_SAMPLES:
        return epochs.astype(np.float32)
    return np.stack([resample_epoch(epoch, TARGET_SAMPLES) for epoch in epochs]).astype(np.float32)


def load_record(path: Path, annotation: Path, fourth_channel: str, eog_channel: str = "auto", normalization: str = "physical-volts") -> tuple[np.ndarray, np.ndarray, dict]:
    reader = pyedflib.EdfReader(str(path))
    try:
        required = [resolve_channel(reader, "eeg_c3", fourth_channel, eog_channel), resolve_channel(reader, "eeg_c4", fourth_channel, eog_channel), resolve_channel(reader, "eog", fourth_channel, eog_channel), resolve_channel(reader, "emg", fourth_channel, eog_channel)]
        labels = reader.getSignalLabels()
        indices = [labels.index(name) for name in required]
        n_epochs = min(int(reader.getNSamples()[i] // (reader.getSampleFrequency(i) * EPOCH_SECONDS)) for i in indices)
        c3 = read_signal(reader, required[0], n_epochs)
        c4 = read_signal(reader, required[1], n_epochs)
        loc = read_signal(reader, required[2], n_epochs)
        fourth = read_signal(reader, required[3], n_epochs)
    finally:
        reader.close()
    n_epochs = min(len(c3), len(c4), len(loc), len(fourth))
    channels = [c3, c4, loc, fourth]
    if normalization == "record-zscore":
        channels = [(channel - np.mean(channel)) / max(float(np.std(channel)), 1e-8) for channel in channels]
        c3, c4, loc, fourth = channels
    y = labels_from_txt(annotation, n_epochs)
    n_epochs = min(n_epochs, len(y))
    x = np.stack([c3[:n_epochs], c4[:n_epochs], loc[:n_epochs], fourth[:n_epochs]], axis=1).astype(np.float32)
    return x, y[:n_epochs], {"n_epochs": n_epochs, "channels": required, "fourth_channel_contract": "X1 or EDF transducer=Chin_Channel"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(
        (path.parent for path in args.isruc_root.glob("*/[0-9]*.rec")),
        key=lambda path: int(path.name) if path.name.isdigit() else path.name,
    )
    unique = []
    seen = set()
    for path in subjects:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if args.max_subjects:
        unique = unique[: args.max_subjects]
    if args.subjects:
        wanted = {item.strip() for item in args.subjects.split(",") if item.strip()}
        unique = [path for path in unique if path.name in wanted]
    if not unique:
        raise ValueError(f"No ISRUC subject/*.rec files found under {args.isruc_root}")
    sys.path.insert(0, str(args.attnsleep_dir.resolve()))
    import torch
    from model.model import AttnSleep
    from torch.func import stack_module_state

    device = resolve_device(args.device)
    models = []
    for fold_id in range(20):
        checkpoint = torch.load(args.source_run_dir / f"fold_{fold_id:02d}" / "checkpoint_best.pt", map_location=device, weights_only=False)
        model = AttnSleep(input_channels=4, channel_fusion="quality-gated", channel_dropout=0.15).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    params, buffers = stack_module_state(models)
    rows, all_true, all_pred = [], [], []
    for number, subject_dir in enumerate(unique, 1):
        rec = subject_dir / f"{subject_dir.name}.rec"
        annotation = subject_dir / f"{subject_dir.name}_{args.annotation_suffix}.txt"
        try:
            x, y_true, meta = load_record(rec, annotation, args.fourth_channel, args.eog_channel, args.normalization)
            logits = predict_ensemble_logits(models[0], params, buffers, x, args.batch_size, [{"kind": "identity"}], device)
            y_pred = logits.argmax(axis=1).astype(np.int64)
            metric = metric_dict(y_true, y_pred)
            rows.append({"subject": subject_dir.name, "record": rec.name, "n_epochs": len(y_true), **{k: v for k, v in metric.items() if k != "classification_report"}})
            all_true.append(y_true)
            all_pred.append(y_pred)
            np.savez_compressed(args.output_dir / f"{subject_dir.name}.npz", y_true=y_true.astype(np.int8), y_pred=y_pred.astype(np.int8), logits=logits.astype(np.float32))
            print(f"{number}/{len(unique)} {subject_dir.name} n={len(y_true)} macro_f1={metric['macro_f1']:.4f}", flush=True)
        except Exception as error:
            rows.append({"subject": subject_dir.name, "record": rec.name, "error": repr(error)})
            print(f"{number}/{len(unique)} {subject_dir.name}: ERROR {error!r}", flush=True)
    valid = [row for row in rows if "error" not in row]
    if not valid:
        raise RuntimeError("No ISRUC records were successfully evaluated")
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    report = {
        "protocol": "ISRUC-SLEEP Cohort I source-only zero-shot evaluation; labels not used for model selection or adaptation",
        "isruc_root": str(args.isruc_root), "annotation_suffix": args.annotation_suffix, "n_subjects_requested": len(unique), "n_subjects_evaluated": len(valid), "n_errors": len(rows) - len(valid),
        "source_run_dir": str(args.source_run_dir), "device": str(device), "metrics": metric_dict(y_true, y_pred),
        "input_contract": {"channel_roles": {"eeg_c3": ["C3-A2", "C3-M2", "C3"], "eeg_c4": ["C4-A1", "C4-M1", "C4"], "eog": ["LOC-A2", "E1-M2", "E2-M1", "LOC", "ROC"], "chin_emg": [args.fourth_channel, "EDF transducer=Chin_Channel"]}, "source_montage_note": "ISRUC montage and labels vary by recording; channels are selected by explicit aliases and EDF transducer semantics, with no learned target-domain calibration", "target_sfreq_hz": 100, "epoch_seconds": 30, "label_map": LABEL_MAP},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "record_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({key: report["metrics"][key] for key in ("accuracy", "macro_f1", "weighted_f1", "kappa")}, indent=2))


if __name__ == "__main__":
    main()

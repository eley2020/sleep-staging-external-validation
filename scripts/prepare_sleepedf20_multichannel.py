"""Prepare EEG/EOG/EMG Sleep-EDF-20 NPZ files.

The existing AttnSleep baseline uses one Fpz-Cz EEG channel. This script keeps
the same Sleep-EDF-20 records and label trimming policy, but stores multiple
signal types per epoch so later models can use EEG, EOG, and EMG together.

Output `x` shape is `(n_epochs, n_channels, 3000)`, where every 30-second epoch
is represented at a target 100 Hz grid. Sleep-EDF's EMG channel is 1 Hz, so it is
upsampled by deterministic repetition to 3000 samples per epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyedflib
from scipy import signal as scipy_signal


ANN_TO_LABEL = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5,
}

UNKNOWN = 5
EPOCH_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/attnsleep_sleepedf20_raw"),
        help="Directory containing Sleep-EDF-20 PSG/Hypnogram EDF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sleepedf20_multichannel_npz/eeg_eog_emg"),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["EEG Fpz-Cz", "EOG horizontal", "EMG submental"],
    )
    parser.add_argument("--target-sfreq", type=float, default=100.0)
    parser.add_argument("--w-edge-mins", type=int, default=30)
    parser.add_argument("--bandpass-low", type=float, default=0.0)
    parser.add_argument("--bandpass-high", type=float, default=0.0)
    parser.add_argument("--notch-freq", type=float, default=0.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument(
        "--normalization",
        choices=["record-zscore", "crop-zscore", "physical-volts"],
        default="record-zscore",
        help=(
            "Signal scaling policy. 'record-zscore' preserves the previous behavior; "
            "'crop-zscore' standardizes after sleep-window trimming; "
            "'physical-volts' matches MNE/AttnSleep physical-unit scaling."
        ),
    )
    parser.add_argument(
        "--reference-npz-dir",
        type=Path,
        default=Path("data/attnsleep_sleepedf20_npz/fpzcz"),
        help="Optional single-channel reference NPZ directory for y alignment audit.",
    )
    return parser.parse_args()


def normalize_annotation(text: Any) -> str:
    if isinstance(text, bytes):
        text = text.decode(errors="ignore")
    return str(text).strip()


def read_labels(hypnogram_path: Path, n_total_epochs: int) -> np.ndarray:
    try:
        ann = pyedflib.EdfReader(str(hypnogram_path))
        try:
            onsets, durations, stages = ann.readAnnotations()
        finally:
            ann.close()
    except OSError as error:
        # Some Sleep-Telemetry hypnograms have EDF+ headers rejected by
        # pyedflib although their annotation stream is valid.
        try:
            import mne
        except ModuleNotFoundError as import_error:
            raise RuntimeError(
                f"pyedflib could not read {hypnogram_path}; install MNE for the "
                "EDF+ annotation fallback"
            ) from import_error
        annotations = mne.read_annotations(str(hypnogram_path))
        onsets = annotations.onset
        durations = annotations.duration
        stages = annotations.description
        print(f"MNE annotation fallback for {hypnogram_path.name}: {error}")

    y = np.full(n_total_epochs, UNKNOWN, dtype=np.int32)
    for onset, duration, stage in zip(onsets, durations, stages):
        onset_sec = int(round(float(onset)))
        duration_sec = int(round(float(duration)))
        stage_name = normalize_annotation(stage)
        if stage_name not in ANN_TO_LABEL:
            raise KeyError(f"Unknown annotation label in {hypnogram_path.name}: {stage_name!r}")
        if onset_sec % EPOCH_SECONDS != 0 or duration_sec % EPOCH_SECONDS != 0:
            raise ValueError(
                f"Annotation is not aligned to {EPOCH_SECONDS}s epochs in "
                f"{hypnogram_path.name}: onset={onset_sec}, duration={duration_sec}"
            )
        start_epoch = onset_sec // EPOCH_SECONDS
        end_epoch = min(start_epoch + duration_sec // EPOCH_SECONDS, n_total_epochs)
        if start_epoch < n_total_epochs:
            y[start_epoch:end_epoch] = ANN_TO_LABEL[stage_name]

    if not np.any(y != UNKNOWN):
        raise ValueError(f"No labels read from {hypnogram_path}")
    return y


def resample_epoch(epoch: np.ndarray, target_samples: int) -> np.ndarray:
    if len(epoch) == target_samples:
        return epoch.astype(np.float32, copy=False)
    if target_samples % len(epoch) == 0:
        return np.repeat(epoch, target_samples // len(epoch)).astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, num=len(epoch), endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=target_samples, endpoint=False)
    return np.interp(new_x, old_x, epoch).astype(np.float32)


def zscore_channel(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return x.astype(np.float32, copy=False) - mean
    return ((x - mean) / std).astype(np.float32, copy=False)


def physical_unit_scale_to_volts(unit: str) -> float:
    normalized = unit.strip().replace("μ", "u").replace("µ", "u").lower()
    scales = {
        "v": 1.0,
        "mv": 1e-3,
        "uv": 1e-6,
        "nv": 1e-9,
    }
    if normalized not in scales:
        raise ValueError(f"Unsupported physical unit for volts conversion: {unit!r}")
    return scales[normalized]


def filter_signal(
    signal_values: np.ndarray,
    sfreq: float,
    bandpass_low: float,
    bandpass_high: float,
    notch_freq: float,
    filter_order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    nyquist = sfreq / 2.0
    filtered = signal_values.astype(np.float32, copy=False)
    meta: dict[str, Any] = {
        "bandpass_low": None,
        "bandpass_high": None,
        "notch_freq": None,
        "filter_order": filter_order,
        "skipped": [],
    }

    if bandpass_low > 0 or bandpass_high > 0:
        low = bandpass_low if bandpass_low > 0 else None
        high = bandpass_high if bandpass_high > 0 else None
        if low is not None and low >= nyquist:
            meta["skipped"].append(f"bandpass_low {low:g}Hz >= Nyquist {nyquist:g}Hz")
            low = None
        if high is not None and high >= nyquist:
            meta["skipped"].append(f"bandpass_high {high:g}Hz >= Nyquist {nyquist:g}Hz")
            high = None
        if low is not None and high is not None and low >= high:
            meta["skipped"].append(f"invalid bandpass {low:g}-{high:g}Hz")
            low = None
            high = None
        if low is not None or high is not None:
            if low is not None and high is not None:
                btype = "bandpass"
                wn: float | list[float] = [low, high]
            elif low is not None:
                btype = "highpass"
                wn = low
            else:
                btype = "lowpass"
                wn = high
            sos = scipy_signal.butter(filter_order, wn, btype=btype, fs=sfreq, output="sos")
            filtered = scipy_signal.sosfiltfilt(sos, filtered).astype(np.float32, copy=False)
            meta["bandpass_low"] = low
            meta["bandpass_high"] = high

    if notch_freq > 0:
        if notch_freq >= nyquist:
            meta["skipped"].append(f"notch_freq {notch_freq:g}Hz >= Nyquist {nyquist:g}Hz")
        else:
            b, a = scipy_signal.iirnotch(w0=notch_freq, Q=30.0, fs=sfreq)
            filtered = scipy_signal.filtfilt(b, a, filtered).astype(np.float32, copy=False)
            meta["notch_freq"] = notch_freq

    return filtered, meta


def read_channel_epochs(
    reader: pyedflib.EdfReader,
    channel_name: str,
    n_total_epochs: int,
    target_samples: int,
    normalization: str,
    bandpass_low: float,
    bandpass_high: float,
    notch_freq: float,
    filter_order: int,
) -> tuple[np.ndarray, float, int, str, float | None, dict[str, Any]]:
    labels = reader.getSignalLabels()
    if channel_name not in labels:
        raise KeyError(f"Channel {channel_name!r} not found. Available: {labels}")
    idx = labels.index(channel_name)
    sfreq = float(reader.getSampleFrequency(idx))
    physical_unit = str(reader.getPhysicalDimension(idx))
    native_epoch_samples = int(round(sfreq * EPOCH_SECONDS))
    if native_epoch_samples <= 0:
        raise ValueError(f"Invalid sampling rate for {channel_name}: {sfreq}")

    signal = reader.readSignal(idx).astype(np.float32)
    # pyedflib can print a read failure and return zero-filled arrays without
    # raising. Such a channel must not silently become a valid training input.
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite signal in channel {channel_name!r}")
    if not np.any(signal != 0):
        raise ValueError(f"Unexpected all-zero native channel {channel_name!r}; check EDF read integrity")
    signal, filter_meta = filter_signal(
        signal,
        sfreq,
        bandpass_low,
        bandpass_high,
        notch_freq,
        filter_order,
    )
    usable_samples = n_total_epochs * native_epoch_samples
    if len(signal) < usable_samples:
        raise ValueError(
            f"{channel_name} has too few samples: {len(signal)} < {usable_samples}"
        )
    signal = signal[:usable_samples]
    epochs = signal.reshape(n_total_epochs, native_epoch_samples)
    scale_to_volts: float | None = None
    if normalization == "record-zscore":
        epochs = zscore_channel(epochs)
    elif normalization == "physical-volts":
        scale_to_volts = physical_unit_scale_to_volts(physical_unit)
        epochs = (epochs * scale_to_volts).astype(np.float32, copy=False)
    elif normalization != "crop-zscore":
        raise ValueError(f"Unknown normalization mode: {normalization}")
    if native_epoch_samples == target_samples:
        return (
            epochs.astype(np.float32, copy=False),
            sfreq,
            native_epoch_samples,
            physical_unit,
            scale_to_volts,
            filter_meta,
        )
    out = np.stack([resample_epoch(epoch, target_samples) for epoch in epochs]).astype(np.float32)
    return out, sfreq, native_epoch_samples, physical_unit, scale_to_volts, filter_meta


def find_hypnogram(raw_dir: Path, psg_path: Path) -> Path:
    matches = sorted(raw_dir.glob(f"{psg_path.name[:6]}*-Hypnogram.edf"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one hypnogram for {psg_path.name}, found {len(matches)}"
        )
    return matches[0]


def prepare_one(
    psg_path: Path,
    hypnogram_path: Path,
    channels: list[str],
    target_samples: int,
    w_edge_mins: int,
    normalization: str,
    bandpass_low: float,
    bandpass_high: float,
    notch_freq: float,
    filter_order: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reader = pyedflib.EdfReader(str(psg_path))
    try:
        n_total_epochs = min(
            int(reader.getNSamples()[i] // (reader.getSampleFrequency(i) * EPOCH_SECONDS))
            for i in range(reader.signals_in_file)
            if reader.getSignalLabels()[i] in channels
        )
        channel_epochs = []
        channel_meta = []
        for channel_name in channels:
            epochs, sfreq, native_samples, physical_unit, scale_to_volts, filter_meta = read_channel_epochs(
                reader,
                channel_name,
                n_total_epochs,
                target_samples,
                normalization,
                bandpass_low,
                bandpass_high,
                notch_freq,
                filter_order,
            )
            channel_epochs.append(epochs)
            channel_meta.append(
                {
                    "name": channel_name,
                    "sfreq": sfreq,
                    "physical_unit": physical_unit,
                    "scale_to_volts": scale_to_volts,
                    "native_epoch_samples": native_samples,
                    "target_epoch_samples": target_samples,
                    "filter": filter_meta,
                }
            )
    finally:
        start_datetime = str(reader.getStartdatetime())
        file_duration = float(reader.getFileDuration())
        reader.close()

    # No live PSG EDF handle may coexist with the failed Hypnogram open/MNE
    # fallback: in pyedflib this can invalidate the PSG handle and zero its data.
    y_all = read_labels(hypnogram_path, n_total_epochs)
    valid_idx = np.where(y_all != UNKNOWN)[0]
    y = y_all[valid_idx].astype(np.int32)
    x_by_channel = [epochs[valid_idx] for epochs in channel_epochs]

    non_wake = np.where(y != 0)[0]
    if len(non_wake) == 0:
        raise ValueError(f"No non-wake epochs found for {psg_path.name}")
    edge_epochs = w_edge_mins * 2
    start_idx = max(int(non_wake[0]) - edge_epochs, 0)
    end_idx = min(int(non_wake[-1]) + edge_epochs, len(y) - 1)
    keep = np.arange(start_idx, end_idx + 1)
    y = y[keep]
    x_by_channel = [ch[keep] for ch in x_by_channel]
    if normalization == "crop-zscore":
        x_by_channel = [zscore_channel(ch) for ch in x_by_channel]
    x = np.stack(x_by_channel, axis=1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite prepared input for {psg_path.name}")
    if not np.any(x != 0):
        raise ValueError(f"Unexpected all-zero prepared record for {psg_path.name}")

    meta = {
        "psg_file": psg_path.name,
        "hypnogram_file": hypnogram_path.name,
        "start_datetime": start_datetime,
        "file_duration": file_duration,
        "n_all_epochs": int(n_total_epochs),
        "n_valid_labeled_epochs": int(len(valid_idx)),
        "n_epochs": int(len(y)),
        "normalization": normalization,
        "filter": {
            "bandpass_low": bandpass_low,
            "bandpass_high": bandpass_high,
            "notch_freq": notch_freq,
            "filter_order": filter_order,
        },
        "channels": channel_meta,
        "label_counts": np.bincount(y, minlength=5)[:5].astype(int).tolist(),
    }
    return x, y, meta


def audit_reference(reference_dir: Path, stem: str, y: np.ndarray) -> dict[str, Any]:
    ref_path = reference_dir / f"{stem}.npz"
    if not ref_path.exists():
        return {"reference_found": False}
    with np.load(ref_path, allow_pickle=True) as ref:
        ref_y = np.asarray(ref["y"], dtype=np.int32)
    same = bool(np.array_equal(ref_y, y))
    return {
        "reference_found": True,
        "reference_y_len": int(len(ref_y)),
        "reference_y_counts": np.bincount(ref_y, minlength=5)[:5].astype(int).tolist(),
        "y_equal_reference": same,
        "first_mismatch": (
            None
            if same or len(ref_y) != len(y)
            else int(np.where(ref_y != y)[0][0])
            if np.any(ref_y != y)
            else None
        ),
    }


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    output_dir = args.output_dir
    if output_dir.exists() and (any(output_dir.glob("*.npz")) or (output_dir / "summary.json").exists()):
        raise FileExistsError(f"Refusing to replace existing prepared data in {output_dir}; use a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_samples = int(round(args.target_sfreq * EPOCH_SECONDS))
    psg_files = sorted(raw_dir.glob("*PSG.edf"))
    if not psg_files:
        raise FileNotFoundError(f"No PSG EDF files found in {raw_dir}")

    summary: dict[str, Any] = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "channels": args.channels,
        "target_sfreq": args.target_sfreq,
        "target_epoch_samples": target_samples,
        "w_edge_mins": args.w_edge_mins,
        "normalization": args.normalization,
        "filter": {
            "bandpass_low": args.bandpass_low,
            "bandpass_high": args.bandpass_high,
            "notch_freq": args.notch_freq,
            "filter_order": args.filter_order,
        },
        "records": [],
    }
    total_counts = np.zeros(5, dtype=np.int64)
    total_epochs = 0
    ref_mismatches = []

    for psg_path in psg_files:
        hypnogram_path = find_hypnogram(raw_dir, psg_path)
        x, y, meta = prepare_one(
            psg_path,
            hypnogram_path,
            args.channels,
            target_samples,
            args.w_edge_mins,
            args.normalization,
            args.bandpass_low,
            args.bandpass_high,
            args.notch_freq,
            args.filter_order,
        )
        stem = psg_path.name.replace("-PSG.edf", "")
        ref_audit = audit_reference(args.reference_npz_dir, stem, y)
        meta["reference_audit"] = ref_audit
        if ref_audit.get("reference_found") and not ref_audit.get("y_equal_reference"):
            ref_mismatches.append(stem)

        np.savez_compressed(
            output_dir / f"{stem}.npz",
            x=x,
            y=y,
            fs=np.float32(args.target_sfreq),
            ch_labels=np.asarray(args.channels),
            normalization=np.asarray(args.normalization),
            filter_config=np.asarray(
                json.dumps(
                    {
                        "bandpass_low": args.bandpass_low,
                        "bandpass_high": args.bandpass_high,
                        "notch_freq": args.notch_freq,
                        "filter_order": args.filter_order,
                    },
                    sort_keys=True,
                )
            ),
            source_psg=psg_path.name,
            source_hypnogram=hypnogram_path.name,
        )
        summary["records"].append(meta)
        total_counts += np.bincount(y, minlength=5)[:5]
        total_epochs += len(y)
        print(
            f"{stem}: x={x.shape} y={y.shape} counts="
            f"{np.bincount(y, minlength=5)[:5].astype(int).tolist()} "
            f"ref_equal={ref_audit.get('y_equal_reference')}"
        )

    summary["n_records"] = len(summary["records"])
    summary["n_epochs"] = int(total_epochs)
    summary["label_counts"] = total_counts.astype(int).tolist()
    summary["reference_mismatch_records"] = ref_mismatches
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("WROTE", output_dir)
    print("TOTAL", summary["n_epochs"], summary["label_counts"])
    print("REFERENCE_MISMATCHES", ref_mismatches)


if __name__ == "__main__":
    main()

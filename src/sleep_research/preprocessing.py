from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import signal
from tqdm import tqdm
import wfdb

from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    REFERENCE_RECORDS,
    STAGE_TO_INDEX,
    ReproductionConfig,
)


def read_records_file(data_dir: Path) -> list[str]:
    records_path = data_dir / "RECORDS"
    if not records_path.exists():
        return REFERENCE_RECORDS
    return [
        line.strip()
        for line in records_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def first_stage_token(aux_note: str) -> str | None:
    if not aux_note:
        return None
    token = aux_note.replace("\x00", "").strip().split(maxsplit=1)[0]
    if token == "M":
        token = "MT"
    if token not in {"W", "R", "MT", "1", "2", "3", "4"}:
        return None
    return token


def stage_to_label(stage: str | None) -> int | None:
    if stage is None or stage == "MT":
        return None
    return STAGE_TO_INDEX.get(stage)


def eeg_channel_index(signal_names: list[str]) -> int:
    for idx, name in enumerate(signal_names):
        if "EEG" in name.upper():
            return idx
    raise ValueError(f"No EEG channel found in signal names: {signal_names}")


def resample_epoch(epoch: np.ndarray, target_points: int) -> np.ndarray:
    if target_points % 3 != 0:
        raise ValueError("target_points must be divisible by 3 for the reference CNN reshape.")
    if len(epoch) == target_points:
        return epoch.astype(np.float32, copy=False)
    return signal.resample(epoch, target_points).astype(np.float32)


def normalize_epoch(epoch: np.ndarray, clip: float = 8.0) -> np.ndarray:
    epoch = np.asarray(epoch, dtype=np.float32)
    mean = float(np.nanmean(epoch))
    std = float(np.nanstd(epoch))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    out = (epoch - mean) / std
    return np.clip(out, -clip, clip).astype(np.float32)


def transform_epoch(epoch: np.ndarray, cfg: ReproductionConfig) -> np.ndarray:
    """Apply the input scaling used by the selected reproduction mode."""
    epoch = np.asarray(epoch, dtype=np.float32)
    if cfg.input_normalization == "paper_scale":
        return (epoch / float(cfg.input_scale)).astype(np.float32)
    if cfg.input_normalization == "zscore":
        return normalize_epoch(epoch)
    if cfg.input_normalization == "none":
        return epoch.astype(np.float32, copy=False)
    raise ValueError(
        "input_normalization must be one of: paper_scale, zscore, none; "
        f"got {cfg.input_normalization!r}"
    )


def load_record_epochs(record_name: str, cfg: ReproductionConfig) -> tuple[np.ndarray, np.ndarray]:
    record_path = cfg.data_dir / record_name
    record = wfdb.rdrecord(str(record_path), physical=True)
    annotation = wfdb.rdann(str(record_path), "st")

    eeg_idx = eeg_channel_index(record.sig_name)
    eeg = record.p_signal[:, eeg_idx].astype(np.float32)

    epochs: list[np.ndarray] = []
    labels: list[int] = []
    for sample, aux_note in zip(annotation.sample, annotation.aux_note, strict=False):
        label = stage_to_label(first_stage_token(aux_note))
        if label is None:
            continue
        start = int(sample)
        if start <= 1:
            start = 0
        stop = start + cfg.raw_epoch_points
        if stop > len(eeg):
            continue
        epoch = eeg[start:stop]
        if not np.isfinite(epoch).all():
            continue
        epoch = resample_epoch(epoch, cfg.target_points)
        epoch = transform_epoch(epoch, cfg)
        epochs.append(epoch)
        labels.append(label)

    if not epochs:
        raise ValueError(f"No usable labeled EEG epochs found for {record_name}")
    return np.stack(epochs).astype(np.float32), np.asarray(labels, dtype=np.int64)


def build_npz_dataset(
    cfg: ReproductionConfig,
    records: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    records = records or read_records_file(cfg.data_dir)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.cache_dir / "mitbih_sleep_eeg_epochs.npz"
    meta_path = cfg.cache_dir / "mitbih_sleep_eeg_epochs.meta.json"
    if out_path.exists() and not overwrite:
        return out_path

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    summary: dict[str, dict[str, object]] = {}
    for record_name in tqdm(records, desc="Preprocessing WFDB records"):
        try:
            x, y = load_record_epochs(record_name, cfg)
        except Exception as exc:
            summary[record_name] = {"error": repr(exc)}
            continue
        xs.append(x)
        ys.append(y)
        groups.append(np.full(len(y), record_name))
        counts = np.bincount(y, minlength=cfg.n_classes).astype(int).tolist()
        summary[record_name] = {"epochs": int(len(y)), "class_counts": counts}

    if not xs:
        raise RuntimeError("No records were preprocessed successfully.")

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    groups_all = np.concatenate(groups, axis=0)
    np.savez_compressed(out_path, x=x_all, y=y_all, record=groups_all)
    meta_path.write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
                "records": summary,
                "total_epochs": int(len(y_all)),
                "class_counts": np.bincount(y_all, minlength=cfg.n_classes).astype(int).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def load_npz_dataset(path: Path, group_key: str = "record") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if group_key not in data.files:
        raise KeyError(f"Group key {group_key!r} not found in {path}; available keys: {data.files}")
    return data["x"].astype(np.float32), data["y"].astype(np.int64), data[group_key].astype(str)


def make_context_windows(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    out_labels: list[int] = []
    out_groups: list[str] = []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) < radius * 2 + 1:
            continue
        for pos in range(radius, len(idx) - radius):
            take = idx[pos - radius : pos + radius + 1]
            windows.append(features[take])
            out_labels.append(int(labels[idx[pos]]))
            out_groups.append(str(group))
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(out_labels, dtype=np.int64),
        np.asarray(out_groups),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MIT-BIH PSG EEG epochs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--records", nargs="*", default=None)
    parser.add_argument(
        "--input-normalization",
        choices=["paper_scale", "zscore", "none"],
        default="paper_scale",
        help="Input transform after resampling. paper_scale divides by 200.0 like paper_lable/cnn.py.",
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=200.0,
        help="Divisor used when --input-normalization paper_scale is selected.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReproductionConfig(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        input_normalization=args.input_normalization,
        input_scale=args.input_scale,
    )
    path = build_npz_dataset(cfg, records=args.records, overwrite=args.overwrite)
    print(path)


if __name__ == "__main__":
    main()

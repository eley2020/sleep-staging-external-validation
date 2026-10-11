"""Train AttnSleep on multi-channel Sleep-EDF-20 NPZ files.

This is the same strict subject-wise baseline contract as
``train_attnsleep_baseline.py``, but it consumes NPZ files whose ``x`` has shape
``(epochs, channels, samples)``. It is intended for the EEG/EOG/EMG dataset
created by ``prepare_sleepedf20_multichannel.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.channel_quality import (
        RecordQualityStats,
        fit_record_quality_stats,
        parse_index_pair,
        piecewise_channel_prior,
        split_train_validation_files_by_strategy,
        soft_channel_prior,
    )
except ModuleNotFoundError:
    from channel_quality import (
        RecordQualityStats,
        fit_record_quality_stats,
        parse_index_pair,
        piecewise_channel_prior,
        split_train_validation_files_by_strategy,
        soft_channel_prior,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--np-data-dir",
        type=Path,
        default=Path("data/sleepedf20_multichannel_npz/eeg_eog_emg"),
    )
    parser.add_argument("--attnsleep-dir", type=Path, default=Path("external_baselines/AttnSleep"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/attnsleep_multichannel_baseline_train"),
    )
    parser.add_argument("--fold-ids", default="19")
    parser.add_argument("--num-folds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-subjects-per-fold", type=int, default=2)
    parser.add_argument(
        "--validation-strategy",
        choices=["tail", "quality-extremes", "fixed-subjects"],
        default="tail",
    )
    parser.add_argument(
        "--validation-subject-ids",
        default="",
        help="Comma-separated predeclared subject ids used only with fixed-subjects strategy.",
    )
    parser.add_argument("--validation-quality-eeg-positions", default="0,1")
    parser.add_argument(
        "--excluded-subject-ids",
        default="",
        help="Predeclared inner assessment subjects excluded from fitting and checkpoint selection.",
    )
    parser.add_argument(
        "--channel-indices",
        default="all",
        help="Comma-separated channel indices to keep from NPZ x, e.g. 0,1 for EEG+EOG.",
    )
    parser.add_argument(
        "--channel-fusion",
        choices=[
            "early",
            "quality-gated",
            "quality-prior-gated",
            "quality-hybrid-gated",
            "quality-piecewise-gated",
            "attention-gated",
            "feature-attention-gated",
        ],
        default="early",
        help="Channel fusion strategy before the AttnSleep convolutional encoder.",
    )
    parser.add_argument(
        "--channel-dropout",
        type=float,
        default=0.0,
        help="Per-epoch channel dropout used only by quality-gated fusion.",
    )
    parser.add_argument(
        "--afr-type",
        choices=["se", "eca", "cbam"],
        default="se",
        help="Attention operator used inside the MRCNN AFR residual block.",
    )
    parser.add_argument(
        "--classifier-head",
        choices=["flatten", "attentive_pool"],
        default="flatten",
        help="Classifier head: original flattening or learned attentive temporal pooling.",
    )
    parser.add_argument(
        "--modal-stem",
        action="store_true",
        help="Add a depthwise modality-specific preprocessing stem before shared MRCNN.",
    )
    parser.add_argument(
        "--modality-groups",
        default="",
        help="Semicolon-separated channel groups for modal stem, e.g. '0,1;2;3'.",
    )
    parser.add_argument(
        "--cross-modal-attention",
        action="store_true",
        help="Add one lightweight cross-modal attention layer after grouped modal stems.",
    )
    parser.add_argument(
        "--film-modulation",
        action="store_true",
        help="Apply identity-initialized per-modality FiLM modulation after grouped modal stems.",
    )
    parser.add_argument(
        "--branch-calibration",
        action="store_true",
        help="Add identity-initialized gains and scale embeddings to MRCNN short/long branches.",
    )
    parser.add_argument(
        "--dense-long-branch",
        action="store_true",
        help="Use stride-25 then stride-4 pooling in the long-scale branch while preserving its output length.",
    )
    parser.add_argument(
        "--channelwise-branch-calibration",
        action="store_true",
        help="Use bounded per-channel gains/biases for the calibrated short/long branches.",
    )
    parser.add_argument(
        "--p2-residual",
        action="store_true",
        help="Add a zero-initialized depthwise temporal residual block to the long branch.",
    )
    parser.add_argument(
        "--afr-bypass",
        action="store_true",
        help="Add a zero-initialized 1x1 residual bypass around AFR.",
    )
    parser.add_argument(
        "--post-p3-mode",
        choices=["full", "projection_tce", "direct_pool"],
        default="full",
        help="Post-P3 path: full AFR+TCE, 1x1 projection+TCE, or direct mean/std pooling head.",
    )
    parser.add_argument(
        "--pid-temporal-mode",
        choices=[
            "none", "integral", "derivative", "pid", "matched",
            "dynamic_integral", "dynamic_derivative", "dynamic_pid",
            "dynamic_matched",
        ],
        default="none",
        help="Identity-initialized P/I/D-inspired temporal input refinement.",
    )
    parser.add_argument("--pid-integral-kernel", type=int, default=101)
    parser.add_argument("--pid-derivative-smooth-kernel", type=int, default=5)
    parser.add_argument("--pid-refinement-kernel", type=int, default=9)
    parser.add_argument("--pid-max-residual-scale", type=float, default=0.5)
    parser.add_argument("--pid-integral-kernels", default="51,101,201,401")
    parser.add_argument("--pid-derivative-smooth-kernels", default="3,5,11,21")
    parser.add_argument("--pid-dynamic-channel-embedding-dim", type=int, default=4)
    parser.add_argument("--pid-dynamic-gate-hidden-size", type=int, default=16)
    parser.add_argument(
        "--entropy-feature-set",
        choices=["none", "apen", "sampen", "apen_sampen", "basic"],
        default="none",
        help="Cached epoch-level ApEn/SampEn or parameter-matched RMS statistics residual.",
    )
    parser.add_argument(
        "--entropy-cache-dir",
        type=Path,
        default=None,
        help="Directory containing one cached feature NPZ per source record.",
    )
    parser.add_argument(
        "--branch-orthogonality-weight",
        type=float,
        default=0.0,
        help="Training-only decorrelation penalty between pooled short/long MRCNN branches.",
    )
    parser.add_argument(
        "--deep-supervision-tap",
        choices=["none", "P2_features2", "P4_AFR"],
        default="none",
        help="Attach a lightweight auxiliary classifier to one intermediate probe tap.",
    )
    parser.add_argument(
        "--deep-supervision-weight",
        type=float,
        default=0.0,
        help="Weight of the intermediate probe loss; 0 disables deep supervision.",
    )
    parser.add_argument("--quality-prior-eeg-positions", default="0,1")
    parser.add_argument("--quality-prior-mad-threshold", type=float, default=3.0)
    parser.add_argument("--quality-prior-slope", type=float, default=2.0)
    parser.add_argument("--quality-prior-min-weight", type=float, default=0.1)
    parser.add_argument("--quality-prior-min-scale", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epoch", type=int, default=10)
    parser.add_argument("--lr-after-drop", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--loss-type",
        choices=["cross-entropy", "focal"],
        default="cross-entropy",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--supcon-weight",
        type=float,
        default=0.0,
        help="Weight for supervised contrastive loss on the pre-classifier feature.",
    )
    parser.add_argument(
        "--supcon-temperature",
        type=float,
        default=0.1,
        help="Temperature for supervised contrastive loss.",
    )
    parser.add_argument(
        "--supcon-projection-dim",
        type=int,
        default=128,
        help="Projection dimension for supervised contrastive loss.",
    )
    parser.add_argument(
        "--supcon-projection-hidden",
        type=int,
        default=256,
        help="Hidden dimension for the supervised contrastive projection head.",
    )
    parser.add_argument(
        "--sampler",
        choices=[
            "none",
            "class-balanced",
            "transition-boundary",
            "n1-transition-boundary",
        ],
        default="none",
        help="Optional train DataLoader sampler. Defaults to the original shuffled training.",
    )
    parser.add_argument(
        "--sampler-replacement",
        action="store_true",
        help="Sample with replacement when using a weighted train sampler.",
    )
    parser.add_argument(
        "--class-balance-power",
        type=float,
        default=1.0,
        help=(
            "Power applied to inverse class counts for --sampler=class-balanced. "
            "1.0 is full inverse-frequency balancing; 0.5 is a milder sqrt inverse."
        ),
    )
    parser.add_argument(
        "--transition-window",
        type=int,
        default=2,
        help="Epoch radius around within-record label transitions for --sampler=transition-boundary.",
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=3.0,
        help="Sampling weight assigned to epochs near within-record label transitions.",
    )
    parser.add_argument(
        "--n1-loss-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplier applied to the N1 class weight. Defaults to no change.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument(
        "--augment-noise-std",
        type=float,
        default=0.0,
        help="Train-only Gaussian noise std in each channel's batch signal units.",
    )
    parser.add_argument(
        "--augment-noise-positions",
        default="all",
        help="Selected input-channel positions receiving train-only noise; use all, none, or e.g. 0,1.",
    )
    parser.add_argument(
        "--augment-scale-std",
        type=float,
        default=0.0,
        help="Train-only per-channel multiplicative scale std, e.g. 0.05.",
    )
    parser.add_argument(
        "--augment-scale-positions",
        default="all",
        help="Selected input-channel positions receiving train-only scaling; use all, none, or e.g. 3.",
    )
    parser.add_argument(
        "--augment-time-shift",
        type=int,
        default=0,
        help="Train-only random circular shift range in samples.",
    )
    parser.add_argument(
        "--augment-time-shift-positions",
        default="all",
        help="Selected input-channel positions receiving train-only time shift; use all, none, or e.g. 0,1,2.",
    )
    parser.add_argument(
        "--augment-time-mask-samples",
        type=int,
        default=0,
        help="Train-only contiguous zero-mask length in samples; 0 disables masking.",
    )
    parser.add_argument(
        "--augment-time-mask-prob",
        type=float,
        default=0.0,
        help="Probability of applying one train-only contiguous time mask per sample.",
    )
    parser.add_argument(
        "--augment-time-mask-positions",
        default="all",
        help="Selected input-channel positions receiving train-only time masking; use all, none, or e.g. 0,1,2.",
    )
    parser.add_argument(
        "--sliding-window-delta-seconds",
        type=float,
        default=0.0,
        help=(
            "Train-only overlapping windows within continuous same-label runs. "
            "0 disables this augmentation; validation/test always use original epochs."
        ),
    )
    parser.add_argument(
        "--sliding-window-max-extra-per-run",
        type=int,
        default=0,
        help=(
            "Maximum non-original windows added per same-label run; 0 keeps all "
            "eligible windows."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--init-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional existing fold run directory used to warm-start each fold "
            "from fold_XX/checkpoint_best.pt. New PID parameters remain at their "
            "identity initialization."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional single checkpoint used to warm-start every requested fold. "
            "This is intended for external-dataset pretraining weights that are "
            "independent of all Sleep-EDF folds."
        ),
    )
    parser.add_argument(
        "--init-exclude-prefixes",
        default="",
        help=(
            "Comma-separated state-dict prefixes to leave randomly initialized "
            "when loading --init-checkpoint, e.g. channel_gate,mrcnn.features1.0."
        ),
    )
    parser.add_argument("--init-checkpoint-name", default="checkpoint_best.pt")
    parser.add_argument(
        "--save-selection-trajectory",
        action="store_true",
        help="Save each epoch's weights and validation stage metrics for paired checkpoint selection.",
    )
    parser.add_argument(
        "--defer-outer-test",
        action="store_true",
        help="Do not load or score outer-test records; write a validation-only training summary.",
    )
    parser.add_argument("--resume", default="auto", choices=["auto", "never"])
    parser.add_argument(
        "--seed-scope",
        choices=["run", "fold"],
        default="run",
        help="Use 'fold' to reset RNG state at the start of every fold.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Reuse existing fold summary files and aggregate them instead of retraining.",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Train requested folds without writing shared aggregate files.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=["val-accuracy", "val-accuracy-macro-f1"],
        default="val-accuracy",
        help="Metric used to select checkpoint_best.pt on the internal validation set.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def import_attnsleep(attnsleep_dir: Path) -> None:
    sys.path.insert(0, str(attnsleep_dir.resolve()))


def parse_fold_ids(value: str, num_folds: int) -> list[int]:
    if value == "all":
        return list(range(num_folds))
    fold_ids = [int(v.strip()) for v in value.split(",") if v.strip()]
    invalid = [v for v in fold_ids if v < 0 or v >= num_folds]
    if invalid:
        raise ValueError(f"Invalid fold ids: {invalid}")
    return fold_ids


def parse_channel_indices(value: str) -> list[int] | None:
    if value == "all":
        return None
    indices = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not indices:
        raise ValueError("--channel-indices must be 'all' or a comma-separated list")
    if any(i < 0 for i in indices):
        raise ValueError(f"Invalid negative channel index in {indices}")
    return indices


def parse_modality_groups(value: str, input_channels: int) -> list[tuple[int, ...]] | None:
    if not value:
        return None
    groups = []
    for raw in value.split(";"):
        group = tuple(int(v.strip()) for v in raw.split(",") if v.strip())
        if not group:
            raise ValueError("--modality-groups contains an empty group")
        groups.append(group)
    flat = [i for group in groups for i in group]
    if sorted(flat) != list(range(input_channels)):
        raise ValueError(
            f"--modality-groups must partition 0..{input_channels - 1}, got {groups}"
        )
    return groups


def parse_odd_kernel_list(value: str, name: str) -> tuple[int, ...]:
    kernels = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not kernels or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError(f"{name} must contain positive odd integers, got {value}")
    return kernels


def parse_augment_positions(value: str, n_channels: int, name: str) -> list[int]:
    if value == "all":
        return list(range(n_channels))
    if value in {"", "none"}:
        return []
    positions = [int(v.strip()) for v in value.split(",") if v.strip()]
    invalid = [v for v in positions if v < 0 or v >= n_channels]
    if invalid:
        raise ValueError(
            f"{name} contains positions outside selected input channels: "
            f"{invalid}, n_channels={n_channels}"
        )
    return sorted(set(positions))


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def set_random_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays(
    files: list[str],
    channel_indices: list[int] | None,
    quality_stats: RecordQualityStats | None = None,
    quality_mad_threshold: float = 3.0,
    quality_slope: float = 2.0,
    quality_min_weight: float = 0.1,
    quality_prior_mode: str = "soft",
    sliding_window_delta_seconds: float = 0.0,
    sliding_window_max_extra_per_run: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xs = []
    ys = []
    quality_priors = []
    record_ids = []
    channels: list[str] = []
    for record_id, file_path in enumerate(sorted(files)):
        with np.load(file_path, allow_pickle=True) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int64)
            if not channels and "ch_labels" in data.files:
                all_channels = [str(v) for v in data["ch_labels"].tolist()]
                if channel_indices is None:
                    channels = all_channels
                else:
                    if max(channel_indices) >= len(all_channels):
                        raise IndexError(
                            f"Channel index out of range for {file_path}: "
                            f"indices={channel_indices}, channels={all_channels}"
                        )
                    channels = [all_channels[i] for i in channel_indices]
        if x.ndim == 2:
            x = x[:, np.newaxis, :]
        if channel_indices is not None:
            x = x[:, channel_indices, :]
        if x.ndim != 3:
            raise ValueError(f"Expected x with 2 or 3 dims in {file_path}, got {x.shape}")
        x, y = expand_same_label_windows(
            x,
            y,
            sliding_window_delta_seconds,
            sliding_window_max_extra_per_run,
        )
        if quality_stats is None:
            prior = np.ones(x.shape[1], dtype=np.float32)
        else:
            if quality_prior_mode == "soft":
                prior_fn = soft_channel_prior
            elif quality_prior_mode == "piecewise":
                prior_fn = piecewise_channel_prior
            else:
                raise ValueError(
                    f"Unsupported quality_prior_mode: {quality_prior_mode}"
                )
            prior, _, _ = prior_fn(
                x,
                quality_stats,
                mad_threshold=quality_mad_threshold,
                slope=quality_slope,
                min_weight=quality_min_weight,
            )
        xs.append(x)
        ys.append(y)
        quality_priors.append(np.repeat(prior[np.newaxis, :], len(y), axis=0))
        record_ids.append(np.full(len(y), record_id, dtype=np.int64))
    if not xs:
        raise ValueError("No files loaded")
    return (
        np.vstack(xs),
        np.concatenate(ys),
        np.vstack(quality_priors),
        np.concatenate(record_ids),
        channels,
    )


def expand_same_label_windows(
    x: np.ndarray,
    y: np.ndarray,
    delta_seconds: float = 0.0,
    max_extra_per_run: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Add overlapping windows only inside continuous same-label runs."""
    if delta_seconds <= 0.0:
        return x, y
    if max_extra_per_run < 0:
        raise ValueError("sliding_window_max_extra_per_run must be non-negative")
    if x.ndim != 3 or len(x) != len(y):
        raise ValueError(
            f"Expected x=(epochs, channels, samples) and aligned y, got {x.shape}, {y.shape}"
        )
    epochs, _channels, samples_per_epoch = x.shape
    sample_rate = samples_per_epoch / 30.0
    delta_samples = int(round(float(delta_seconds) * sample_rate))
    if delta_samples <= 0:
        raise ValueError(
            f"sliding-window-delta-seconds={delta_seconds} is below one sample at {sample_rate:g} Hz"
        )
    if delta_samples >= samples_per_epoch:
        raise ValueError("sliding-window-delta-seconds must be smaller than 30 seconds")

    entries: list[tuple[int, int]] = []
    start_epoch = 0
    while start_epoch < epochs:
        end_epoch = start_epoch + 1
        label = int(y[start_epoch])
        while end_epoch < epochs and int(y[end_epoch]) == label and label >= 0:
            end_epoch += 1
        if label < 0:
            entries.append((start_epoch * samples_per_epoch, label))
            start_epoch += 1
            continue
        run_start = start_epoch * samples_per_epoch
        run_samples = (end_epoch - start_epoch) * samples_per_epoch
        original_starts = [run_start + offset * samples_per_epoch for offset in range(end_epoch - start_epoch)]
        candidate_starts = list(
            range(run_start, run_start + run_samples - samples_per_epoch + 1, delta_samples)
        )
        original_set = set(original_starts)
        extras = [start for start in candidate_starts if start not in original_set]
        if max_extra_per_run and len(extras) > max_extra_per_run:
            selected = np.linspace(0, len(extras) - 1, num=max_extra_per_run, dtype=int)
            extras = [extras[int(index)] for index in selected]
        entries.extend((start, label) for start in original_starts)
        entries.extend((start, label) for start in extras)
        start_epoch = end_epoch

    entries.sort(key=lambda item: item[0])
    signal = np.concatenate([x[index] for index in range(epochs)], axis=-1)
    windows = np.stack(
        [signal[:, start : start + samples_per_epoch] for start, _label in entries],
        axis=0,
    )
    labels = np.asarray([label for _start, label in entries], dtype=y.dtype)
    return windows, labels


def cached_feature_columns(feature_set: str, channel_indices: list[int] | None) -> list[int]:
    channels = list(range(4)) if channel_indices is None else channel_indices
    if any(channel >= 4 for channel in channels):
        raise ValueError("Cached entropy features were prepared for the four Sleep-EDF channels")
    blocks = {
        "apen": (0,),
        "sampen": (4,),
        "apen_sampen": (0, 4),
        "basic": (8, 12),
    }
    if feature_set not in blocks:
        raise ValueError(f"Unsupported cached feature set: {feature_set}")
    return [offset + channel for offset in blocks[feature_set] for channel in channels]


def read_cached_record_features(
    file_path: str | Path,
    cache_dir: Path,
    feature_set: str,
    channel_indices: list[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    file_path = Path(file_path)
    cache_path = cache_dir / f"{file_path.stem}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached feature file: {cache_path}")
    stat = file_path.stat()
    with np.load(file_path, allow_pickle=True) as source:
        source_labels = np.asarray(source["y"], dtype=np.int64)
    with np.load(cache_path, allow_pickle=False) as cached:
        labels = np.asarray(cached["labels"], dtype=np.int64)
        matrix = np.asarray(cached["features"], dtype=np.float32)
        source_size = int(np.asarray(cached["source_size"]).item())
        source_mtime_ns = int(np.asarray(cached["source_mtime_ns"]).item())
        target_hz = int(np.asarray(cached["target_hz"]).item())
        embedding_dim = int(np.asarray(cached["embedding_dim"]).item())
        r_fraction = float(np.asarray(cached["r_fraction"]).item())
    if source_size != stat.st_size or source_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError(
            f"Cached features are stale for {file_path.name}; regenerate {cache_path}"
        )
    if target_hz != 25 or embedding_dim != 2 or not np.isclose(r_fraction, 0.2):
        raise RuntimeError(
            f"Entropy cache parameters differ from the fixed screening protocol: {cache_path}"
        )
    if labels.shape != source_labels.shape or not np.array_equal(labels, source_labels):
        raise RuntimeError(
            f"Cached feature labels/order do not match source epochs in {file_path.name}"
        )
    if matrix.ndim != 2 or matrix.shape != (len(source_labels), 16):
        raise RuntimeError(
            f"Expected {file_path.name} cache shape ({len(source_labels)}, 16), "
            f"got {matrix.shape}"
        )
    columns = cached_feature_columns(feature_set, channel_indices)
    selected = matrix[:, columns]
    if not np.isfinite(selected).all():
        raise RuntimeError(f"Non-finite cached features in {cache_path}")
    return selected, labels


def fit_cached_feature_standardizer(
    training_files: list[str],
    cache_dir: Path,
    feature_set: str,
    channel_indices: list[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    parts = [
        read_cached_record_features(file_path, cache_dir, feature_set, channel_indices)[0]
        for file_path in sorted(training_files)
    ]
    if not parts:
        raise ValueError("Cannot fit cached feature standardizer without training records")
    features = np.vstack(parts).astype(np.float64, copy=False)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def load_cached_feature_arrays(
    files: list[str],
    cache_dir: Path,
    feature_set: str,
    channel_indices: list[int] | None,
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> np.ndarray:
    if mean is None or std is None:
        raise ValueError("Training-only cached feature mean/std must be provided")
    parts = [
        read_cached_record_features(file_path, cache_dir, feature_set, channel_indices)[0]
        for file_path in sorted(files)
    ]
    features = np.vstack(parts).astype(np.float32, copy=False)
    if features.shape[1] != len(mean) or len(mean) != len(std):
        raise RuntimeError("Cached feature standardizer dimension mismatch")
    return ((features - mean) / std).astype(np.float32, copy=False)


class NumpySleepDataset:
    def __init__(
        self,
        files: list[str],
        channel_indices: list[int] | None,
        quality_stats: RecordQualityStats | None = None,
        quality_mad_threshold: float = 3.0,
        quality_slope: float = 2.0,
        quality_min_weight: float = 0.1,
        quality_prior_mode: str = "soft",
        augment_noise_std: float = 0.0,
        augment_noise_positions: str = "all",
        augment_scale_std: float = 0.0,
        augment_scale_positions: str = "all",
        augment_time_shift: int = 0,
        augment_time_shift_positions: str = "all",
        augment_time_mask_samples: int = 0,
        augment_time_mask_prob: float = 0.0,
        augment_time_mask_positions: str = "all",
        entropy_feature_dir: Path | None = None,
        entropy_feature_set: str = "none",
        entropy_mean: np.ndarray | None = None,
        entropy_std: np.ndarray | None = None,
        sliding_window_delta_seconds: float = 0.0,
        sliding_window_max_extra_per_run: int = 0,
    ):
        import torch
        from torch.utils.data import Dataset

        x, y, quality_prior, record_ids, channels = load_arrays(
            files,
            channel_indices,
            quality_stats,
            quality_mad_threshold,
            quality_slope,
            quality_min_weight,
            quality_prior_mode,
            sliding_window_delta_seconds,
            sliding_window_max_extra_per_run,
        )
        original_labels = np.concatenate(
            [
                np.asarray(np.load(file_path, allow_pickle=True)["y"], dtype=np.int64)
                for file_path in sorted(files)
            ]
        ) if files else np.empty(0, dtype=np.int64)
        if sliding_window_delta_seconds > 0.0 and entropy_feature_set != "none":
            raise ValueError(
                "Sliding-window augmentation is incompatible with epoch-level cached entropy features"
            )
        entropy_features = None
        if entropy_feature_set != "none":
            if entropy_feature_dir is None:
                raise ValueError("--entropy-cache-dir is required for entropy features")
            entropy_features = load_cached_feature_arrays(
                files,
                entropy_feature_dir,
                entropy_feature_set,
                channel_indices,
                entropy_mean,
                entropy_std,
            )
            if len(entropy_features) != len(y):
                raise RuntimeError(
                    "Cached entropy rows do not align with concatenated epoch labels: "
                    f"{len(entropy_features)} != {len(y)}"
                )

        class DatasetImpl(Dataset):
            def __init__(self):
                self.x_data = torch.from_numpy(x)
                self.y_data = torch.from_numpy(y).long()
                self.quality_prior_data = torch.from_numpy(quality_prior)
                self.entropy_feature_data = (
                    torch.from_numpy(entropy_features)
                    if entropy_features is not None
                    else None
                )
                self.record_id_data = torch.from_numpy(record_ids).long()
                self.channels = channels
                self.sliding_window_original_count = int(len(original_labels))
                self.sliding_window_expanded_count = int(len(y))
                self.sliding_window_original_class_counts = np.bincount(
                    original_labels[original_labels >= 0], minlength=5
                ).astype(int).tolist()
                self.sliding_window_expanded_class_counts = np.bincount(
                    y[y >= 0], minlength=5
                ).astype(int).tolist()
                self.sliding_window_added_class_counts = (
                    np.asarray(self.sliding_window_expanded_class_counts)
                    - np.asarray(self.sliding_window_original_class_counts)
                ).astype(int).tolist()
                self.augment_noise_std = float(augment_noise_std)
                self.augment_noise_positions = parse_augment_positions(
                    augment_noise_positions,
                    int(self.x_data.shape[1]),
                    "--augment-noise-positions",
                )
                self.augment_scale_std = float(augment_scale_std)
                self.augment_scale_positions = parse_augment_positions(
                    augment_scale_positions,
                    int(self.x_data.shape[1]),
                    "--augment-scale-positions",
                )
                self.augment_time_shift = int(augment_time_shift)
                self.augment_time_shift_positions = parse_augment_positions(
                    augment_time_shift_positions,
                    int(self.x_data.shape[1]),
                    "--augment-time-shift-positions",
                )
                self.augment_time_mask_samples = int(augment_time_mask_samples)
                self.augment_time_mask_prob = float(augment_time_mask_prob)
                self.augment_time_mask_positions = parse_augment_positions(
                    augment_time_mask_positions,
                    int(self.x_data.shape[1]),
                    "--augment-time-mask-positions",
                )
                if self.augment_noise_std < 0.0:
                    raise ValueError("--augment-noise-std must be non-negative")
                if self.augment_scale_std < 0.0:
                    raise ValueError("--augment-scale-std must be non-negative")
                if self.augment_time_shift < 0:
                    raise ValueError("--augment-time-shift must be non-negative")
                if self.augment_time_mask_samples < 0:
                    raise ValueError("--augment-time-mask-samples must be non-negative")
                if not 0.0 <= self.augment_time_mask_prob <= 1.0:
                    raise ValueError("--augment-time-mask-prob must be in [0, 1]")

            def __getitem__(self, index):
                x_item = self.x_data[index]
                if (
                    self.augment_noise_std > 0.0
                    or self.augment_scale_std > 0.0
                    or self.augment_time_shift > 0
                ):
                    x_item = x_item.clone()
                    if self.augment_scale_std > 0.0 and self.augment_scale_positions:
                        scale_positions = torch.as_tensor(
                            self.augment_scale_positions,
                            dtype=torch.long,
                            device=x_item.device,
                        )
                        scale = 1.0 + torch.randn(
                            len(self.augment_scale_positions),
                            1,
                            dtype=x_item.dtype,
                            device=x_item.device,
                        ) * self.augment_scale_std
                        x_item[scale_positions] = (
                            x_item[scale_positions] * scale.clamp_min(0.5)
                        )
                    if self.augment_noise_std > 0.0 and self.augment_noise_positions:
                        noise_positions = torch.as_tensor(
                            self.augment_noise_positions,
                            dtype=torch.long,
                            device=x_item.device,
                        )
                        x_item[noise_positions] = (
                            x_item[noise_positions]
                            + torch.randn_like(x_item[noise_positions])
                            * self.augment_noise_std
                        )
                    if self.augment_time_shift > 0 and self.augment_time_shift_positions:
                        shift = int(
                            torch.randint(
                                -self.augment_time_shift,
                                self.augment_time_shift + 1,
                                (1,),
                            ).item()
                        )
                        if shift:
                            shift_positions = torch.as_tensor(
                                self.augment_time_shift_positions,
                                dtype=torch.long,
                                device=x_item.device,
                            )
                            x_item[shift_positions] = torch.roll(
                                x_item[shift_positions],
                                shifts=shift,
                                dims=-1,
                            )
                    if (
                        self.augment_time_mask_samples > 0
                        and self.augment_time_mask_prob > 0.0
                        and self.augment_time_mask_positions
                        and torch.rand(()) < self.augment_time_mask_prob
                    ):
                        mask_len = min(self.augment_time_mask_samples, x_item.shape[-1])
                        max_start = x_item.shape[-1] - mask_len
                        start = int(torch.randint(max_start + 1, (1,)).item())
                        mask_positions = torch.as_tensor(
                            self.augment_time_mask_positions,
                            dtype=torch.long,
                            device=x_item.device,
                        )
                        x_item[mask_positions, start : start + mask_len] = 0.0
                batch = (
                    x_item,
                    self.y_data[index],
                    self.quality_prior_data[index],
                )
                if self.entropy_feature_data is not None:
                    return (*batch, self.entropy_feature_data[index])
                return batch

            def __len__(self):
                return len(self.y_data)

        self.dataset = DatasetImpl()


def make_loaders(
    training_files: list[str],
    eval_files: list[str],
    batch_size: int,
    channel_indices: list[int] | None,
    quality_stats: RecordQualityStats | None = None,
    quality_mad_threshold: float = 3.0,
    quality_slope: float = 2.0,
    quality_min_weight: float = 0.1,
    quality_prior_mode: str = "soft",
    sampler: str = "none",
    sampler_replacement: bool = False,
    class_balance_power: float = 1.0,
    transition_window: int = 2,
    transition_weight: float = 3.0,
    augment_noise_std: float = 0.0,
    augment_noise_positions: str = "all",
    augment_scale_std: float = 0.0,
    augment_scale_positions: str = "all",
    augment_time_shift: int = 0,
    augment_time_shift_positions: str = "all",
    augment_time_mask_samples: int = 0,
    augment_time_mask_prob: float = 0.0,
    augment_time_mask_positions: str = "all",
    entropy_feature_dir: Path | None = None,
    entropy_feature_set: str = "none",
    entropy_mean: np.ndarray | None = None,
    entropy_std: np.ndarray | None = None,
    sliding_window_delta_seconds: float = 0.0,
    sliding_window_max_extra_per_run: int = 0,
):
    import torch

    train_dataset = NumpySleepDataset(
        training_files,
        channel_indices,
        quality_stats,
        quality_mad_threshold,
        quality_slope,
        quality_min_weight,
        quality_prior_mode,
        augment_noise_std,
        augment_noise_positions,
        augment_scale_std,
        augment_scale_positions,
        augment_time_shift,
        augment_time_shift_positions,
        augment_time_mask_samples,
        augment_time_mask_prob,
        augment_time_mask_positions,
        entropy_feature_dir,
        entropy_feature_set,
        entropy_mean,
        entropy_std,
        sliding_window_delta_seconds,
        sliding_window_max_extra_per_run,
    ).dataset
    eval_dataset = NumpySleepDataset(
        eval_files,
        channel_indices,
        quality_stats,
        quality_mad_threshold,
        quality_slope,
        quality_min_weight,
        quality_prior_mode,
        0.0,
        "none",
        0.0,
        "none",
        0,
        "none",
        0,
        0.0,
        "none",
        entropy_feature_dir,
        entropy_feature_set,
        entropy_mean,
        entropy_std,
        0.0,
        0,
    ).dataset
    y_train = train_dataset.y_data.tolist()
    counts = [y_train.count(i) for i in range(5)]
    train_sampler = None
    train_shuffle = True
    if sampler == "class-balanced":
        if class_balance_power < 0.0:
            raise ValueError("--class-balance-power must be non-negative")
        y_np = train_dataset.y_data.detach().cpu().numpy().astype(np.int64)
        class_counts = np.bincount(y_np, minlength=5).astype(np.float64)
        inv_counts = (1.0 / np.maximum(class_counts, 1.0)) ** float(class_balance_power)
        sample_weights = inv_counts[y_np]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=sampler_replacement,
        )
        train_shuffle = False
    elif sampler in {"transition-boundary", "n1-transition-boundary"}:
        if transition_window < 0:
            raise ValueError("--transition-window must be non-negative")
        if transition_weight <= 0.0:
            raise ValueError("--transition-weight must be positive")
        y_np = train_dataset.y_data.detach().cpu().numpy().astype(np.int64)
        record_np = train_dataset.record_id_data.detach().cpu().numpy().astype(np.int64)
        sample_weights = np.ones(len(y_np), dtype=np.float64)
        transition_mask = (y_np[1:] != y_np[:-1]) & (record_np[1:] == record_np[:-1])
        if sampler == "n1-transition-boundary":
            transition_mask &= (y_np[:-1] == 1) | (y_np[1:] == 1)
        transition_indices = np.flatnonzero(transition_mask)
        boundary_mask = np.zeros(len(y_np), dtype=bool)
        for left_index in transition_indices:
            start = max(0, int(left_index) - transition_window + 1)
            end = min(len(y_np), int(left_index) + transition_window + 1)
            same_record = record_np[start:end] == record_np[int(left_index)]
            boundary_mask[start:end] |= same_record
        sample_weights[boundary_mask] = float(transition_weight)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=sampler_replacement,
        )
        train_shuffle = False
    elif sampler != "none":
        raise ValueError(f"Unsupported sampler: {sampler}")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=False,
        num_workers=0,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    channels = getattr(train_dataset, "channels", [])
    input_channels = int(train_dataset.x_data.shape[1])
    return train_loader, eval_loader, counts, channels, input_channels


def split_train_validation_files(
    train_files: list[str],
    val_subjects_per_fold: int,
    strategy: str = "tail",
    channel_indices: list[int] | None = None,
    eeg_positions: tuple[int, int] = (0, 1),
    validation_subject_ids: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    model_train_files, val_files, val_subjects, _ = (
        split_train_validation_files_with_info(
            train_files,
            val_subjects_per_fold,
            strategy,
            channel_indices,
            eeg_positions,
            validation_subject_ids,
        )
    )
    return model_train_files, val_files, val_subjects


def split_train_validation_files_with_info(
    train_files: list[str],
    val_subjects_per_fold: int,
    strategy: str = "tail",
    channel_indices: list[int] | None = None,
    eeg_positions: tuple[int, int] = (0, 1),
    validation_subject_ids: list[str] | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    return split_train_validation_files_by_strategy(
        train_files,
        val_subjects_per_fold,
        strategy,
        channel_indices,
        eeg_positions,
        validation_subject_ids,
    )


def forward_model(model, x, quality_prior, entropy_features=None):
    extra = (
        {"entropy_features": entropy_features}
        if entropy_features is not None
        else {}
    )
    if getattr(model, "channel_fusion", "early") in (
        "quality-prior-gated",
        "quality-hybrid-gated",
        "quality-piecewise-gated",
        "feature-attention-gated",
    ):
        return model(x, channel_prior=quality_prior, **extra)
    return model(x, **extra)


def unpack_batch(batch):
    if len(batch) == 3:
        x, y, quality_prior = batch
        return x, y, quality_prior, None
    if len(batch) == 4:
        x, y, quality_prior, entropy_features = batch
        return x, y, quality_prior, entropy_features
    raise ValueError(f"Expected a 3- or 4-item batch, got {len(batch)} items")


class FeatureCapture:
    """Capture the tensor fed into AttnSleep.fc without editing external code."""

    def __init__(self):
        self.features = None

    def hook(self, _module, inputs, _output):
        self.features = inputs[0]


class IntermediateCapture:
    """Capture one intermediate tensor for optional deep supervision."""

    def __init__(self):
        self.features = None

    def hook(self, _module, _inputs, output):
        self.features = output


def aggregate_probe_features(features):
    """Make P2/P4 representations compatible with one shared linear probe."""
    import torch

    if features is None or features.ndim != 3:
        raise RuntimeError(
            "Deep-supervision tap must produce a 3-D tensor, got "
            f"{None if features is None else tuple(features.shape)}"
        )
    # TCE-like tensors are (B,T,C), while MRCNN tensors are (B,C,T).
    z = features.transpose(1, 2) if features.shape[1] <= 40 and features.shape[2] >= 60 else features
    return torch.cat((z.mean(dim=-1), z.std(dim=-1, unbiased=False)), dim=1)


def build_projection_head(input_dim: int, args: argparse.Namespace):
    import torch.nn as nn

    if args.supcon_projection_dim <= 0:
        raise ValueError("--supcon-projection-dim must be positive")
    if args.supcon_projection_hidden <= 0:
        raise ValueError("--supcon-projection-hidden must be positive")
    return nn.Sequential(
        nn.Linear(input_dim, int(args.supcon_projection_hidden)),
        nn.ReLU(inplace=True),
        nn.Linear(int(args.supcon_projection_hidden), int(args.supcon_projection_dim)),
    )


def supervised_contrastive_loss(features, labels, temperature: float):
    import torch
    import torch.nn.functional as F

    if temperature <= 0.0:
        raise ValueError("--supcon-temperature must be positive")
    if features.ndim != 2:
        raise ValueError(f"SupCon expects 2D features, got {tuple(features.shape)}")
    labels = labels.reshape(-1)
    if len(labels) != features.shape[0]:
        raise ValueError("SupCon features/labels batch size mismatch")
    if features.shape[0] <= 1:
        return features.sum() * 0.0

    z = F.normalize(features, dim=1)
    logits = torch.matmul(z, z.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    positive_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    exp_logits = torch.exp(logits) * (~self_mask).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return features.sum() * 0.0
    mean_log_prob_pos = (
        (positive_mask.to(log_prob.dtype) * log_prob).sum(dim=1)
        / positive_counts.clamp_min(1).to(log_prob.dtype)
    )
    return -mean_log_prob_pos[valid].mean()


def branch_orthogonality_loss(short_features, long_features):
    """Decorrelate pooled short/long MRCNN representations for training only."""
    import torch

    if short_features is None or long_features is None:
        raise RuntimeError("Both MRCNN branches must be captured")
    if short_features.ndim != 3 or long_features.ndim != 3:
        raise ValueError("MRCNN branch features must have shape (B,C,T)")
    if short_features.shape[:2] != long_features.shape[:2]:
        raise ValueError(
            "MRCNN branch batch/channel dimensions differ: "
            f"{tuple(short_features.shape)} vs {tuple(long_features.shape)}"
        )

    def summarize(features):
        z = torch.cat(
            (features.mean(dim=-1), features.std(dim=-1, unbiased=False)), dim=1
        )
        z = z - z.mean(dim=0, keepdim=True)
        z = z / z.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        return z

    short = summarize(short_features)
    long = summarize(long_features)
    cross = short.transpose(0, 1) @ long / max(1, short.shape[0])
    return cross.square().mean()


def build_criterion(args: argparse.Namespace, class_weights):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if args.label_smoothing < 0.0 or args.label_smoothing >= 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.n1_loss_multiplier <= 0.0:
        raise ValueError("--n1-loss-multiplier must be positive")
    if args.supcon_weight < 0.0:
        raise ValueError("--supcon-weight must be non-negative")
    if args.branch_orthogonality_weight < 0.0:
        raise ValueError("--branch-orthogonality-weight must be non-negative")

    if args.loss_type == "cross-entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

    class FocalLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", class_weights.detach().clone())

        def forward(self, logits, targets):
            ce = F.cross_entropy(
                logits,
                targets,
                weight=self.weight,
                reduction="none",
                label_smoothing=args.label_smoothing,
            )
            log_probs = F.log_softmax(logits, dim=1)
            target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = target_log_probs.exp().clamp_min(1e-6)
            return (((1.0 - pt) ** args.focal_gamma) * ce).mean()

    return FocalLoss()


def checkpoint_key(args: argparse.Namespace, val_eval: dict[str, Any]) -> tuple[float, ...] | None:
    if val_eval["accuracy"] is None:
        return None
    accuracy = float(val_eval["accuracy"])
    macro_f1 = float(val_eval["macro_f1"] or 0.0)
    if args.checkpoint_selection == "val-accuracy":
        return (accuracy,)
    if args.checkpoint_selection == "val-accuracy-macro-f1":
        return (accuracy, macro_f1)
    raise ValueError(f"Unsupported checkpoint selection: {args.checkpoint_selection}")


def evaluate(model, loader, criterion, device, max_batches: int = 0) -> dict[str, Any]:
    import torch
    from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score

    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    logits_parts: list[np.ndarray] = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if max_batches and step > max_batches:
                break
            x, y, quality_prior, entropy_features = unpack_batch(batch)
            x = x.float().to(device)
            y = y.to(device)
            quality_prior = quality_prior.float().to(device)
            if entropy_features is not None:
                entropy_features = entropy_features.float().to(device)
            out = forward_model(model, x, quality_prior, entropy_features)
            loss = criterion(out, y)
            losses.append(float(loss.detach().cpu()))
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(out.argmax(1).detach().cpu().tolist())
            logits_parts.append(out.detach().cpu().numpy().astype(np.float32))
    if not y_true:
        return {
            "loss": None,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "kappa": None,
            "classification_report": {},
            "y_true": [],
            "y_pred": [],
            "logits": np.empty((0, 5), dtype=np.float32),
        }
    logits = np.concatenate(logits_parts, axis=0) if logits_parts else np.empty((0, 5), dtype=np.float32)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=["W", "N1", "N2", "N3", "REM"],
            output_dict=True,
            zero_division=0,
        ),
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits,
    }


def write_history_row(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ce_loss",
        "train_supcon_loss",
        "train_auxiliary_loss",
        "train_branch_orthogonality_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "val_kappa",
        "lr",
        "elapsed_sec",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fieldnames})


def train_fold(args: argparse.Namespace, fold_id: int, device) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score

    from model.model import AttnSleep
    from utils.util import calc_class_weight, load_folds_data

    fold_seed = int(args.seed) + int(fold_id)
    if args.seed_scope == "fold":
        set_random_seed(fold_seed)

    fold_dir = args.output_dir / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    folds = load_folds_data(str(args.np_data_dir.resolve()), args.num_folds)
    train_files, test_files = folds[fold_id]
    train_sids = sorted({os.path.basename(f)[3:5] for f in train_files})
    test_sids = sorted({os.path.basename(f)[3:5] for f in test_files})
    subject_overlap = sorted(set(train_sids) & set(test_sids))
    if subject_overlap:
        raise RuntimeError(f"Subject leakage detected in fold {fold_id}: {subject_overlap}")

    excluded_subjects = sorted({sid.strip().zfill(2) for sid in args.excluded_subject_ids.split(",") if sid.strip()})
    if not set(excluded_subjects) <= set(train_sids):
        raise ValueError("Excluded assessment subjects must belong to the outer training fold")
    train_files = [path for path in train_files if os.path.basename(path)[3:5] not in excluded_subjects]

    channel_indices = parse_channel_indices(args.channel_indices)
    validation_eeg_positions = parse_index_pair(args.validation_quality_eeg_positions)
    model_train_files, val_files, val_sids, validation_split_info = (
        split_train_validation_files_with_info(
            train_files,
            args.val_subjects_per_fold,
            args.validation_strategy,
            channel_indices,
            validation_eeg_positions,
            [sid.strip() for sid in args.validation_subject_ids.split(",") if sid.strip()],
        )
    )
    if args.defer_outer_test and not val_files:
        raise ValueError("--defer-outer-test requires subject-disjoint validation files")
    quality_stats = None
    quality_eeg_positions = None
    quality_prior_mode = (
        "piecewise"
        if args.channel_fusion == "quality-piecewise-gated"
        else "soft"
    )
    if args.quality_prior_min_scale < 0.0:
        raise ValueError("quality-prior-min-scale must be non-negative")
    if (
        args.channel_fusion
        in (
            "quality-hybrid-gated",
            "quality-piecewise-gated",
            "feature-attention-gated",
        )
        and args.quality_prior_min_scale <= 0.0
    ):
        raise ValueError(
            "hybrid, piecewise, and feature-attention quality fusion require a positive "
            "quality-prior-min-scale"
        )
    if args.channel_fusion in (
        "quality-prior-gated",
        "quality-hybrid-gated",
        "quality-piecewise-gated",
        "feature-attention-gated",
    ):
        quality_eeg_positions = parse_index_pair(args.quality_prior_eeg_positions)
        quality_stats = fit_record_quality_stats(
            model_train_files,
            channel_indices,
            quality_eeg_positions,
        )
    if val_files:
        entropy_feature_mean = None
        entropy_feature_std = None
        if args.entropy_feature_set != "none":
            if args.entropy_cache_dir is None:
                raise ValueError("--entropy-cache-dir is required with --entropy-feature-set")
            if (
                args.augment_noise_std > 0.0
                or args.augment_scale_std > 0.0
                or args.augment_time_shift > 0
                or (args.augment_time_mask_samples > 0 and args.augment_time_mask_prob > 0.0)
            ):
                raise ValueError(
                    "Cached entropy experiments must disable waveform augmentation so "
                    "features remain aligned with the waveform"
                )
            entropy_feature_mean, entropy_feature_std = fit_cached_feature_standardizer(
                model_train_files,
                args.entropy_cache_dir,
                args.entropy_feature_set,
                channel_indices,
            )
        entropy_feature_dim = (
            len(cached_feature_columns(args.entropy_feature_set, channel_indices))
            if args.entropy_feature_set != "none"
            else 0
        )
        train_loader, val_loader, counts, channels, input_channels = make_loaders(
            model_train_files,
            val_files,
            args.batch_size,
            channel_indices,
            quality_stats,
            args.quality_prior_mad_threshold,
            args.quality_prior_slope,
            args.quality_prior_min_weight,
            quality_prior_mode,
            args.sampler,
            args.sampler_replacement,
            args.class_balance_power,
            args.transition_window,
            args.transition_weight,
            args.augment_noise_std,
            args.augment_noise_positions,
            args.augment_scale_std,
            args.augment_scale_positions,
            args.augment_time_shift,
            args.augment_time_shift_positions,
            args.augment_time_mask_samples,
            args.augment_time_mask_prob,
            args.augment_time_mask_positions,
            entropy_feature_dir=args.entropy_cache_dir,
            entropy_feature_set=args.entropy_feature_set,
            entropy_mean=entropy_feature_mean,
            entropy_std=entropy_feature_std,
            sliding_window_delta_seconds=args.sliding_window_delta_seconds,
            sliding_window_max_extra_per_run=args.sliding_window_max_extra_per_run,
        )
        test_loader = None
        if not args.defer_outer_test:
            _, test_loader, _, _, _ = make_loaders(
                model_train_files,
                test_files,
                args.batch_size,
                channel_indices,
                quality_stats,
                args.quality_prior_mad_threshold,
                args.quality_prior_slope,
                args.quality_prior_min_weight,
                quality_prior_mode,
                "none",
                False,
                1.0,
                args.transition_window,
                args.transition_weight,
                entropy_feature_dir=args.entropy_cache_dir,
                entropy_feature_set=args.entropy_feature_set,
                entropy_mean=entropy_feature_mean,
                entropy_std=entropy_feature_std,
                sliding_window_delta_seconds=0.0,
                sliding_window_max_extra_per_run=0,
            )
    else:
        entropy_feature_mean = None
        entropy_feature_std = None
        if args.entropy_feature_set != "none":
            if args.entropy_cache_dir is None:
                raise ValueError("--entropy-cache-dir is required with --entropy-feature-set")
            if (
                args.augment_noise_std > 0.0
                or args.augment_scale_std > 0.0
                or args.augment_time_shift > 0
                or (args.augment_time_mask_samples > 0 and args.augment_time_mask_prob > 0.0)
            ):
                raise ValueError(
                    "Cached entropy experiments must disable waveform augmentation so "
                    "features remain aligned with the waveform"
                )
            entropy_feature_mean, entropy_feature_std = fit_cached_feature_standardizer(
                model_train_files,
                args.entropy_cache_dir,
                args.entropy_feature_set,
                channel_indices,
            )
        entropy_feature_dim = (
            len(cached_feature_columns(args.entropy_feature_set, channel_indices))
            if args.entropy_feature_set != "none"
            else 0
        )
        train_loader, test_loader, counts, channels, input_channels = make_loaders(
            model_train_files,
            test_files,
            args.batch_size,
            channel_indices,
            quality_stats,
            args.quality_prior_mad_threshold,
            args.quality_prior_slope,
            args.quality_prior_min_weight,
            quality_prior_mode,
            args.sampler,
            args.sampler_replacement,
            args.class_balance_power,
            args.transition_window,
            args.transition_weight,
            args.augment_noise_std,
            args.augment_noise_positions,
            args.augment_scale_std,
            args.augment_scale_positions,
            args.augment_time_shift,
            args.augment_time_shift_positions,
            args.augment_time_mask_samples,
            args.augment_time_mask_prob,
            args.augment_time_mask_positions,
            entropy_feature_dir=args.entropy_cache_dir,
            entropy_feature_set=args.entropy_feature_set,
            entropy_mean=entropy_feature_mean,
            entropy_std=entropy_feature_std,
            sliding_window_delta_seconds=args.sliding_window_delta_seconds,
            sliding_window_max_extra_per_run=args.sliding_window_max_extra_per_run,
        )
        val_loader = test_loader

    weights = torch.tensor(calc_class_weight(counts), dtype=torch.float32, device=device)
    if args.n1_loss_multiplier != 1.0:
        weights[1] = weights[1] * float(args.n1_loss_multiplier)
    modality_groups = parse_modality_groups(args.modality_groups, input_channels)
    pid_integral_kernels = parse_odd_kernel_list(
        args.pid_integral_kernels, "--pid-integral-kernels"
    )
    pid_derivative_smooth_kernels = parse_odd_kernel_list(
        args.pid_derivative_smooth_kernels,
        "--pid-derivative-smooth-kernels",
    )
    model = AttnSleep(
        input_channels=input_channels,
        channel_fusion=args.channel_fusion,
        channel_dropout=args.channel_dropout,
        record_prior_min_scale=args.quality_prior_min_scale,
        afr_type=args.afr_type,
        classifier_head=args.classifier_head,
        modal_stem=args.modal_stem,
        modality_groups=modality_groups,
        cross_modal_attention=args.cross_modal_attention,
        film_modulation=args.film_modulation,
        branch_calibration=args.branch_calibration,
        dense_long_branch=args.dense_long_branch,
        channelwise_branch_calibration=args.channelwise_branch_calibration,
        p2_residual=args.p2_residual,
        afr_bypass=args.afr_bypass,
        post_p3_mode=args.post_p3_mode,
        pid_temporal_mode=args.pid_temporal_mode,
        pid_integral_kernel=args.pid_integral_kernel,
        pid_derivative_smooth_kernel=args.pid_derivative_smooth_kernel,
        pid_refinement_kernel=args.pid_refinement_kernel,
        pid_max_residual_scale=args.pid_max_residual_scale,
        pid_integral_kernels=pid_integral_kernels,
        pid_derivative_smooth_kernels=pid_derivative_smooth_kernels,
        pid_dynamic_channel_embedding_dim=args.pid_dynamic_channel_embedding_dim,
        pid_dynamic_gate_hidden_size=args.pid_dynamic_gate_hidden_size,
        entropy_feature_dim=entropy_feature_dim,
    ).to(device)
    initialization_info = None
    if args.init_run_dir is not None and args.init_checkpoint is not None:
        raise ValueError("Use only one of --init-run-dir and --init-checkpoint")
    if args.init_run_dir is not None or args.init_checkpoint is not None:
        init_path = (
            args.init_checkpoint
            if args.init_checkpoint is not None
            else args.init_run_dir / f"fold_{fold_id:02d}" / args.init_checkpoint_name
        )
        if not init_path.exists():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {init_path}")
        init_checkpoint = torch.load(
            init_path, map_location=device, weights_only=False
        )
        exclude_prefixes = tuple(
            prefix.strip()
            for prefix in str(args.init_exclude_prefixes).split(",")
            if prefix.strip()
        )
        initial_state = init_checkpoint["model_state"]
        if exclude_prefixes:
            initial_state = {
                key: value
                for key, value in initial_state.items()
                if not key.startswith(exclude_prefixes)
            }
        incompatible = model.load_state_dict(initial_state, strict=False)
        allowed_missing_prefixes = (
            "mrcnn.pid_temporal_stem.",
            "entropy_residual_head.",
        ) + exclude_prefixes
        unexpected_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Warm-start state mismatch: missing={}, unexpected={}".format(
                    unexpected_missing, incompatible.unexpected_keys
                )
            )
        initialization_info = {
            "checkpoint": str(init_path),
            "source_epoch": init_checkpoint.get("epoch"),
            "excluded_prefixes": list(exclude_prefixes),
            "missing_identity_initialized_keys": incompatible.missing_keys,
        }
    if args.deep_supervision_weight < 0.0:
        raise ValueError("--deep-supervision-weight must be non-negative")
    if args.deep_supervision_weight > 0.0 and args.deep_supervision_tap == "none":
        raise ValueError("--deep-supervision-tap is required when auxiliary loss is enabled")
    auxiliary_head = None
    auxiliary_capture = None
    auxiliary_hook_handle = None
    if args.deep_supervision_tap != "none":
        tap_module = {
            "P2_features2": model.mrcnn.features2,
            "P4_AFR": model.mrcnn.AFR,
        }[args.deep_supervision_tap]
        auxiliary_capture = IntermediateCapture()
        auxiliary_hook_handle = tap_module.register_forward_hook(auxiliary_capture.hook)
        # P2 is (B,128,T)->256 summary features; P4 is (B,T,80)->160.
        auxiliary_input_dim = 256 if args.deep_supervision_tap == "P2_features2" else 160
        auxiliary_head = nn.Sequential(
            nn.LayerNorm(auxiliary_input_dim),
            nn.Linear(auxiliary_input_dim, 5),
        ).to(device)
    projection_head = None
    feature_capture = None
    feature_hook_handle = None
    if args.supcon_weight > 0.0:
        projection_head = build_projection_head(int(model.fc.in_features), args).to(device)
        feature_capture = FeatureCapture()
        feature_hook_handle = model.fc.register_forward_hook(feature_capture.hook)
    orthogonality_capture_short = None
    orthogonality_capture_long = None
    orthogonality_hook_handles = []
    if args.branch_orthogonality_weight > 0.0:
        if model.mrcnn is None:
            raise ValueError("Branch orthogonality requires the standard MRCNN path")
        orthogonality_capture_short = IntermediateCapture()
        orthogonality_capture_long = IntermediateCapture()
        orthogonality_hook_handles = [
            model.mrcnn.features1.register_forward_hook(
                orthogonality_capture_short.hook
            ),
            model.mrcnn.features2.register_forward_hook(
                orthogonality_capture_long.hook
            ),
        ]
    criterion = build_criterion(args, weights)
    optimizer_params = list(model.parameters())
    if auxiliary_head is not None:
        optimizer_params += list(auxiliary_head.parameters())
    if projection_head is not None:
        optimizer_params += list(projection_head.parameters())
    optimizer = torch.optim.Adam(
        optimizer_params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True
    )

    baseline_names = {
        "early": "AttnSleep-multichannel",
        "quality-gated": "AttnSleep-multichannel-quality-gated",
        "quality-prior-gated": "AttnSleep-multichannel-quality-prior-gated",
        "quality-hybrid-gated": "AttnSleep-multichannel-quality-hybrid-gated",
        "quality-piecewise-gated": "AttnSleep-multichannel-quality-piecewise-gated",
        "attention-gated": "AttnSleep-multichannel-attention-gated",
        "feature-attention-gated": "AttnSleep-multichannel-feature-attention-gated",
    }
    metadata = {
        "baseline": baseline_names[args.channel_fusion],
        "dataset": f"Sleep-EDF-20, {' + '.join(channels)}",
        "fold_id": fold_id,
        "train_subjects": train_sids,
        "model_train_subjects": sorted({os.path.basename(path)[3:5] for path in model_train_files}),
        "checkpoint_selection_subjects": val_sids,
        "excluded_assessment_subjects": excluded_subjects,
        "validation_strategy": args.validation_strategy,
        "validation_quality_eeg_positions": list(validation_eeg_positions),
        "validation_split_info": validation_split_info,
        "test_subjects": test_sids,
        "subject_overlap": subject_overlap,
        "train_files": len(model_train_files),
        "validation_files": len(val_files),
        "test_files": len(test_files),
        "channels": channels,
        "channel_indices": channel_indices if channel_indices is not None else "all",
        "input_channels": input_channels,
        "channel_fusion": args.channel_fusion,
        "channel_dropout": args.channel_dropout,
        "afr_type": args.afr_type,
        "classifier_head": args.classifier_head,
        "modal_stem": args.modal_stem,
        "modality_groups": [list(group) for group in modality_groups] if modality_groups else None,
        "cross_modal_attention": args.cross_modal_attention,
        "film_modulation": args.film_modulation,
        "branch_calibration": args.branch_calibration,
        "dense_long_branch": args.dense_long_branch,
        "channelwise_branch_calibration": args.channelwise_branch_calibration,
        "p2_residual": args.p2_residual,
        "afr_bypass": args.afr_bypass,
        "post_p3_mode": args.post_p3_mode,
        "pid_temporal_mode": args.pid_temporal_mode,
        "pid_integral_kernel": args.pid_integral_kernel,
        "pid_derivative_smooth_kernel": args.pid_derivative_smooth_kernel,
        "pid_refinement_kernel": args.pid_refinement_kernel,
        "pid_max_residual_scale": args.pid_max_residual_scale,
        "pid_integral_kernels": list(pid_integral_kernels),
        "pid_derivative_smooth_kernels": list(pid_derivative_smooth_kernels),
        "pid_dynamic_channel_embedding_dim": args.pid_dynamic_channel_embedding_dim,
        "pid_dynamic_gate_hidden_size": args.pid_dynamic_gate_hidden_size,
        "entropy_feature_set": args.entropy_feature_set,
        "entropy_cache_dir": (
            str(args.entropy_cache_dir) if args.entropy_cache_dir is not None else None
        ),
        "entropy_feature_dim": entropy_feature_dim,
        "entropy_feature_parameters": (
            {"m": 2, "r_fraction_of_epoch_sd": 0.2, "downsample_hz": 25}
            if args.entropy_feature_set != "none"
            else None
        ),
        "entropy_feature_mean_train_only": (
            entropy_feature_mean.tolist() if entropy_feature_mean is not None else None
        ),
        "entropy_feature_std_train_only": (
            entropy_feature_std.tolist() if entropy_feature_std is not None else None
        ),
        "initialization": initialization_info,
        "deep_supervision_tap": args.deep_supervision_tap,
        "deep_supervision_weight": args.deep_supervision_weight,
        "branch_orthogonality_weight": args.branch_orthogonality_weight,
        "record_quality_stats": (
            quality_stats.to_dict() if quality_stats is not None else None
        ),
        "quality_prior_eeg_positions": (
            list(quality_eeg_positions) if quality_eeg_positions is not None else None
        ),
        "quality_prior_mad_threshold": args.quality_prior_mad_threshold,
        "quality_prior_slope": args.quality_prior_slope,
        "quality_prior_min_weight": args.quality_prior_min_weight,
        "quality_prior_min_scale": args.quality_prior_min_scale,
        "quality_prior_mode": quality_prior_mode,
        "class_weight_counts_train_only": counts,
        "sliding_window_original_count_train": int(
            getattr(train_loader.dataset, "sliding_window_original_count", len(train_loader.dataset))
        ),
        "sliding_window_expanded_count_train": int(
            getattr(train_loader.dataset, "sliding_window_expanded_count", len(train_loader.dataset))
        ),
        "sliding_window_original_class_counts_train": getattr(
            train_loader.dataset, "sliding_window_original_class_counts", counts
        ),
        "sliding_window_expanded_class_counts_train": getattr(
            train_loader.dataset, "sliding_window_expanded_class_counts", counts
        ),
        "sliding_window_added_class_counts_train": getattr(
            train_loader.dataset, "sliding_window_added_class_counts", [0, 0, 0, 0, 0]
        ),
        "class_weights": weights.detach().cpu().tolist(),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "lr": args.lr,
        "lr_drop_epoch": args.lr_drop_epoch,
        "lr_after_drop": args.lr_after_drop,
        "weight_decay": args.weight_decay,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "focal_gamma": args.focal_gamma,
        "supcon_weight": args.supcon_weight,
        "supcon_temperature": args.supcon_temperature,
        "supcon_projection_dim": args.supcon_projection_dim,
        "supcon_projection_hidden": args.supcon_projection_hidden,
        "n1_loss_multiplier": args.n1_loss_multiplier,
        "sampler": args.sampler,
        "sampler_replacement": args.sampler_replacement,
        "class_balance_power": args.class_balance_power,
        "transition_window": args.transition_window,
        "transition_weight": args.transition_weight,
        "augment_noise_std": args.augment_noise_std,
        "augment_noise_positions": args.augment_noise_positions,
        "augment_scale_std": args.augment_scale_std,
        "augment_scale_positions": args.augment_scale_positions,
        "augment_time_shift": args.augment_time_shift,
        "augment_time_shift_positions": args.augment_time_shift_positions,
        "augment_time_mask_samples": args.augment_time_mask_samples,
        "augment_time_mask_prob": args.augment_time_mask_prob,
        "augment_time_mask_positions": args.augment_time_mask_positions,
        "sliding_window_delta_seconds": args.sliding_window_delta_seconds,
        "sliding_window_max_extra_per_run": args.sliding_window_max_extra_per_run,
        "sliding_window_train_only": args.sliding_window_delta_seconds > 0.0,
        "seed_scope": args.seed_scope,
        "seed": args.seed,
        "fold_seed": fold_seed if args.seed_scope == "fold" else None,
        "checkpoint_selection": args.checkpoint_selection,
        "save_selection_trajectory": args.save_selection_trajectory,
        "outer_test_deferred": args.defer_outer_test,
    }
    save_json(fold_dir / "metadata.json", metadata)

    start_epoch = 1
    best_val_accuracy = -1.0
    best_val_macro_f1 = -1.0
    best_checkpoint_key: tuple[float, ...] = tuple([-1.0])
    last_ckpt = fold_dir / "checkpoint_last.pt"
    best_ckpt = fold_dir / "checkpoint_best.pt"
    if args.resume == "auto" and last_ckpt.exists():
        checkpoint = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if auxiliary_head is not None and checkpoint.get("auxiliary_head_state") is not None:
            auxiliary_head.load_state_dict(checkpoint["auxiliary_head_state"])
        if projection_head is not None and checkpoint.get("projection_head_state") is not None:
            projection_head.load_state_dict(checkpoint["projection_head_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_accuracy = float(checkpoint.get("best_val_accuracy", -1.0))
        best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", -1.0))
        best_checkpoint_key = tuple(checkpoint.get("best_checkpoint_key", [best_val_accuracy]))
        print(f"fold {fold_id}: resumed from epoch {checkpoint['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.lr_drop_epoch + 1:
            for group in optimizer.param_groups:
                group["lr"] = args.lr_after_drop
        t0 = time.time()
        model.train()
        if auxiliary_head is not None:
            auxiliary_head.train()
        if projection_head is not None:
            projection_head.train()
        losses: list[float] = []
        ce_losses: list[float] = []
        supcon_losses: list[float] = []
        auxiliary_losses: list[float] = []
        orthogonality_losses: list[float] = []
        train_true: list[int] = []
        train_pred: list[int] = []
        for step, batch in enumerate(train_loader, start=1):
            if args.max_train_batches and step > args.max_train_batches:
                break
            x, y, quality_prior, entropy_features = unpack_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            x = x.float().to(device)
            y = y.to(device)
            quality_prior = quality_prior.float().to(device)
            if entropy_features is not None:
                entropy_features = entropy_features.float().to(device)
            if feature_capture is not None:
                feature_capture.features = None
            if auxiliary_capture is not None:
                auxiliary_capture.features = None
            if orthogonality_capture_short is not None:
                orthogonality_capture_short.features = None
                orthogonality_capture_long.features = None
            out = forward_model(model, x, quality_prior, entropy_features)
            ce_loss = criterion(out, y)
            auxiliary_loss = out.sum() * 0.0
            if auxiliary_head is not None and args.deep_supervision_weight > 0.0:
                if auxiliary_capture is None or auxiliary_capture.features is None:
                    raise RuntimeError("Deep-supervision tap did not capture an output")
                auxiliary_logits = auxiliary_head(
                    aggregate_probe_features(auxiliary_capture.features)
                )
                auxiliary_loss = criterion(auxiliary_logits, y)
            supcon_loss = out.sum() * 0.0
            if projection_head is not None:
                if feature_capture is None or feature_capture.features is None:
                    raise RuntimeError("SupCon feature hook did not capture fc input")
                projections = projection_head(feature_capture.features)
                supcon_loss = supervised_contrastive_loss(
                    projections,
                    y,
                    float(args.supcon_temperature),
                )
            orthogonality_loss = out.sum() * 0.0
            if args.branch_orthogonality_weight > 0.0:
                orthogonality_loss = branch_orthogonality_loss(
                    orthogonality_capture_short.features,
                    orthogonality_capture_long.features,
                )
            loss = (
                ce_loss
                + float(args.supcon_weight) * supcon_loss
                + float(args.deep_supervision_weight) * auxiliary_loss
                + float(args.branch_orthogonality_weight) * orthogonality_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(ce_loss.detach().cpu()))
            supcon_losses.append(float(supcon_loss.detach().cpu()))
            auxiliary_losses.append(float(auxiliary_loss.detach().cpu()))
            orthogonality_losses.append(float(orthogonality_loss.detach().cpu()))
            train_true.extend(y.detach().cpu().tolist())
            train_pred.extend(out.argmax(1).detach().cpu().tolist())

        val_eval = evaluate(model, val_loader, criterion, device, args.max_eval_batches)
        train_acc = accuracy_score(train_true, train_pred) if train_true else None
        train_macro_f1 = (
            f1_score(train_true, train_pred, average="macro", zero_division=0)
            if train_true
            else None
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_ce_loss": float(np.mean(ce_losses)) if ce_losses else None,
            "train_supcon_loss": float(np.mean(supcon_losses)) if supcon_losses else None,
            "train_auxiliary_loss": float(np.mean(auxiliary_losses)) if auxiliary_losses else None,
            "train_branch_orthogonality_loss": float(np.mean(orthogonality_losses)) if orthogonality_losses else None,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_eval["loss"],
            "val_accuracy": val_eval["accuracy"],
            "val_macro_f1": val_eval["macro_f1"],
            "val_weighted_f1": val_eval["weighted_f1"],
            "val_kappa": val_eval["kappa"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(time.time() - t0, 3),
        }
        write_history_row(fold_dir / "history.csv", row)
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "projection_head_state": (
                projection_head.state_dict() if projection_head is not None else None
            ),
            "auxiliary_head_state": (
                auxiliary_head.state_dict() if auxiliary_head is not None else None
            ),
            "optimizer_state": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_val_macro_f1": best_val_macro_f1,
            "best_checkpoint_key": list(best_checkpoint_key),
            "metadata": metadata,
        }
        current_key = checkpoint_key(args, val_eval)
        if current_key is not None and current_key > best_checkpoint_key:
            best_val_accuracy = float(val_eval["accuracy"])
            best_val_macro_f1 = float(val_eval["macro_f1"])
            best_checkpoint_key = current_key
            ckpt["best_val_accuracy"] = best_val_accuracy
            ckpt["best_val_macro_f1"] = best_val_macro_f1
            ckpt["best_checkpoint_key"] = list(best_checkpoint_key)
            torch.save(ckpt, best_ckpt)
        torch.save(ckpt, last_ckpt)
        if args.save_selection_trajectory:
            trajectory_dir = fold_dir / "selection_trajectory"
            trajectory_dir.mkdir(exist_ok=True)
            validation_metrics = {
                key: val_eval[key]
                for key in ("loss", "accuracy", "macro_f1", "weighted_f1", "kappa", "classification_report")
            }
            epoch_checkpoint = trajectory_dir / f"epoch_{epoch:02d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "metadata": metadata,
                    "validation_metrics": validation_metrics,
                },
                epoch_checkpoint,
            )
            save_json(
                trajectory_dir / f"epoch_{epoch:02d}.json",
                {
                    "epoch": epoch,
                    "checkpoint": str(epoch_checkpoint),
                    "validation_metrics": validation_metrics,
                },
            )
        print(
            f"fold {fold_id} epoch {epoch}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_eval['accuracy']:.4f} "
            f"val_macro_f1={val_eval['macro_f1']:.4f} "
            f"sec={row['elapsed_sec']}"
        )

    if args.defer_outer_test:
        summary = {
            **metadata,
            "completed_epochs": args.epochs,
            "best_val_accuracy": best_val_accuracy,
            "best_val_macro_f1": best_val_macro_f1,
            "best_checkpoint_key": list(best_checkpoint_key),
            "final_test_accuracy": None,
            "final_test_macro_f1": None,
            "final_test_weighted_f1": None,
            "final_test_kappa": None,
        }
        save_json(fold_dir / "summary.json", summary)
        for handle in (feature_hook_handle, auxiliary_hook_handle, *orthogonality_hook_handles):
            if handle is not None:
                handle.remove()
        print(f"fold {fold_id}: outer-test records deferred; validation trajectory saved")
        return summary

    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if auxiliary_head is not None and checkpoint.get("auxiliary_head_state") is not None:
            auxiliary_head.load_state_dict(checkpoint["auxiliary_head_state"])
        if projection_head is not None and checkpoint.get("projection_head_state") is not None:
            projection_head.load_state_dict(checkpoint["projection_head_state"])
    final_eval = evaluate(model, test_loader, criterion, device, args.max_eval_batches)
    summary = {
        **metadata,
        "completed_epochs": args.epochs,
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_val_macro_f1,
        "best_checkpoint_key": list(best_checkpoint_key),
        "final_test_loss": final_eval["loss"],
        "final_test_accuracy": final_eval["accuracy"],
        "final_test_macro_f1": final_eval["macro_f1"],
        "final_test_weighted_f1": final_eval["weighted_f1"],
        "final_test_kappa": final_eval["kappa"],
    }
    save_json(fold_dir / "summary.json", summary)
    save_json(fold_dir / "classification_report.json", final_eval["classification_report"])
    np.save(fold_dir / "y_true.npy", np.asarray(final_eval["y_true"], dtype=np.int64))
    np.save(fold_dir / "y_pred.npy", np.asarray(final_eval["y_pred"], dtype=np.int64))
    final_logits = np.asarray(final_eval["logits"], dtype=np.float32)
    np.save(fold_dir / "logits.npy", final_logits)
    shifted = final_logits - final_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    y_prob = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    np.save(fold_dir / "y_prob.npy", y_prob.astype(np.float32))
    if feature_hook_handle is not None:
        feature_hook_handle.remove()
    if auxiliary_hook_handle is not None:
        auxiliary_hook_handle.remove()
    for hook_handle in orthogonality_hook_handles:
        hook_handle.remove()
    return summary


def aggregate_summaries(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        f1_score,
    )

    keys = [
        "fold_id",
        "test_subjects",
        "best_val_accuracy",
        "final_test_accuracy",
        "final_test_macro_f1",
        "final_test_weighted_f1",
        "final_test_kappa",
    ]
    with (output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k) for k in keys})
    numeric_keys = [
        "best_val_accuracy",
        "final_test_accuracy",
        "final_test_macro_f1",
        "final_test_weighted_f1",
        "final_test_kappa",
    ]
    aggregate: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(s[key]) for s in summaries if s.get(key) is not None]
        aggregate[f"{key}_mean"] = float(np.mean(values)) if values else None
        aggregate[f"{key}_std"] = float(np.std(values)) if values else None
    y_true = []
    y_pred = []
    for row in summaries:
        fold_dir = output_dir / f"fold_{int(row['fold_id']):02d}"
        y_true_path = fold_dir / "y_true.npy"
        y_pred_path = fold_dir / "y_pred.npy"
        if y_true_path.exists() and y_pred_path.exists():
            y_true.append(np.load(y_true_path))
            y_pred.append(np.load(y_pred_path))
    if y_true and y_pred:
        y = np.concatenate(y_true)
        pred = np.concatenate(y_pred)
        aggregate["pooled_accuracy"] = float(accuracy_score(y, pred))
        aggregate["pooled_macro_f1"] = float(
            f1_score(y, pred, average="macro", labels=list(range(5)), zero_division=0)
        )
        aggregate["pooled_weighted_f1"] = float(
            f1_score(y, pred, average="weighted", labels=list(range(5)), zero_division=0)
        )
        aggregate["pooled_kappa"] = float(cohen_kappa_score(y, pred))
        aggregate["pooled_classification_report"] = classification_report(
            y,
            pred,
            labels=list(range(5)),
            target_names=["W", "N1", "N2", "N3", "REM"],
            output_dict=True,
            zero_division=0,
        )
    save_json(output_dir / "aggregate_metrics.json", aggregate)


def main() -> None:
    args = parse_args()
    import_attnsleep(args.attnsleep_dir)
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fold_id in parse_fold_ids(args.fold_ids, args.num_folds):
        summary_path = args.output_dir / f"fold_{fold_id:02d}" / "summary.json"
        if args.skip_completed and summary_path.exists():
            print(f"fold {fold_id}: using completed summary {summary_path}")
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        summaries.append(train_fold(args, fold_id, device))
    if not args.no_aggregate:
        aggregate_summaries(args.output_dir, summaries)
    print("WROTE", args.output_dir)


if __name__ == "__main__":
    main()

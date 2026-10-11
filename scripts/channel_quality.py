"""Record-level signal quality statistics for multichannel Sleep-EDF models."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RecordQualityStats:
    first_eeg_position: int
    second_eeg_position: int
    center: float
    mad: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_eeg_position": self.first_eeg_position,
            "second_eeg_position": self.second_eeg_position,
            "center": self.center,
            "mad": self.mad,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordQualityStats":
        return cls(
            first_eeg_position=int(data["first_eeg_position"]),
            second_eeg_position=int(data["second_eeg_position"]),
            center=float(data["center"]),
            mad=float(data["mad"]),
        )


def parse_index_pair(value: str) -> tuple[int, int]:
    values = [int(v.strip()) for v in value.split(",") if v.strip()]
    if len(values) != 2 or values[0] == values[1] or any(v < 0 for v in values):
        raise ValueError(f"Expected two distinct non-negative indices, got {value!r}")
    return values[0], values[1]


def select_npz_channels(
    file_path: str,
    channel_indices: list[int] | None,
) -> np.ndarray:
    with np.load(file_path, allow_pickle=True) as data:
        x = np.asarray(data["x"], dtype=np.float32)
    if x.ndim == 2:
        x = x[:, np.newaxis, :]
    if channel_indices is not None:
        x = x[:, channel_indices, :]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D signal array in {file_path}, got {x.shape}")
    return x


def record_diff_rms_ratio(
    x: np.ndarray,
    first_eeg_position: int,
    second_eeg_position: int,
) -> float:
    if max(first_eeg_position, second_eeg_position) >= x.shape[1]:
        raise IndexError(
            f"EEG quality position exceeds input channels: positions="
            f"{(first_eeg_position, second_eeg_position)}, shape={x.shape}"
        )
    eeg = x[:, [first_eeg_position, second_eeg_position], :]
    diff = np.diff(eeg, axis=2)
    diff_rms = np.sqrt(np.mean(diff * diff, axis=2))
    median_diff_rms = np.median(diff_rms, axis=0)
    return float(median_diff_rms[0] / max(float(median_diff_rms[1]), 1e-12))


def fit_record_quality_stats(
    training_files: list[str],
    channel_indices: list[int] | None,
    eeg_positions: tuple[int, int],
) -> RecordQualityStats:
    ratios = [
        record_diff_rms_ratio(
            select_npz_channels(file_path, channel_indices),
            *eeg_positions,
        )
        for file_path in sorted(training_files)
    ]
    if not ratios:
        raise ValueError("No training records available for record quality statistics")
    values = np.asarray(ratios, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return RecordQualityStats(
        first_eeg_position=eeg_positions[0],
        second_eeg_position=eeg_positions[1],
        center=center,
        mad=max(mad, 1e-6),
    )


def split_train_validation_files_by_strategy(
    train_files: list[str],
    val_subjects_per_fold: int,
    strategy: str = "tail",
    channel_indices: list[int] | None = None,
    eeg_positions: tuple[int, int] = (0, 1),
    validation_subject_ids: list[str] | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Split training records into model-train and validation sets.

    ``tail`` preserves the original deterministic behavior. ``quality-extremes``
    selects validation subjects that cover high and low EEG diff-RMS ratio
    outliers using only records available inside the outer training fold.
    ``fixed-subjects`` uses a predeclared subject-id list supplied by the caller.
    """
    if val_subjects_per_fold <= 0:
        return train_files, [], [], {"validation_strategy": strategy}

    by_subject: dict[str, list[str]] = {}
    for file_path in sorted(train_files):
        subject = os.path.basename(file_path)[3:5]
        by_subject.setdefault(subject, []).append(file_path)
    subjects = sorted(by_subject)
    if val_subjects_per_fold >= len(subjects):
        raise ValueError(
            f"val_subjects_per_fold={val_subjects_per_fold} leaves no training subjects"
        )

    info: dict[str, Any] = {
        "validation_strategy": strategy,
        "val_subjects_per_fold": val_subjects_per_fold,
    }
    if strategy == "tail":
        val_subjects = subjects[-val_subjects_per_fold:]
    elif strategy == "fixed-subjects":
        if not validation_subject_ids:
            raise ValueError("fixed-subjects requires validation_subject_ids")
        val_subjects = sorted(str(subject).zfill(2) for subject in validation_subject_ids)
        if len(val_subjects) != val_subjects_per_fold or len(set(val_subjects)) != len(val_subjects):
            raise ValueError(
                "validation_subject_ids must contain exactly val_subjects_per_fold unique ids"
            )
        missing = sorted(set(val_subjects) - set(subjects))
        if missing:
            raise ValueError(f"Fixed validation subjects are not in the outer training fold: {missing}")
        info["selection_rule"] = "caller-predeclared subject ids; no labels or signal quality used"
    elif strategy == "quality-extremes":
        ratios: list[tuple[str, str, float]] = []
        for subject in subjects:
            for file_path in sorted(by_subject[subject]):
                ratio = record_diff_rms_ratio(
                    select_npz_channels(file_path, channel_indices),
                    *eeg_positions,
                )
                ratios.append((subject, os.path.basename(file_path), ratio))
        if not ratios:
            raise ValueError("No records available for validation split selection")
        values = np.asarray([row[2] for row in ratios], dtype=np.float64)
        center = float(np.median(values))
        mad = max(float(np.median(np.abs(values - center))), 1e-6)
        subject_scores: dict[str, dict[str, Any]] = {}
        for subject in subjects:
            subject_rows = [row for row in ratios if row[0] == subject]
            z_values = [(row[2] - center) / mad for row in subject_rows]
            subject_scores[subject] = {
                "max_z": float(max(z_values)),
                "min_z": float(min(z_values)),
                "max_abs_z": float(max(abs(value) for value in z_values)),
                "records": [
                    {
                        "record": row[1],
                        "quality_ratio": float(row[2]),
                        "quality_robust_z": float((row[2] - center) / mad),
                    }
                    for row in subject_rows
                ],
            }
        selected: list[str] = []

        def add_subject(subject: str) -> None:
            if subject not in selected:
                selected.append(subject)

        if val_subjects_per_fold == 1:
            add_subject(
                sorted(
                    subjects,
                    key=lambda subject: (
                        -subject_scores[subject]["max_abs_z"],
                        subject,
                    ),
                )[0]
            )
        else:
            add_subject(
                sorted(
                    subjects,
                    key=lambda subject: (
                        -subject_scores[subject]["max_z"],
                        subject,
                    ),
                )[0]
            )
            add_subject(
                sorted(
                    [subject for subject in subjects if subject not in selected],
                    key=lambda subject: (
                        subject_scores[subject]["min_z"],
                        subject,
                    ),
                )[0]
            )
        for subject in sorted(
            [subject for subject in subjects if subject not in selected],
            key=lambda subject: (
                -subject_scores[subject]["max_abs_z"],
                subject,
            ),
        ):
            if len(selected) >= val_subjects_per_fold:
                break
            add_subject(subject)
        val_subjects = sorted(selected)
        info.update(
            {
                "quality_center": center,
                "quality_mad": mad,
                "quality_eeg_positions": list(eeg_positions),
                "selected_subject_scores": {
                    subject: subject_scores[subject] for subject in val_subjects
                },
            }
        )
    else:
        raise ValueError(f"Unsupported validation strategy: {strategy}")

    val_set = set(val_subjects)
    model_train_files: list[str] = []
    val_files: list[str] = []
    for subject in subjects:
        if subject in val_set:
            val_files.extend(by_subject[subject])
        else:
            model_train_files.extend(by_subject[subject])
    info["validation_subjects"] = val_subjects
    return model_train_files, val_files, val_subjects, info


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def soft_channel_prior(
    x: np.ndarray,
    stats: RecordQualityStats,
    mad_threshold: float = 3.0,
    slope: float = 2.0,
    min_weight: float = 0.1,
) -> tuple[np.ndarray, float, float]:
    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive")
    if slope <= 0:
        raise ValueError("slope must be positive")
    if min_weight <= 0 or min_weight > 1:
        raise ValueError("min_weight must be in (0, 1]")

    ratio = record_diff_rms_ratio(
        x,
        stats.first_eeg_position,
        stats.second_eeg_position,
    )
    robust_z = (ratio - stats.center) / stats.mad
    high_first = _sigmoid(slope * (robust_z - mad_threshold))
    high_second = _sigmoid(slope * (-robust_z - mad_threshold))

    prior = np.ones(x.shape[1], dtype=np.float32)
    prior[stats.first_eeg_position] = 1.0 - (1.0 - min_weight) * high_first
    prior[stats.second_eeg_position] = 1.0 - (1.0 - min_weight) * high_second
    prior *= float(len(prior)) / float(prior.sum())
    return prior, ratio, robust_z


def piecewise_channel_prior(
    x: np.ndarray,
    stats: RecordQualityStats,
    mad_threshold: float = 3.0,
    slope: float = 2.0,
    min_weight: float = 0.001,
) -> tuple[np.ndarray, float, float]:
    """Keep inlier records uniform and suppress only threshold exceedances."""
    if mad_threshold <= 0:
        raise ValueError("mad_threshold must be positive")
    if slope <= 0:
        raise ValueError("slope must be positive")
    if min_weight <= 0 or min_weight > 1:
        raise ValueError("min_weight must be in (0, 1]")

    ratio = record_diff_rms_ratio(
        x,
        stats.first_eeg_position,
        stats.second_eeg_position,
    )
    robust_z = (ratio - stats.center) / stats.mad
    high_excess = max(robust_z - mad_threshold, 0.0)
    low_excess = max(-robust_z - mad_threshold, 0.0)

    first_weight = min_weight + (1.0 - min_weight) * np.exp(
        -slope * high_excess
    )
    second_weight = min_weight + (1.0 - min_weight) * np.exp(
        -slope * low_excess
    )
    prior = np.ones(x.shape[1], dtype=np.float32)
    prior[stats.first_eeg_position] = float(first_weight)
    prior[stats.second_eeg_position] = float(second_weight)
    prior *= float(len(prior)) / float(prior.sum())
    return prior, ratio, robust_z

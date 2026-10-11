from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "mit-bih-polysomnographic-database-1.0.0"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache"
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "runs"


STAGE_TO_INDEX = {
    "W": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 3,
    "R": 4,
}

INDEX_TO_STAGE = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3/N4",
    4: "REM",
}

REFERENCE_RECORDS = [
    "slp01a",
    "slp01b",
    "slp02a",
    "slp02b",
    "slp03",
    "slp04",
    "slp14",
    "slp16",
    "slp32",
    "slp37",
    "slp41",
    "slp45",
    "slp48",
    "slp59",
    "slp60",
    "slp61",
    "slp66",
    "slp67x",
]


@dataclass(frozen=True)
class ReproductionConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR
    run_dir: Path = DEFAULT_RUN_DIR
    sample_rate: int = 250
    epoch_seconds: int = 30
    target_points: int = 9000
    input_normalization: str = "paper_scale"
    input_scale: float = 200.0
    context_radius: int = 2
    cnn_feature_dim: int = 288
    n_classes: int = 5
    seed: int = 42

    @property
    def raw_epoch_points(self) -> int:
        return self.sample_rate * self.epoch_seconds

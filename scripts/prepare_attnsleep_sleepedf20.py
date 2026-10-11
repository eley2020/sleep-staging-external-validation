"""Prepare Sleep-EDF-20 inputs for the AttnSleep baseline.

This script creates a symlink directory containing the first 20 Sleep-Cassette
subjects, then calls AttnSleep's PhysioNet preprocessor to generate one-channel
EEG NPZ files. The resulting split is subject-wise because AttnSleep groups
files by SC filename characters 3:5.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sleep-cassette-dir",
        type=Path,
        default=Path(
            "data/sleep-edfx-1.0.0/"
            "sleep-edf-database-expanded-1.0.0/sleep-cassette"
        ),
    )
    parser.add_argument(
        "--raw-link-dir",
        type=Path,
        default=Path("data/attnsleep_sleepedf20_raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/attnsleep_sleepedf20_npz/fpzcz"),
    )
    parser.add_argument(
        "--attnsleep-dir",
        type=Path,
        default=Path("external_baselines/AttnSleep"),
    )
    parser.add_argument("--select-ch", default="EEG Fpz-Cz")
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--skip-preprocess", action="store_true")
    return parser.parse_args()


def build_raw_links(src: Path, dst: Path, n_subjects: int) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Sleep-Cassette directory not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    for old_link in dst.glob("*.edf"):
        old_link.unlink()

    psg_files: list[Path] = []
    for sid in range(n_subjects):
        psg_files.extend(sorted(src.glob(f"SC4{sid:02d}[12]E*-PSG.edf")))

    if not psg_files:
        raise FileNotFoundError(f"No PSG files found under {src}")

    for psg in psg_files:
        hyp = sorted(src.glob(f"{psg.name[:6]}*-Hypnogram.edf"))
        if not hyp:
            raise FileNotFoundError(f"Missing hypnogram for {psg.name}")
        for file_path in (psg, hyp[0]):
            link = dst / file_path.name
            link.symlink_to(file_path.resolve())

    subjects = sorted({p.name[3:5] for p in psg_files})
    print(f"Linked PSG files: {len(list(dst.glob('*PSG.edf')))}")
    print(f"Linked hypnogram files: {len(list(dst.glob('*Hypnogram.edf')))}")
    print(f"Subjects: {len(subjects)} {subjects}")


def run_preprocess(attnsleep_dir: Path, raw_dir: Path, out_dir: Path, select_ch: str) -> None:
    cmd = [
        sys.executable,
        "prepare_datasets/prepare_physionet.py",
        "--data_dir",
        str(raw_dir.resolve()),
        "--output_dir",
        str(out_dir.resolve()),
        "--select_ch",
        select_ch,
    ]
    subprocess.run(cmd, cwd=attnsleep_dir, check=True)


def main() -> None:
    args = parse_args()
    build_raw_links(args.sleep_cassette_dir, args.raw_link_dir, args.subjects)
    if not args.skip_preprocess:
        run_preprocess(args.attnsleep_dir, args.raw_link_dir, args.output_dir, args.select_ch)


if __name__ == "__main__":
    main()

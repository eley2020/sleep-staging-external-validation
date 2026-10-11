"""Create fixed subject-disjoint SHHS adaptation splits from valid zero-shot records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record-metrics",
        type=Path,
        default=Path("outputs/external_validation/shhs/frozen_ensemble_identity_shhs1/record_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="20260813,20260814,20260815")
    parser.add_argument("--train-subjects", type=int, default=50)
    parser.add_argument("--validation-subjects", type=int, default=20)
    parser.add_argument("--test-subjects", type=int, default=500)
    args = parser.parse_args()
    with args.record_metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    subjects = sorted({row["subject"] for row in rows})
    required = args.train_subjects + args.validation_subjects + args.test_subjects
    if required > len(subjects):
        raise ValueError(f"Requested {required} subjects but only {len(subjects)} valid SHHS1 records exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed_text in args.seeds.split(","):
        seed = int(seed_text)
        sampled = np.random.default_rng(seed).choice(subjects, size=required, replace=False).tolist()
        split = {
            "train_subjects": sorted(sampled[: args.train_subjects]),
            "validation_subjects": sorted(sampled[args.train_subjects : args.train_subjects + args.validation_subjects]),
            "test_subjects": sorted(sampled[args.train_subjects + args.validation_subjects :]),
        }
        payload = {
            "protocol": "fixed SHHS1 subject-disjoint small-sample adaptation split; unselected subjects are unused",
            "seed": seed,
            "n_valid_shhs1_subjects": len(subjects),
            "n_unused_subjects": len(subjects) - required,
            "split": split,
        }
        output = args.output_dir / f"seed{seed}.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        subjects_file = args.output_dir / f"seed{seed}_trainval_subjects.txt"
        subjects_file.write_text("\n".join(split["train_subjects"] + split["validation_subjects"]) + "\n", encoding="utf-8")
        print(f"{output}: train={len(split['train_subjects'])} val={len(split['validation_subjects'])} test={len(split['test_subjects'])}")


if __name__ == "__main__":
    main()

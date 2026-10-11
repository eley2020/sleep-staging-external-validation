"""Cache selected SHHS records for target-domain adaptation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_attnsleep_shhs_ensemble import load_record, subject_id, xml_for
except ModuleNotFoundError:
    from evaluate_attnsleep_shhs_ensemble import load_record, subject_id, xml_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shhs-root", type=Path, default=Path("/path/to/datasets/SHHS/polysomnography"))
    parser.add_argument("--cohort", choices=["shhs1", "shhs2"], default="shhs1")
    parser.add_argument("--subjects-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--w-edge-mins", type=int, default=30)
    args = parser.parse_args()
    subjects = {line.strip() for line in args.subjects_file.read_text().splitlines() if line.strip()}
    edf_dir = args.shhs_root / "edfs" / args.cohort
    ann_dir = args.shhs_root / "annotations-events-nsrr" / args.cohort
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for edf in sorted(edf_dir.glob("*.edf")):
        if subject_id(edf.stem) in subjects and xml_for(edf, ann_dir).exists():
            pairs.append((edf, xml_for(edf, ann_dir)))
    manifest = []
    for number, (edf, xml) in enumerate(pairs, 1):
        out = args.output_dir / f"{edf.stem}.npz"
        if not out.exists():
            x, y, meta = load_record(edf, xml, args.w_edge_mins)
            np.savez_compressed(out, x=x, y=y, normalization=np.asarray("physical-volts"), subject=np.asarray(subject_id(edf.stem)))
        else:
            with np.load(out, allow_pickle=True) as data:
                meta = {"n_epochs": int(len(data["y"]))}
        manifest.append({"record": edf.stem, "subject": subject_id(edf.stem), **meta})
        if number % 10 == 0 or number == len(pairs):
            print(f"{number}/{len(pairs)} {edf.stem}", flush=True)
    (args.output_dir / "manifest.json").write_text(json.dumps({"cohort": args.cohort, "subjects": sorted(subjects), "records": manifest}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

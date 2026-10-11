"""Collect ISRUC original-montage budget reports with explicit protocol labels."""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("reports", nargs="+", help="ISRUC fine-tuning report JSON paths")
    args = parser.parse_args()
    rows = []
    for raw in args.reports:
        for path_text in glob.glob(raw):
            path = Path(path_text)
            report = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "report": str(path),
                "seed": report["seed"],
                "calibration_subjects": report["calibration_subjects"],
                "n_calibration_subjects": len(report["calibration_subjects"]),
                "n_held_out_subjects": len(report["held_out_subjects"]),
                "frozen": report["frozen_held_out"],
                "finetuned": report["finetuned_held_out"],
                "delta": report["delta"],
            })
    if not rows:
        raise SystemExit("No ISRUC budget reports found")
    rows.sort(key=lambda row: (row["n_calibration_subjects"], row["seed"], row["n_held_out_subjects"]))
    budget_summary = {}
    for budget in sorted({row["n_calibration_subjects"] for row in rows}):
        subset = [row for row in rows if row["n_calibration_subjects"] == budget]
        def stats(key, source):
            values = [float(row[source][key]) for row in subset]
            return {
                "mean": statistics.mean(values),
                "sd": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        deltas = [float(row["delta"]["macro_f1"]) for row in subset]
        budget_summary[str(budget)] = {
            "n_selections": len(subset),
            "held_out_subject_counts": sorted({row["n_held_out_subjects"] for row in subset}),
            "macro_f1_frozen": stats("macro_f1", "frozen"),
            "macro_f1_finetuned": stats("macro_f1", "finetuned"),
            "macro_f1_delta": {"mean": statistics.mean(deltas), "sd": statistics.pstdev(deltas), "min": min(deltas), "max": max(deltas)},
            "positive_delta_count": sum(value > 0 for value in deltas),
            "positive_delta_fraction": sum(value > 0 for value in deltas) / len(deltas),
        }
    payload = {
        "protocol": "ISRUC original-montage budget sensitivity; explicit calibration selections and held-out subject counts retained per row",
        "rows": rows,
        "budget_summary": budget_summary,
        "interpretation": "One-subject runs use 12 held-out subjects for three calibration selections plus one seven-subject sensitivity split; five-subject runs include one matched full-finetuning selection; ten-subject runs include three explicit calibration selections, each with seven held-out subjects because the original montage pool has 17 subjects.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

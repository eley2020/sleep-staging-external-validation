"""Locked subject-wise checkpoint-selection study; GPU training/inference only.

Each training trajectory supplies both selection rules. Inner assessment subjects
are excluded from fitting and checkpoint selection. Outer tests remain deferred.
The study is exploratory because Sleep-EDF-20 has been inspected previously.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_baselines/AttnSleep"))

STAGES = ("W", "N1", "N2", "N3", "REM")
GUARDS = {"accuracy": 0.005, "n1_recall": 0.02, "n3_f1": 0.01}


def stage_metrics(metrics):
    report = metrics["classification_report"]
    return {
        **{key: float(metrics[key]) for key in ("accuracy", "macro_f1", "kappa")},
        "n1_precision": float(report["N1"]["precision"]),
        "n1_recall": float(report["N1"]["recall"]),
        "n1_f1": float(report["N1"]["f1-score"]),
        "n3_f1": float(report["N3"]["f1-score"]),
    }


def select_epochs(rows):
    if not rows:
        raise ValueError("No complete validation trajectory")
    # Match the original accuracy-only rule: earliest epoch wins a tie.
    accuracy = max(rows, key=lambda row: (row["validation_metrics"]["accuracy"], -row["epoch"]))
    reference = stage_metrics(accuracy["validation_metrics"])
    eligible = [
        row for row in rows
        if all(stage_metrics(row["validation_metrics"])[key] >= reference[key] - tolerance
               for key, tolerance in GUARDS.items())
    ]
    macro = max(eligible, key=lambda row: (
        row["validation_metrics"]["macro_f1"], row["validation_metrics"]["accuracy"], -row["epoch"]
    ))
    return {"accuracy": accuracy, "guarded_macro": macro}


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--np-data-dir", type=Path, default=ROOT / "data/sleepedf20_multichannel_npz/eeg2_eog_emg_physical_volts")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/attnsleep_selection_study_20261010")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def build_protocol(args):
    from utils.util import load_folds_data
    folds = load_folds_data(str(args.np_data_dir.resolve()), 20)
    subjects = sorted({path.name[3:5] for path in args.np_data_dir.glob("*.npz")})
    rng = np.random.default_rng(20261010)
    order = list(rng.permutation(subjects))
    used_assessment = set()
    panels = []
    for fold_id in (0, 4, 8, 12, 16):
        training, testing = folds[fold_id]
        available = {Path(path).name[3:5] for path in training}
        assessment = [sid for sid in order if sid in available and sid not in used_assessment][:2]
        used_assessment.update(assessment)
        eligible = sorted(available - set(assessment))
        selection = sorted(rng.choice(eligible, 4, replace=False).tolist())
        fitting = sorted(available - set(assessment) - set(selection))
        outer = sorted({Path(path).name[3:5] for path in testing})
        groups = [set(fitting), set(selection), set(assessment), set(outer)]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
            raise RuntimeError("Subject overlap in split")
        panels.append({"fold_id": fold_id, "fit": fitting, "selection": selection,
                       "assessment": assessment, "outer_test_deferred": outer})
    inventory = []
    for path in sorted(args.np_data_dir.glob("*.npz")):
        stat = path.stat()
        inventory.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "study": "paired checkpoint rules on identical training trajectories",
        "created_date": "2026-10-10",
        "data_dir": str(args.np_data_dir.resolve()), "input_inventory": inventory,
        "epochs": 30, "batch_size": 64, "seeds": [7777, 8888, 9999],
        "gpu_ids": [0, 1], "seed_scope": "fold", "channels": [0, 1, 2, 3],
        "fusion": "quality-gated", "channel_dropout": 0.15, "weight_decay": 0.001,
        "learning_rate": 0.001, "lr_drop_epoch": 10, "lr_after_drop": 0.0001,
        "selection_guards": GUARDS, "panels": panels,
        "assessment_subject_count": len(used_assessment),
        "interpretation": "Exploratory inner assessment; prior project inspection precludes pristine-test claims. Outer test signals/labels are not loaded.",
        "stage1_decision": {
            "minimum_mean_assessment_macro_delta": 0.005,
            "minimum_positive_seed_means": 2,
            "minimum_mean_accuracy_delta": -0.005,
            "minimum_mean_n1_recall_delta": -0.02,
            "minimum_mean_n1_f1_delta": 0.0,
            "minimum_mean_n3_f1_delta": -0.01,
            "note": "Operational thresholds fixed before outcomes; no population significance claim from overlapping fits or repeated seeds.",
        },
        "later_stages": [
            "Stage 2: same panels/seeds/30 epochs, locked stage-1 selector; compare four versus three channels. Do not tune on external labels.",
            "Stage 3: prespecify context residual, fixed-fusion and matched residual controls after stages 1/2; no automatic promotion of architecture.",
        ],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def assess(args, seed, panel, run_dir, protocol_digest, channel_indices=(0, 1, 2, 3)):
    import torch
    from torch.utils.data import DataLoader
    from model.model import AttnSleep
    from scripts.train_attnsleep_multichannel_baseline import NumpySleepDataset, evaluate

    folder = run_dir / f"fold_{panel['fold_id']:02d}"
    result_path = folder / "selection_comparison.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["protocol_sha256"] != protocol_digest:
            raise RuntimeError("Saved comparison protocol does not match")
        return saved
    trajectory = folder / "selection_trajectory"
    rows = [json.loads((trajectory / f"epoch_{ep:02d}.json").read_text()) for ep in range(1, 31)]
    training_summary = json.loads((folder / "summary.json").read_text())
    if (training_summary["completed_epochs"] != 30
            or not training_summary["outer_test_deferred"]
            or training_summary["model_train_subjects"] != panel["fit"]
            or training_summary["checkpoint_selection_subjects"] != panel["selection"]
            or set(training_summary["excluded_assessment_subjects"]) != set(panel["assessment"])):
        raise RuntimeError("Training output does not match the locked subject partition")
    selected = select_epochs(rows)
    # Freeze both decisions before accessing inner assessment signals or labels.
    write_json(folder / "locked_selection.json", {
        "protocol_sha256": protocol_digest,
        "rules": {rule: {"epoch": row["epoch"], "validation_metrics": row["validation_metrics"]}
                  for rule, row in selected.items()},
    })
    files = [str(path) for path in sorted(args.np_data_dir.glob("*.npz"))
             if path.name[3:5] in panel["assessment"]]
    device = torch.device("cuda")
    results = {}
    cache = {}
    for rule, row in selected.items():
        epoch = row["epoch"]
        if epoch not in cache:
            checkpoint = torch.load(trajectory / f"epoch_{epoch:02d}.pt", map_location=device, weights_only=False)
            if checkpoint["metadata"]["channel_indices"] != list(channel_indices):
                raise RuntimeError("Checkpoint input channels do not match assessment channels")
            model = AttnSleep(input_channels=len(channel_indices), channel_fusion="quality-gated", channel_dropout=0.15).to(device)
            model.load_state_dict(checkpoint["model_state"])
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(checkpoint["metadata"]["class_weights"], device=device))
            records = []
            for path in files:
                dataset = NumpySleepDataset([path], list(channel_indices)).dataset
                metrics = evaluate(model, DataLoader(dataset, batch_size=128), criterion, device)
                from sklearn.metrics import confusion_matrix
                records.append({"record": Path(path).name, "subject": Path(path).name[3:5],
                                "metrics": stage_metrics(metrics),
                                "confusion_matrix": confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=range(5)).tolist()})
            from scripts.audit_attnsleep_generalization_gap import metrics_from_confusion
            pooled = metrics_from_confusion(sum(np.array(record["confusion_matrix"]) for record in records))
            pooled.update({"n1_precision": pooled["class_precision"][1], "n1_recall": pooled["class_recall"][1],
                           "n1_f1": pooled["class_f1"][1], "n3_f1": pooled["class_f1"][3]})
            cache[epoch] = {"records": records, "pooled": pooled}
            del model
        results[rule] = {"epoch": epoch, **cache[epoch]}
    deltas = {key: results["guarded_macro"]["pooled"][key] - results["accuracy"]["pooled"][key]
              for key in ("accuracy", "macro_f1", "kappa", "n1_precision", "n1_recall", "n1_f1", "n3_f1")}
    result = {"seed": seed, "panel": panel, "device": torch.cuda.get_device_name(0),
              "channel_indices": list(channel_indices),
              "protocol_sha256": protocol_digest, "rules": results, "paired_deltas": deltas}
    write_json(result_path, result)
    return result


def run_queue(gpu_id, jobs, args, digest):
    # Assessment must use the same visibility as training. Each worker is a subprocess.
    for seed, panel in jobs:
        run_dir = args.output_root / f"seed{seed}"
        folder = run_dir / f"fold_{panel['fold_id']:02d}"
        if not (folder / "summary.json").exists():
            if (folder / "history.csv").exists():
                raise RuntimeError(f"Incomplete trajectory: {folder}; refusing overwrite or non-identical RNG resume")
            command = [sys.executable, str(ROOT / "scripts/train_attnsleep_multichannel_baseline.py"),
                       "--np-data-dir", str(args.np_data_dir), "--output-dir", str(run_dir),
                       "--fold-ids", str(panel["fold_id"]), "--epochs", "30", "--batch-size", "64",
                       "--val-subjects-per-fold", "4", "--validation-strategy", "fixed-subjects",
                       "--validation-subject-ids", ",".join(panel["selection"]),
                       "--excluded-subject-ids", ",".join(panel["assessment"]),
                       "--channel-indices", "0,1,2,3", "--channel-fusion", "quality-gated",
                       "--channel-dropout", "0.15", "--weight-decay", "1e-3",
                       "--lr", "1e-3", "--lr-drop-epoch", "10", "--lr-after-drop", "1e-4",
                       "--seed", str(seed), "--seed-scope", "fold", "--device", "cuda",
                       "--resume", "never", "--defer-outer-test", "--save-selection-trajectory", "--no-aggregate"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id), "PYTHONHASHSEED": str(seed), "PYTHONUNBUFFERED": "1"}
            log_dir = args.output_root / "logs"
            log_dir.mkdir(exist_ok=True)
            with (log_dir / f"seed{seed}_fold{panel['fold_id']:02d}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--assess-job", str(seed), str(panel["fold_id"]),
                   "--np-data-dir", str(args.np_data_dir), "--output-root", str(args.output_root)]
        subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}, check=True)


def main():
    assessment_job = None
    if "--assess-job" in sys.argv:
        index = sys.argv.index("--assess-job")
        assessment_job = tuple(map(int, sys.argv[index + 1:index + 3]))
        del sys.argv[index:index + 3]
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(args)
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("Locked protocol/input inventory changed; use a new output directory")
    else:
        write_json(protocol_path, protocol)
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    print(f"Protocol locked: {protocol_path}; sha256={digest}", flush=True)
    if args.prepare_only:
        print("Prepared 15 paired trajectories; 10 distinct inner assessment subjects; no training started.")
        return
    import torch
    if not torch.cuda.is_available():
        write_json(args.output_root / "runtime_status.json", {
            "status": "gpu_unavailable", "training_started": False,
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "protocol_sha256": digest,
        })
        raise RuntimeError("GPU unavailable; this study does not permit CPU training or inference")
    if assessment_job:
        seed, fold = assessment_job
        panel = next(panel for panel in protocol["panels"] if panel["fold_id"] == fold)
        assess(args, seed, panel, args.output_root / f"seed{seed}", digest)
        return
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two GPUs required for the locked execution queues")
    write_json(args.output_root / "runtime_status.json", {
        "status": "stage1_running", "training_started": True,
        "pid": os.getpid(), "protocol_sha256": digest,
        "gpu_names": [torch.cuda.get_device_name(i) for i in (0, 1)],
    })
    jobs = [(seed, panel) for seed in protocol["seeds"] for panel in protocol["panels"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_queue, gpu, jobs[gpu::2], args, digest) for gpu in (0, 1)]
        for future in futures:
            future.result()
    results = [json.loads((args.output_root / f"seed{seed}" / f"fold_{panel['fold_id']:02d}" / "selection_comparison.json").read_text())
               for seed, panel in jobs]
    keys = list(results[0]["paired_deltas"])
    mean = {key: float(np.mean([result["paired_deltas"][key] for result in results])) for key in keys}
    seed_means = {str(seed): float(np.mean([result["paired_deltas"]["macro_f1"] for result in results if result["seed"] == seed]))
                  for seed in protocol["seeds"]}
    checks = {"macro_gain": mean["macro_f1"] >= 0.005, "seed_consistency": sum(value > 0 for value in seed_means.values()) >= 2,
              "accuracy": mean["accuracy"] >= -0.005, "n1_recall": mean["n1_recall"] >= -0.02,
              "n1_f1": mean["n1_f1"] >= 0, "n3_f1": mean["n3_f1"] >= -0.01}
    write_json(args.output_root / "summary.json", {
        "protocol_sha256": digest, "results": results, "mean_paired_deltas": mean,
        "seed_macro_deltas": seed_means, "decision_checks": checks,
        "stage2_selector": "guarded_macro" if all(checks.values()) else "accuracy",
        "interpretation": protocol["interpretation"],
    })
    write_json(args.output_root / "runtime_status.json", {"status": "stage1_complete", "protocol_sha256": digest})


if __name__ == "__main__":
    main()

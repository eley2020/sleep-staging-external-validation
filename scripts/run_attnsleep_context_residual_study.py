"""Stage 3: frozen mainline plus bounded, zero-initialized causal residual heads.

Uses the locked earlier decisions. Each head trains for 30 epochs on model-fit
subjects only. Comparison includes unchanged baseline, raw residual, fixed
context residual, and prediction-conditioned context residual. No outer tests.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue as job_queue
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_baselines/AttnSleep"))
from scripts.run_attnsleep_selection_study import select_epochs, stage_metrics, write_json

MODES = ("raw_residual", "fixed_context", "selective_context")
KEYS = ("accuracy", "macro_f1", "kappa", "n1_precision", "n1_recall", "n1_f1", "n3_f1")


class ResidualHead(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        self.project = nn.Sequential(nn.Linear(3 * dim, 64), nn.GELU(), nn.Dropout(0.2))
        self.delta = nn.Linear(64, 5)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.gate = nn.Sequential(nn.Linear(9, 8), nn.GELU(), nn.Linear(8, 1)) if mode == "selective_context" else None
        if self.gate is not None:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, windows, logits, gate_features):
        if self.mode == "raw_residual":
            windows = windows[:, -1:].expand(-1, 3, -1)
        residual = 2 * torch.tanh(self.delta(self.project(windows.flatten(1))))
        weight = torch.sigmoid(self.gate(gate_features)) if self.gate is not None else 0.5
        return logits + weight * residual


def parser():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--stage1-root", type=Path, default=ROOT / "outputs/attnsleep_selection_study_20261010")
    p.add_argument("--stage2-root", type=Path, default=ROOT / "outputs/attnsleep_emg_study_20261010")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/attnsleep_context_residual_study_20261010")
    p.add_argument("--job", nargs=2, type=int)
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def locked_protocol(args):
    stage1 = json.loads((args.stage1_root / "protocol.json").read_text())
    stage2 = json.loads((args.stage2_root / "summary.json").read_text())
    if len(stage2["results"]) != 15:
        raise RuntimeError("Stage 2 incomplete")
    return {
        "study": "bounded zero-initialized causal context residual on a frozen mainline",
        "stage2_summary_sha256": hashlib.sha256((args.stage2_root / "summary.json").read_bytes()).hexdigest(),
        "selector": stage2["selector"], "channels": stage2["stage3_channels"],
        "np_data_dir": stage1["data_dir"], "panels": stage1["panels"], "seeds": stage1["seeds"],
        "epochs": 30, "batch_size": 256, "optimizer": "AdamW", "lr": 0.0003, "weight_decay": 0.001,
        "hidden_dim": 64, "dropout": 0.2, "modes": list(MODES),
        "head_input": "per-epoch mean/std over 30 TCE tokens (160 values) plus five original logits; baseline retains its full 2400-dimensional classifier input",
        "residual": "2*tanh(zero-initialized output); fixed weight 0.5 or learned sigmoid gate",
        "gate_features": "three entropies, two L1 probability disagreements, two predicted transitions, two history masks; no true labels at inference",
        "history": "current plus two preceding valid contiguous epochs; reset per record and original-timeline gaps",
        "guard_reference": "candidate validation metrics must preserve frozen baseline accuracy/N1 recall/N3 F1; epoch0 baseline always eligible",
        "parameter_control": "raw and fixed-context heads share exactly the same head size; selective gate adds 89 parameters",
        "promotion": "selective-context mean assessment Macro-F1 >=0.005 above every control, positive delta in >=2 seeds versus every control; Accuracy >=-0.005, N1 recall >=-0.02, N1 F1 >=0, N3 F1 >=-0.01 versus frozen baseline; internal candidate only pending independent external confirmation",
        "interpretation": "Exploratory development assessment reused for research decisions; not an unbiased final test. Same backbone and preprocessing, no outer-test subjects loaded within each panel.",
    }


def original_indices(path, expected_y, summary):
    from scripts.prepare_sleepedf20_multichannel import read_labels, UNKNOWN
    name = path.stem + "-PSG.edf"
    record = next(row for row in summary["records"] if row["psg_file"] == name)
    raw_dir = Path(summary["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    y_all = read_labels(raw_dir / record["hypnogram_file"], record["n_all_epochs"])
    valid = np.flatnonzero(y_all != UNKNOWN)
    y = y_all[valid]
    sleep = np.flatnonzero(y != 0)
    edge = int(summary["w_edge_mins"]) * 2
    keep = np.arange(max(int(sleep[0]) - edge, 0), min(int(sleep[-1]) + edge, len(y) - 1) + 1)
    if not np.array_equal(y[keep], expected_y):
        raise RuntimeError(f"Original timeline does not match cached labels: {path}")
    return valid[keep]


def encode_records(model, paths, channels, device, summary):
    from scripts.train_attnsleep_multichannel_baseline import load_arrays
    records = []
    model.eval()
    for path in paths:
        x, y, *_ = load_arrays([str(path)], channels)
        parts, logits = [], []
        with torch.no_grad():
            for start in range(0, len(y), 128):
                batch = torch.as_tensor(x[start:start + 128], device=device)
                weighted = model.channel_gate(batch)
                sequence = model.tce(model.mrcnn(weighted))
                original_logits = model.fc(sequence.contiguous().flatten(1))
                compact = torch.cat((sequence.mean(1), sequence.std(1, unbiased=False), original_logits), 1)
                parts.append(compact.cpu().numpy())
                logits.append(original_logits.cpu().numpy())
        records.append({"name": path.name, "subject": path.name[3:5], "y": y,
                        "indices": original_indices(path, y, summary),
                        "features": np.concatenate(parts), "logits": np.concatenate(logits)})
    return records


def windows_for(record, mean, std):
    z = (record["features"] - mean) / std
    count, dim = z.shape
    windows = np.zeros((count, 3, dim), dtype=np.float32)
    history = np.zeros((count, 3), dtype=np.float32)
    probabilities = torch.softmax(torch.from_numpy(record["logits"]), -1).numpy()
    prob_windows = np.zeros((count, 3, 5), dtype=np.float32)
    for t in range(count):
        windows[t, 2] = z[t]
        prob_windows[t, 2] = probabilities[t]
        history[t, 2] = 1
        for age in (1, 2):
            previous = t - age
            if previous < 0 or record["indices"][t] - record["indices"][previous] != age:
                break
            windows[t, 2 - age] = z[previous]
            prob_windows[t, 2 - age] = probabilities[previous]
            history[t, 2 - age] = 1
    entropy = -(prob_windows * np.log(np.maximum(prob_windows, 1e-8))).sum(-1)
    disagreements = np.stack([np.abs(prob_windows[:, i] - prob_windows[:, 2]).sum(-1) * history[:, i] for i in (0, 1)], -1)
    changes = np.stack([(prob_windows[:, i].argmax(-1) != prob_windows[:, 2].argmax(-1)) * history[:, i] for i in (0, 1)], -1)
    gate = np.concatenate((entropy, disagreements, changes, history[:, :2]), -1).astype(np.float32)
    return windows, gate


def tensors(records, mean, std, device):
    arrays = [windows_for(record, mean, std) for record in records]
    return tuple(torch.as_tensor(np.concatenate(items), device=device) for items in (
        [array[0] for array in arrays], [record["logits"] for record in records],
        [array[1] for array in arrays], [record["y"] for record in records]))


def predict(model, data):
    windows, logits, gate, y = data
    model.eval()
    with torch.no_grad():
        return torch.cat([model(windows[i:i + 256], logits[i:i + 256], gate[i:i + 256])
                          for i in range(0, len(y), 256)]).argmax(-1).cpu().numpy()


def metrics(y, predicted):
    from scripts.evaluate_attnsleep_external_ensemble import metric_dict
    return metric_dict(np.asarray(y), np.asarray(predicted))


def choose_candidate(rows, baseline, rule):
    from scripts.run_attnsleep_selection_study import GUARDS
    reference = stage_metrics(baseline)
    eligible = [row for row in rows if all(
        stage_metrics(row["validation_metrics"])[key] >= reference[key] - tolerance
        for key, tolerance in GUARDS.items())]
    # Extra common baseline guards apply to all heads, not just the selective head.
    return select_epochs(eligible)[rule]


def run_job(args, locked, digest, seed, panel):
    from model.model import AttnSleep
    from scripts.train_attnsleep_multichannel_baseline import set_random_seed
    from scripts.audit_attnsleep_generalization_gap import metrics_from_confusion
    from sklearn.metrics import confusion_matrix

    folder = args.output_root / f"seed{seed}" / f"fold_{panel['fold_id']:02d}"
    if (folder / "comparison.json").exists():
        saved = json.loads((folder / "comparison.json").read_text())
        if saved["protocol_sha256"] != digest:
            raise RuntimeError("Context protocol mismatch")
        return
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob("*_validation.json")):
        raise RuntimeError("Partial context study exists; refusing inconsistent continuation")
    device = torch.device("cuda")
    channels = locked["channels"]
    source_root = args.stage2_root if len(channels) == 3 else args.stage1_root
    source_folder = source_root / f"seed{seed}" / f"fold_{panel['fold_id']:02d}"
    selection = json.loads((source_folder / "locked_selection.json").read_text())["rules"][locked["selector"]]
    source_path = source_folder / "selection_trajectory" / f"epoch_{selection['epoch']:02d}.pt"
    checkpoint = torch.load(source_path, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    if metadata["model_train_subjects"] != panel["fit"] or metadata["checkpoint_selection_subjects"] != panel["selection"]:
        raise RuntimeError("Source checkpoint trained on wrong subjects")
    encoder = AttnSleep(input_channels=len(channels), channel_fusion="quality-gated", channel_dropout=0.15).to(device)
    encoder.load_state_dict(checkpoint["model_state"])
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    data_dir = Path(locked["np_data_dir"])
    paths = sorted(data_dir.glob("*.npz"))
    data_summary = json.loads((data_dir / "summary.json").read_text())
    fit = encode_records(encoder, [path for path in paths if path.name[3:5] in panel["fit"]], channels, device, data_summary)
    validation = encode_records(encoder, [path for path in paths if path.name[3:5] in panel["selection"]], channels, device, data_summary)
    fit_features = np.concatenate([record["features"] for record in fit])
    mean, std = fit_features.mean(0), np.maximum(fit_features.std(0), 1e-6)
    np.savez_compressed(folder / "feature_standardizer.npz", mean=mean, std=std)
    train_data = tensors(fit, mean, std, device)
    val_data = tensors(validation, mean, std, device)
    baseline_validation = metrics(val_data[-1].cpu().numpy(), val_data[1].argmax(-1).cpu().numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(metadata["class_weights"], device=device))
    choices = {}
    parameters = {}
    for mode in MODES:
        set_random_seed(seed + panel["fold_id"])
        model = ResidualHead(fit_features.shape[1], mode).to(device)
        parameters[mode] = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
        trajectory = [{"epoch": 0, "validation_metrics": baseline_validation}]
        torch.save(model.state_dict(), folder / f"{mode}_epoch00.pt")
        for epoch in range(1, 31):
            set_random_seed(seed + panel["fold_id"] + 10000 * epoch)
            # Explicit common order, independent of mode-specific RNG draws.
            generator = torch.Generator(device=device).manual_seed(seed + panel["fold_id"] + 10000 * epoch)
            permutation = torch.randperm(len(train_data[-1]), generator=generator, device=device)
            model.train()
            for start in range(0, len(permutation), 256):
                indices = permutation[start:start + 256]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(*(value[indices] for value in train_data[:3])), train_data[3][indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            outcome = metrics(val_data[-1].cpu().numpy(), predict(model, val_data))
            trajectory.append({"epoch": epoch, "validation_metrics": outcome})
            torch.save(model.state_dict(), folder / f"{mode}_epoch{epoch:02d}.pt")
            print(seed, panel["fold_id"], mode, epoch, stage_metrics(outcome), flush=True)
        write_json(folder / f"{mode}_validation.json", trajectory)
        choices[mode] = choose_candidate(trajectory, baseline_validation, locked["selector"])
        del model
    write_json(folder / "locked_selection.json", {"protocol_sha256": digest, "choices": choices,
                                                  "source_checkpoint": str(source_path)})
    # Assessment accessed only after every head's checkpoint is locked.
    assessment = encode_records(encoder, [path for path in paths if path.name[3:5] in panel["assessment"]], channels, device, data_summary)
    del encoder
    outcomes = {}
    for mode in ("baseline", *MODES):
        model = None
        if mode != "baseline":
            model = ResidualHead(fit_features.shape[1], mode).to(device)
            model.load_state_dict(torch.load(folder / f"{mode}_epoch{choices[mode]['epoch']:02d}.pt", weights_only=True, map_location=device))
        records = []
        for record in assessment:
            data = tensors([record], mean, std, device)
            predicted = data[1].argmax(-1).cpu().numpy() if model is None else predict(model, data)
            cm = confusion_matrix(record["y"], predicted, labels=range(5))
            records.append({"record": record["name"], "subject": record["subject"], "confusion_matrix": cm.tolist()})
        pooled = metrics_from_confusion(sum(np.array(record["confusion_matrix"]) for record in records))
        pooled.update({"n1_precision": pooled["class_precision"][1], "n1_recall": pooled["class_recall"][1],
                       "n1_f1": pooled["class_f1"][1], "n3_f1": pooled["class_f1"][3]})
        outcomes[mode] = {"records": records, "pooled": pooled, "epoch": 0 if mode == "baseline" else choices[mode]["epoch"]}
    write_json(folder / "comparison.json", {"seed": seed, "panel": panel, "protocol_sha256": digest,
                                           "device": torch.cuda.get_device_name(0), "parameters": parameters, "outcomes": outcomes})


def queue(gpu, jobs, args):
    for seed, panel in jobs:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--stage1-root", str(args.stage1_root),
               "--stage2-root", str(args.stage2_root), "--output-root", str(args.output_root),
               "--job", str(seed), str(panel["fold_id"])]
        with (args.output_root / "logs" / f"seed{seed}_fold{panel['fold_id']:02d}.log").open("a") as log:
            subprocess.run(cmd, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"},
                           stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    args = parser()
    locked = locked_protocol(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    path = args.output_root / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != locked:
            raise RuntimeError("Context protocol changed")
    else:
        write_json(path, locked)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.prepare_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for all model training and inference")
    if args.job:
        seed, fold = args.job
        panel = next(panel for panel in locked["panels"] if panel["fold_id"] == fold)
        run_job(args, locked, digest, seed, panel)
        return
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two GPUs required")
    write_json(args.output_root / "runtime_status.json", {"status": "stage3_running", "pid": os.getpid()})
    jobs = [(seed, panel) for seed in locked["seeds"] for panel in locked["panels"]]
    pending = job_queue.Queue()
    for job in jobs:
        pending.put(job)
    def drain(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except job_queue.Empty:
                return
            try:
                queue(gpu, [job], args)
            finally:
                pending.task_done()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(drain, gpu) for gpu in (0, 1)]
        for future in futures:
            future.result()
    results = [json.loads((args.output_root / f"seed{seed}" / f"fold_{panel['fold_id']:02d}" / "comparison.json").read_text()) for seed, panel in jobs]
    deltas = {}
    for control in ("baseline", "raw_residual", "fixed_context"):
        mean = {key: float(np.mean([row["outcomes"]["selective_context"]["pooled"][key] - row["outcomes"][control]["pooled"][key] for row in results])) for key in KEYS}
        seed_mean = {str(seed): float(np.mean([row["outcomes"]["selective_context"]["pooled"]["macro_f1"] - row["outcomes"][control]["pooled"]["macro_f1"] for row in results if row["seed"] == seed])) for seed in locked["seeds"]}
        deltas[control] = {"mean": mean, "seed_macro_means": seed_mean}
    base = deltas["baseline"]["mean"]
    checks = {"above_all_controls": all(value["mean"]["macro_f1"] >= 0.005 for value in deltas.values()),
              "seed_consistency": all(sum(x > 0 for x in value["seed_macro_means"].values()) >= 2 for value in deltas.values()),
              "accuracy": base["accuracy"] >= -0.005, "n1_recall": base["n1_recall"] >= -0.02,
              "n1_f1": base["n1_f1"] >= 0, "n3_f1": base["n3_f1"] >= -0.01}
    write_json(args.output_root / "summary.json", {"protocol_sha256": digest, "results": results,
               "selective_minus_controls": deltas, "decision_checks": checks,
               "candidate_for_external_confirmation": all(checks.values()), "interpretation": locked["interpretation"]})
    write_json(args.output_root / "runtime_status.json", {"status": "stage3_complete", "protocol_sha256": digest})


if __name__ == "__main__":
    main()

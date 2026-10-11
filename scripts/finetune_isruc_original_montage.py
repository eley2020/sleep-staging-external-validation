"""Exploratory full-model fine-tuning on ISRUC original A2/A1 montage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.evaluate_attnsleep_external_ensemble import metric_dict
    from scripts.evaluate_attnsleep_isruc import load_record
except ModuleNotFoundError:
    from evaluate_attnsleep_external_ensemble import metric_dict
    from evaluate_attnsleep_isruc import load_record

ORIGINAL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 22, 24, 26]

def load_subjects(root: Path, subjects: list[int]):
    out = {}
    for subject in subjects:
        d = root / str(subject)
        out[str(subject)] = load_record(d / f"{subject}.rec", d / f"{subject}_1.txt", "X1", "auto", "physical-volts")[:2]
    return out

def evaluate(model, records, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x, y in records.values():
            ps.append(model(torch.from_numpy(x).to(device)).argmax(1).cpu().numpy()); ys.append(y)
    values = metric_dict(np.concatenate(ys), np.concatenate(ps))
    return {k: float(values[k]) for k in ("accuracy", "macro_f1", "weighted_f1", "kappa")}

def predict_records(model, records, device):
    model.eval(); out = {}
    with torch.no_grad():
        for subject, (x, y) in records.items():
            pred = model(torch.from_numpy(x).to(device)).argmax(1).cpu().numpy().astype(np.int8)
            out[subject] = (y.astype(np.int8), pred)
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isruc-root", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, default=Path("outputs/attnsleep_multichannel_baseline_train/full20_quality_gate_4ch_emg_30epoch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calibration-subjects", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-list", default="", help="comma-separated explicit calibration subject IDs")
    parser.add_argument("--held-out-list", default="", help="comma-separated explicit held-out subject IDs")
    parser.add_argument("--source-fold", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    calibration = sorted([int(x) for x in args.calibration_list.split(",") if x.strip()]) if args.calibration_list else sorted(rng.choice(ORIGINAL, args.calibration_subjects, replace=False).tolist())
    if not calibration or not set(calibration).issubset(set(ORIGINAL)):
        raise ValueError("Invalid calibration-list for original montage subjects")
    held_out = sorted([int(x) for x in args.held_out_list.split(",") if x.strip()]) if args.held_out_list else sorted(set(ORIGINAL) - set(calibration))
    if not held_out or not set(held_out).issubset(set(ORIGINAL)) or set(held_out) & set(calibration):
        raise ValueError("Invalid or overlapping held-out subject list")
    data = load_subjects(args.isruc_root, ORIGINAL)
    sys.path.insert(0, str(Path("external_baselines/AttnSleep").resolve()))
    from model.model import AttnSleep
    checkpoint = torch.load(args.source_run_dir / f"fold_{args.source_fold:02d}" / "checkpoint_best.pt", map_location=device, weights_only=False)
    model = AttnSleep(input_channels=4, channel_fusion="quality-gated", channel_dropout=0.15).to(device)
    model.load_state_dict(checkpoint["model_state"])
    heldout_data = {str(s): data[str(s)] for s in held_out}
    before = evaluate(model, heldout_data, device)
    frozen_predictions = predict_records(model, heldout_data, device)
    x = np.concatenate([data[str(s)][0] for s in calibration]); y = np.concatenate([data[str(s)][1] for s in calibration])
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long()), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4); criterion = torch.nn.CrossEntropyLoss(); losses=[]
    for epoch in range(args.epochs):
        model.train(); total=0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(); loss=criterion(model(xb), yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); total += float(loss.detach().cpu()) * len(xb)
        losses.append(total / len(loader.dataset))
    after = evaluate(model, heldout_data, device)
    finetuned_predictions = predict_records(model, heldout_data, device)
    for subject, (y_true, y_pred) in frozen_predictions.items():
        np.savez_compressed(args.output.parent / f"{args.output.stem}_subject_{subject}.npz", y_true=y_true, frozen_y_pred=y_pred, finetuned_y_pred=finetuned_predictions[subject][1])
    payload = {"protocol": f"Exploratory full-model fine-tuning on ISRUC original A2/A1 montage; fixed subject-disjoint calibration/test split; initialized from source fold {args.source_fold:02d}", "seed": args.seed, "source_fold": args.source_fold, "calibration_subjects": calibration, "held_out_subjects": held_out, "epochs": args.epochs, "learning_rate": args.lr, "device": str(device), "train_loss": losses, "frozen_held_out": before, "finetuned_held_out": after, "delta": {k: after[k]-before[k] for k in before}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8"); print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()

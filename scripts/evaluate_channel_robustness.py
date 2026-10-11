"""Evaluate trained four-channel AttnSleep models under channel corruption.

No parameters are updated.  The same held-out test records and checkpoints are
used for early-fusion and quality-gated models; only the input corruption is
changed at inference time.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--np-data-dir", type=Path, default=Path("data/sleepedf20_multichannel_npz/eeg2_eog_emg_physical_volts"))
    p.add_argument("--attnsleep-dir", type=Path, default=Path("external_baselines/AttnSleep"))
    p.add_argument("--quality-dir", type=Path, default=Path("outputs/attnsleep_multichannel_baseline_train/full20_quality_gate_4ch_emg_30epoch"))
    p.add_argument("--early-dir", type=Path, default=Path("outputs/attnsleep_multichannel_baseline_train/full20_early_fusion_4ch_emg_30epoch"))
    p.add_argument("--output", type=Path, default=Path("outputs/attnsleep_multichannel_baseline_train/channel_robustness_stress_test.json"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def main():
    a = args()
    sys.path.insert(0, str(a.attnsleep_dir.resolve()))
    import torch
    from model.model import AttnSleep
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_attnsleep_multichannel_baseline import load_arrays, forward_model
    from utils.util import load_folds_data

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else a.device if a.device != "auto" else "cpu")
    folds = load_folds_data(str(a.np_data_dir.resolve()), 20)
    model_dirs = {"quality_gated": a.quality_dir, "early_fusion": a.early_dir}
    scenarios = ["clean"] + [f"zero_ch{c}" for c in range(4)] + [f"noise3x_ch{c}" for c in range(4)]
    rows = []
    pooled = {}
    for model_name, model_dir in model_dirs.items():
        for scenario in scenarios:
            all_y, all_p = [], []
            for fold_id in range(20):
                fold_dir = model_dir / f"fold_{fold_id:02d}"
                ck = torch.load(fold_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
                meta = json.loads((fold_dir / "metadata.json").read_text())
                fusion = meta.get("channel_fusion", "early")
                model = AttnSleep(input_channels=4, channel_fusion=fusion, channel_dropout=float(meta.get("channel_dropout") or 0.0)).to(device)
                model.load_state_dict(ck["model_state"])
                model.eval()
                test_files = list(folds[fold_id][1])
                x, y, prior, _, _ = load_arrays(test_files, None, None)
                x = x.astype(np.float32, copy=True)
                if scenario != "clean":
                    kind, chs = scenario.split("_ch")
                    ch = int(chs)
                    rng = np.random.default_rng(10000 + fold_id * 17 + ch)
                    if kind == "zero":
                        x[:, ch, :] = 0.0
                    else:
                        # Replace one channel by strong broadband noise whose
                        # standard deviation is three times that channel's
                        # clean global standard deviation.
                        sd = float(np.std(x[:, ch, :]))
                        x[:, ch, :] = rng.normal(0.0, max(sd * 3.0, 1e-8), size=x[:, ch, :].shape).astype(np.float32)
                preds = []
                with torch.no_grad():
                    for start in range(0, len(y), a.batch_size):
                        xb = torch.from_numpy(x[start:start+a.batch_size]).to(device)
                        pb = torch.from_numpy(prior[start:start+a.batch_size]).float().to(device)
                        out = forward_model(model, xb, pb)
                        preds.append(out.argmax(1).cpu().numpy())
                p = np.concatenate(preds)
                all_y.append(y); all_p.append(p)
                del model
            yt, yp = np.concatenate(all_y), np.concatenate(all_p)
            from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
            report = classification_report(yt, yp, labels=list(range(5)), target_names=["W","N1","N2","N3","REM"], output_dict=True, zero_division=0)
            rec = {"model": model_name, "scenario": scenario, "accuracy": float(accuracy_score(yt, yp)), "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)), "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)), "kappa": float(cohen_kappa_score(yt, yp)), "n_epochs": int(len(yt)), "report": report}
            rows.append(rec); pooled[f"{model_name}__{scenario}"] = rec
            print(model_name, scenario, rec["accuracy"], rec["macro_f1"], flush=True)
    clean = {(r["model"]): r for r in rows if r["scenario"] == "clean"}
    for r in rows:
        if r["scenario"] != "clean":
            c = clean[r["model"]]
            r["accuracy_drop"] = r["accuracy"] - c["accuracy"]
            r["macro_f1_drop"] = r["macro_f1"] - c["macro_f1"]
            r["kappa_drop"] = r["kappa"] - c["kappa"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"protocol": "Frozen checkpoints; same Sleep-EDF-20 20-fold test subjects; zeroed or 3x-SD broadband noise on one channel at a time.", "rows": rows}, indent=2))
    with a.output.with_suffix(".csv").open("w", newline="") as f:
        fields = ["model","scenario","accuracy","macro_f1","weighted_f1","kappa","accuracy_drop","macro_f1_drop","kappa_drop"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k) for k in fields} for r in rows)


if __name__ == "__main__":
    main()

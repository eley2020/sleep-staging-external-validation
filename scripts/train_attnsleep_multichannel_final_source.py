"""Train the deployable AttnSleep source model on all Sleep-EDF-20 records.

The epoch count and hyperparameters are fixed from the completed source-CV
experiment.  This entry point deliberately has no target-domain inputs and no
validation-based checkpoint selection: ``checkpoint_final.pt`` is epoch N.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from scripts.train_attnsleep_multichannel_baseline import (
        NumpySleepDataset,
        build_criterion,
        forward_model,
        parse_channel_indices,
        set_random_seed,
    )
except ModuleNotFoundError:
    from train_attnsleep_multichannel_baseline import (
        NumpySleepDataset,
        build_criterion,
        forward_model,
        parse_channel_indices,
        set_random_seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--np-data-dir",
        type=Path,
        default=Path("data/sleepedf20_multichannel_npz/eeg2_eog_emg_physical_volts"),
    )
    parser.add_argument("--attnsleep-dir", type=Path, default=Path("external_baselines/AttnSleep"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/attnsleep_multichannel_final_source/quality_gate_4ch_30epoch"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--channel-indices", default="all")
    parser.add_argument("--channel-fusion", choices=["quality-gated"], default="quality-gated")
    parser.add_argument("--channel-dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-drop-epoch", type=int, default=10)
    parser.add_argument("--lr-after-drop", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--loss-type", choices=["cross-entropy"], default="cross-entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--n1-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--supcon-projection-dim", type=int, default=128)
    parser.add_argument("--supcon-projection-hidden", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", choices=["auto", "never"], default="auto")
    return parser.parse_args()


def resolve_device(value: str):
    import torch

    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.channel_dropout < 0.0 or args.channel_dropout >= 1.0:
        raise ValueError("--channel-dropout must be in [0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)

    sys.path.insert(0, str(args.attnsleep_dir.resolve()))
    import torch
    from model.model import AttnSleep
    from utils.util import calc_class_weight

    device = resolve_device(args.device)
    source_files = sorted(args.np_data_dir.glob("*.npz"))
    if len(source_files) != 39:
        raise ValueError(f"Expected exactly 39 Sleep-Cassette NPZ files, got {len(source_files)}")
    source_subjects = sorted({path.stem[3:5] for path in source_files})
    if len(source_subjects) != 20:
        raise ValueError(f"Expected 20 source subjects, got {len(source_subjects)}")
    for path in source_files:
        with np.load(path, allow_pickle=True) as record:
            normalization = str(record["normalization"].item())
            if normalization != "physical-volts":
                raise ValueError(f"Expected physical-volts normalization in {path}, got {normalization}")

    channel_indices = parse_channel_indices(args.channel_indices)
    dataset = NumpySleepDataset(
        [str(path) for path in source_files],
        channel_indices,
        quality_stats=None,
    ).dataset
    if dataset.x_data.shape[1:] != (4, 3000):
        raise ValueError(f"Expected 4x3000 input, got {tuple(dataset.x_data.shape[1:])}")
    counts = np.bincount(dataset.y_data.detach().cpu().numpy(), minlength=5).astype(int).tolist()
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    weights = torch.tensor(calc_class_weight(counts), dtype=torch.float32, device=device)
    if args.n1_loss_multiplier != 1.0:
        weights[1] *= float(args.n1_loss_multiplier)
    model = AttnSleep(
        input_channels=int(dataset.x_data.shape[1]),
        channel_fusion=args.channel_fusion,
        channel_dropout=args.channel_dropout,
    ).to(device)
    criterion = build_criterion(args, weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    metadata = {
        "baseline": "AttnSleep-multichannel-quality-gated-final-source",
        "dataset": "Sleep-EDF-20, all source records only",
        "source_np_data_dir": str(args.np_data_dir),
        "source_records": len(source_files),
        "source_subjects": source_subjects,
        "target_data_used": False,
        "checkpoint_policy": "fixed final epoch; no validation or target-domain checkpoint selection",
        "channels": dataset.channels,
        "channel_indices": channel_indices if channel_indices is not None else "all",
        "input_channels": int(dataset.x_data.shape[1]),
        "channel_fusion": args.channel_fusion,
        "channel_dropout": args.channel_dropout,
        "class_weight_counts_all_source": counts,
        "class_weights": weights.detach().cpu().tolist(),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "lr_drop_epoch": args.lr_drop_epoch,
        "lr_after_drop": args.lr_after_drop,
        "weight_decay": args.weight_decay,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(args.output_dir / "metadata.json", metadata)

    last_checkpoint = args.output_dir / "checkpoint_last.pt"
    final_checkpoint = args.output_dir / "checkpoint_final.pt"
    history_path = args.output_dir / "history.csv"
    start_epoch = 1
    if args.resume == "auto" and last_checkpoint.exists():
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("metadata") != metadata:
            raise RuntimeError("Existing checkpoint metadata does not match this final-source run")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming final-source training at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.lr_drop_epoch + 1:
            for group in optimizer.param_groups:
                group["lr"] = args.lr_after_drop
        model.train()
        t0 = time.time()
        losses: list[float] = []
        correct = 0
        total = 0
        for x, y, quality_prior in loader:
            optimizer.zero_grad(set_to_none=True)
            x = x.float().to(device)
            y = y.to(device)
            quality_prior = quality_prior.float().to(device)
            logits = forward_model(model, x, quality_prior)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_accuracy": correct / total,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(time.time() - t0, 3),
        }
        write_header = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
        }
        torch.save(checkpoint, last_checkpoint)
        if epoch == args.epochs:
            torch.save(checkpoint, final_checkpoint)
        print(
            f"final-source epoch {epoch}/{args.epochs} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} sec={row['elapsed_sec']}"
        )

    if not final_checkpoint.exists():
        raise RuntimeError("Final-source checkpoint was not created")


if __name__ == "__main__":
    main()

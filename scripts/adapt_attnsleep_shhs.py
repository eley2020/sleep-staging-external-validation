"""Strict subject-wise SHHS adaptation from the fixed all-source model."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_attnsleep_external_ensemble import metric_dict
    from scripts.evaluate_attnsleep_shhs_ensemble import load_record, subject_id, xml_for
    from scripts.train_attnsleep_multichannel_baseline import NumpySleepDataset, build_criterion, forward_model, set_random_seed
except ModuleNotFoundError:
    from evaluate_attnsleep_external_ensemble import metric_dict
    from evaluate_attnsleep_shhs_ensemble import load_record, subject_id, xml_for
    from train_attnsleep_multichannel_baseline import NumpySleepDataset, build_criterion, forward_model, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path("outputs/attnsleep_multichannel_final_source/quality_gate_4ch_30epoch/checkpoint_final.pt"),
    )
    parser.add_argument(
        "--source-np-data-dir",
        type=Path,
        default=Path("data/sleepedf20_multichannel_npz/eeg2_eog_emg_physical_volts"),
    )
    parser.add_argument("--target-cache-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--shhs-root", type=Path, default=Path("/path/to/datasets/SHHS/polysomnography"))
    parser.add_argument("--cohort", choices=["shhs1", "shhs2"], default="shhs1")
    parser.add_argument("--attnsleep-dir", type=Path, default=Path("external_baselines/AttnSleep"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["head", "tce-head-gate", "full"], required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--context-lr", type=float, default=3e-5)
    parser.add_argument("--full-lr", type=float, default=2e-5)
    parser.add_argument("--full-mrcnn-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--source-replay-weight", type=float, default=0.2)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--w-edge-mins", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_device(value: str):
    import torch

    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value)


def cached_subject(path: Path) -> str:
    with np.load(path, allow_pickle=True) as data:
        return str(data["subject"].item())


def validate_cache(paths: list[Path], label: str) -> None:
    if not paths:
        raise ValueError(f"No {label} cached SHHS records")
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            if tuple(data["x"].shape[1:]) != (4, 3000):
                raise ValueError(f"Incompatible cached shape in {path}")
            if str(data["normalization"].item()) != "physical-volts":
                raise ValueError(f"Incompatible normalization in {path}")


def make_loader(paths: list[Path], batch_size: int, shuffle: bool):
    import torch

    dataset = NumpySleepDataset([str(path) for path in paths], None, quality_stats=None).dataset
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0), dataset


def configure_mode(model, mode: str, args: argparse.Namespace):
    import torch

    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "head":
        modules, lr = [model.fc], args.head_lr
    elif mode == "tce-head-gate":
        modules, lr = [model.tce, model.fc, model.channel_gate], args.context_lr
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
        mrcnn = list(model.mrcnn.parameters())
        other = [parameter for name, parameter in model.named_parameters() if not name.startswith("mrcnn.")]
        return torch.optim.Adam(
            [{"params": mrcnn, "lr": args.full_mrcnn_lr}, {"params": other, "lr": args.full_lr}],
            weight_decay=args.weight_decay, amsgrad=True,
        )
    for module in modules:
        if module is None:
            raise RuntimeError(f"Mode {mode} requires a quality-gated model")
        for parameter in module.parameters():
            parameter.requires_grad = True
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=args.weight_decay, amsgrad=True)


def predict_cached_records(model, paths: list[Path], batch_size: int, device) -> list[dict]:
    import torch

    records = []
    model.eval()
    with torch.no_grad():
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                x = np.asarray(data["x"], dtype=np.float32)
                y_true = np.asarray(data["y"], dtype=np.int64)
                subject = str(data["subject"].item())
            predicted = []
            for start in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[start : start + batch_size]).to(device)
                predicted.append(model(xb).argmax(dim=1).cpu().numpy())
            records.append({"record": path.stem, "subject": subject, "y_true": y_true, "y_pred": np.concatenate(predicted)})
    return records


def predict_raw_records(model, subjects: set[str], args: argparse.Namespace, device) -> list[dict]:
    import torch

    edf_dir = args.shhs_root / "edfs" / args.cohort
    ann_dir = args.shhs_root / "annotations-events-nsrr" / args.cohort
    pairs = [
        (edf, xml_for(edf, ann_dir))
        for edf in sorted(edf_dir.glob("*.edf"))
        if subject_id(edf.stem) in subjects and xml_for(edf, ann_dir).exists()
    ]
    if len(pairs) != len(subjects):
        raise RuntimeError(f"Expected one paired EDF/XML per test subject, got {len(pairs)} pairs for {len(subjects)} subjects")
    records = []
    model.eval()
    with torch.no_grad():
        for number, (edf, xml) in enumerate(pairs, 1):
            x, y_true, _ = load_record(edf, xml, args.w_edge_mins)
            predicted = []
            for start in range(0, len(x), args.eval_batch_size):
                xb = torch.from_numpy(x[start : start + args.eval_batch_size]).to(device)
                predicted.append(model(xb).argmax(dim=1).cpu().numpy())
            records.append({"record": edf.stem, "subject": subject_id(edf.stem), "y_true": y_true, "y_pred": np.concatenate(predicted)})
            if number % 25 == 0 or number == len(pairs):
                print(f"test {number}/{len(pairs)} {edf.stem}", flush=True)
    return records


def summarize(records: list[dict], samples: int, seed: int) -> dict:
    y_true = np.concatenate([record["y_true"] for record in records])
    y_pred = np.concatenate([record["y_pred"] for record in records])
    metrics = metric_dict(y_true, y_pred)
    metrics["subject_bootstrap_95ci"] = bootstrap_ci_from_confusions(records, samples, seed)
    return metrics


def scalar_metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    total = float(confusion.sum())
    diagonal = np.diag(confusion).astype(np.float64)
    true_counts = confusion.sum(axis=1).astype(np.float64)
    predicted_counts = confusion.sum(axis=0).astype(np.float64)
    precision = diagonal / np.maximum(predicted_counts, 1.0)
    recall = diagonal / np.maximum(true_counts, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    accuracy = float(diagonal.sum() / total)
    expected = float(np.dot(true_counts, predicted_counts) / total**2)
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, true_counts) / total),
        "kappa": float((accuracy - expected) / max(1.0 - expected, 1e-12)),
    }


def bootstrap_ci_from_confusions(records: list[dict], samples: int, seed: int) -> dict[str, list[float]]:
    if samples <= 0:
        return {}
    matrices = []
    for record in records:
        matrix = np.zeros((5, 5), dtype=np.int64)
        np.add.at(matrix, (record["y_true"], record["y_pred"]), 1)
        matrices.append(matrix)
    matrices = np.stack(matrices)
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in ("accuracy", "macro_f1", "weighted_f1", "kappa")}
    for _ in range(samples):
        values = scalar_metrics_from_confusion(matrices[rng.integers(0, len(matrices), size=len(matrices))].sum(axis=0))
        for key, value in values.items():
            draws[key].append(value)
    return {key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for key, values in draws.items()}


def save_predictions(path: Path, records: list[dict]) -> None:
    offsets, y_true, y_pred, start = [], [], [], 0
    for record in records:
        end = start + len(record["y_true"])
        offsets.append([record["record"], record["subject"], start, end])
        y_true.append(record["y_true"].astype(np.int8))
        y_pred.append(record["y_pred"].astype(np.int8))
        start = end
    np.savez_compressed(path, y_true=np.concatenate(y_true), y_pred=np.concatenate(y_pred), spans=np.asarray(offsets, dtype=object))


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Epoch and batch sizes must be positive")
    if min(args.source_replay_weight, args.distillation_weight) < 0 or args.distillation_temperature <= 0:
        raise ValueError("Invalid replay/distillation settings")
    split_payload = json.loads(args.split_file.read_text(encoding="utf-8"))
    split = split_payload["split"]
    seed = int(split_payload["seed"] if args.seed is None else args.seed)
    set_random_seed(seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as functional

    device = resolve_device(args.device)
    sys.path.insert(0, str(args.attnsleep_dir.resolve()))
    from model.model import AttnSleep

    cached_paths = sorted(args.target_cache_dir.glob("*.npz"))
    train_set, val_set = set(split["train_subjects"]), set(split["validation_subjects"])
    train_paths = [path for path in cached_paths if cached_subject(path) in train_set]
    val_paths = [path for path in cached_paths if cached_subject(path) in val_set]
    validate_cache(train_paths, "training")
    validate_cache(val_paths, "validation")
    if {cached_subject(path) for path in train_paths} != train_set or {cached_subject(path) for path in val_paths} != val_set:
        raise RuntimeError("Target cache does not exactly match the requested train/validation subjects")

    source_paths = sorted(args.source_np_data_dir.glob("*.npz"))
    if len(source_paths) != 39:
        raise ValueError(f"Expected 39 source records, got {len(source_paths)}")
    target_loader, target_data = make_loader(train_paths, args.batch_size, True)
    source_loader, source_data = make_loader(source_paths, args.batch_size, True)
    target_counts = np.bincount(target_data.y_data.numpy(), minlength=5).astype(int)
    source_counts = np.bincount(source_data.y_data.numpy(), minlength=5).astype(int)
    target_weights = torch.tensor(1.0 / np.maximum(target_counts, 1), dtype=torch.float32, device=device)
    target_weights /= target_weights.mean()
    source_weights = torch.tensor(1.0 / np.maximum(source_counts, 1), dtype=torch.float32, device=device)
    source_weights /= source_weights.mean()
    criterion_args = argparse.Namespace(loss_type="cross-entropy", label_smoothing=0.0, focal_gamma=2.0, n1_loss_multiplier=1.0, supcon_weight=0.0)
    target_criterion = build_criterion(criterion_args, target_weights)
    source_criterion = build_criterion(criterion_args, source_weights)

    checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if metadata.get("channel_fusion") != "quality-gated" or metadata.get("input_channels") != 4:
        raise ValueError("Expected the fixed four-channel quality-gated all-source checkpoint")
    model = AttnSleep(input_channels=4, channel_fusion="quality-gated", channel_dropout=0.15).to(device)
    model.load_state_dict(checkpoint["model_state"])
    teacher = AttnSleep(input_channels=4, channel_fusion="quality-gated", channel_dropout=0.15).to(device)
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = configure_mode(model, args.mode, args)

    run_metadata = {
        "protocol": "SHHS1 subject-disjoint labeled small-sample adaptation; validation-only checkpoint selection; test subjects unused until final evaluation",
        "mode": args.mode, "split_file": str(args.split_file), "split": split, "split_seed": seed,
        "source_checkpoint": str(args.source_checkpoint), "cohort": args.cohort,
        "n_target_train_records": len(train_paths), "n_target_validation_records": len(val_paths),
        "n_target_test_records": len(split["test_subjects"]), "n_target_train_epochs": int(len(target_data)),
        "target_class_counts": target_counts.tolist(), "source_class_counts": source_counts.tolist(),
        "source_replay_weight": args.source_replay_weight, "distillation_weight": args.distillation_weight,
        "distillation_temperature": args.distillation_temperature, "epochs": args.epochs,
        "batch_size": args.batch_size, "device": str(device),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    best_key, source_iterator = (-1.0, -1.0), iter(source_loader)
    history_path, best_path = args.output_dir / "history.csv", args.output_dir / "checkpoint_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, replay_losses, distill_losses = [], [], []
        for x_target, y_target, target_prior in target_loader:
            try:
                x_source, y_source, source_prior = next(source_iterator)
            except StopIteration:
                source_iterator = iter(source_loader)
                x_source, y_source, source_prior = next(source_iterator)
            x_target, y_target, target_prior = x_target.float().to(device), y_target.to(device), target_prior.float().to(device)
            x_source, y_source, source_prior = x_source.float().to(device), y_source.to(device), source_prior.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            target_loss = target_criterion(forward_model(model, x_target, target_prior), y_target)
            source_logits = forward_model(model, x_source, source_prior)
            replay_loss = source_criterion(source_logits, y_source)
            with torch.no_grad():
                teacher_logits = teacher(x_source)
            temperature = args.distillation_temperature
            distill_loss = functional.kl_div(functional.log_softmax(source_logits / temperature, dim=1), functional.softmax(teacher_logits / temperature, dim=1), reduction="batchmean") * temperature ** 2
            loss = target_loss + args.source_replay_weight * replay_loss + args.distillation_weight * distill_loss
            loss.backward()
            optimizer.step()
            losses.append(float(target_loss.detach().cpu()))
            replay_losses.append(float(replay_loss.detach().cpu()))
            distill_losses.append(float(distill_loss.detach().cpu()))
        validation = summarize(predict_cached_records(model, val_paths, args.eval_batch_size, device), 0, seed)
        row = {
            "epoch": epoch, "target_loss": float(np.mean(losses)), "source_replay_loss": float(np.mean(replay_losses)),
            "distillation_loss": float(np.mean(distill_losses)), "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"], "validation_kappa": validation["kappa"], "lr": optimizer.param_groups[0]["lr"],
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        key = (float(validation["macro_f1"]), float(validation["accuracy"]))
        if key > best_key:
            best_key = key
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metadata": run_metadata, "validation": validation}, best_path)
        print(f"{args.mode} epoch {epoch}/{args.epochs} val_macro_f1={validation['macro_f1']:.4f} val_acc={validation['accuracy']:.4f}", flush=True)

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_subjects = set(split["test_subjects"])
    adapted_records = predict_raw_records(model, test_subjects, args, device)
    frozen = AttnSleep(input_channels=4, channel_fusion="quality-gated", channel_dropout=0.15).to(device)
    frozen.load_state_dict(checkpoint["model_state"])
    frozen_records = predict_raw_records(frozen, test_subjects, args, device)
    if [r["record"] for r in adapted_records] != [r["record"] for r in frozen_records]:
        raise RuntimeError("Frozen and adapted test records differ")
    payload = {
        "metadata": run_metadata, "selected_epoch": best["epoch"], "validation_at_selected_epoch": best["validation"],
        "frozen_source_held_out_test_metrics": summarize(frozen_records, args.bootstrap_samples, seed),
        "held_out_test_metrics": summarize(adapted_records, args.bootstrap_samples, seed),
    }
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_predictions(args.output_dir / "frozen_test_predictions.npz", frozen_records)
    save_predictions(args.output_dir / "adapted_test_predictions.npz", adapted_records)
    with (args.output_dir / "record_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{"record": r["record"], "subject": r["subject"], "n_epochs": len(r["y_true"]), **{k: v for k, v in metric_dict(r["y_true"], r["y_pred"]).items() if k != "classification_report"}} for r in adapted_records]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"selected_epoch": best["epoch"], "frozen": payload["frozen_source_held_out_test_metrics"], "adapted": payload["held_out_test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()

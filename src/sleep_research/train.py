from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from .config import DEFAULT_CACHE_DIR, DEFAULT_RUN_DIR, INDEX_TO_STAGE, ReproductionConfig
from .models import ContextBiLSTM, PaperCNN, require_torch
from .preprocessing import build_npz_dataset, load_npz_dataset, make_context_windows

REFERENCE_BATCH_COUNTS = (100, 25, 100, 25, 50)
CLASS_WEIGHT_CHOICES = ("none", "balanced")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    require_torch()
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=5).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return weights.astype(np.float32)


def make_criterion(y: np.ndarray, device: str, class_weights: str):
    require_torch()
    import torch

    if class_weights == "balanced":
        weights = torch.from_numpy(balanced_weights(y)).to(device)
        return torch.nn.CrossEntropyLoss(weight=weights)
    if class_weights == "none":
        return torch.nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported class_weights mode: {class_weights}")


def class_distribution(values: np.ndarray, n_classes: int) -> dict[str, int]:
    counts = np.bincount(values.astype(np.int64), minlength=n_classes)
    return {INDEX_TO_STAGE[i]: int(counts[i]) for i in range(n_classes)}


def reference_decay_step_count(train_size: int, batch_size: int = sum(REFERENCE_BATCH_COUNTS)) -> int:
    """Match the TF1 exponential_decay cadence: decay once per effective data pass."""
    return max(1, train_size // max(1, batch_size))


class BalancedBatchSampler:
    """Reference-code style class-balanced mini-batches.

    The original TF1 code sampled classes with fixed counts:
    W/N1/N2/N3-4/REM = 100/25/100/25/50. For small smoke tests, classes with fewer
    examples are sampled with replacement so the training loop remains usable.
    """

    def __init__(
        self,
        labels: np.ndarray,
        class_counts: tuple[int, ...] = REFERENCE_BATCH_COUNTS,
        seed: int = 42,
    ) -> None:
        self.labels = np.asarray(labels)
        self.class_counts = class_counts
        self.rng = np.random.default_rng(seed)
        self.class_indices = {
            class_id: np.flatnonzero(self.labels == class_id)
            for class_id in range(len(class_counts))
        }
        self.batch_size = int(sum(class_counts))

    def sample(self) -> np.ndarray:
        sampled: list[np.ndarray] = []
        for class_id, count in enumerate(self.class_counts):
            candidates = self.class_indices[class_id]
            if len(candidates) == 0 or count <= 0:
                continue
            replace = len(candidates) < count
            sampled.append(self.rng.choice(candidates, size=count, replace=replace))
        if not sampled:
            raise ValueError("Cannot sample a balanced batch because no class indices are available.")
        batch = np.concatenate(sampled)
        self.rng.shuffle(batch)
        return batch


def train_epoch(model, loader, optimizer, criterion, device: str) -> float:
    import torch

    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(yb)
        n += len(yb)
    return total / max(n, 1)


def train_steps(
    model,
    x: np.ndarray,
    y: np.ndarray,
    optimizer,
    criterion,
    device: str,
    steps: int,
    seed: int,
    class_counts: tuple[int, ...] = REFERENCE_BATCH_COUNTS,
    lr_scheduler=None,
) -> list[float]:
    require_torch()
    import torch

    model.train()
    sampler = BalancedBatchSampler(y, class_counts=class_counts, seed=seed)
    losses: list[float] = []
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).long()
    for _ in range(steps):
        batch_idx = sampler.sample()
        xb = x_tensor[batch_idx].to(device)
        yb = y_tensor[batch_idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        losses.append(float(loss.item()))
    return losses


def run_validation(
    model,
    loader,
    device: str,
    cfg: ReproductionConfig,
    held_out_record: str,
) -> dict[str, object]:
    pred, true, _ = predict(model, loader, device)
    return evaluate_predictions(true, pred, cfg, held_out_record)


def evaluate_sample(
    model,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    cfg: ReproductionConfig,
    batch_size: int,
    max_samples: int = 2048,
) -> dict[str, object]:
    if len(y) == 0:
        return {
            "sample_size": 0,
            "accuracy": None,
            "true_distribution": class_distribution(y, cfg.n_classes),
            "pred_distribution": class_distribution(y, cfg.n_classes),
        }
    sample_size = min(max_samples, len(y))
    indices = np.linspace(0, len(y) - 1, sample_size, dtype=np.int64)
    loader = make_loader(x[indices], y[indices], batch_size, shuffle=False)
    pred, true, _ = predict(model, loader, device)
    return {
        "sample_size": int(sample_size),
        "accuracy": float(accuracy_score(true, pred)),
        "true_distribution": class_distribution(true, cfg.n_classes),
        "pred_distribution": class_distribution(pred, cfg.n_classes),
    }


def save_checkpoint(path: Path, model, optimizer, step: int, summary: dict[str, object]) -> None:
    require_torch()
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "summary": summary,
        },
        path,
    )


def load_model_checkpoint(path: Path, model, device: str) -> dict[str, object]:
    require_torch()
    import torch

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train_steps_with_validation(
    model,
    x: np.ndarray,
    y: np.ndarray,
    optimizer,
    criterion,
    device: str,
    steps: int,
    seed: int,
    validation_loader,
    cfg: ReproductionConfig,
    held_out_record: str,
    checkpoint_path: Path,
    batch_size: int,
    class_counts: tuple[int, ...] = REFERENCE_BATCH_COUNTS,
    lr_scheduler=None,
    eval_every: int = 100,
) -> tuple[list[float], list[dict[str, object]], dict[str, object]]:
    require_torch()
    import torch

    model.train()
    sampler = BalancedBatchSampler(y, class_counts=class_counts, seed=seed)
    losses: list[float] = []
    history: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_accuracy = -1.0
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).long()
    eval_every = max(1, eval_every)

    for step in range(1, steps + 1):
        batch_idx = sampler.sample()
        xb = x_tensor[batch_idx].to(device)
        yb = y_tensor[batch_idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        losses.append(float(loss.item()))

        if step == 1 or step == steps or step % eval_every == 0:
            summary = run_validation(model, validation_loader, device, cfg, held_out_record)
            summary["step"] = step
            summary["train_loss"] = float(loss.item())
            summary["lr"] = float(optimizer.param_groups[0]["lr"])
            summary["prediction_distribution"] = {
                INDEX_TO_STAGE[i]: int(np.asarray(summary["confusion_matrix"])[:, i].sum())
                for i in range(cfg.n_classes)
            }
            summary["target_distribution"] = {
                INDEX_TO_STAGE[i]: int(np.asarray(summary["confusion_matrix"])[i, :].sum())
                for i in range(cfg.n_classes)
            }
            summary["train_sample"] = evaluate_sample(
                model,
                x,
                y,
                device,
                cfg,
                batch_size=batch_size,
            )
            history.append(summary)
            if float(summary["accuracy"]) > best_accuracy:
                best_accuracy = float(summary["accuracy"])
                best_summary = summary
                save_checkpoint(checkpoint_path, model, optimizer, step, summary)
        model.train()

    if best_summary is None:
        best_summary = run_validation(model, validation_loader, device, cfg, held_out_record)
        best_summary["step"] = 0
        save_checkpoint(checkpoint_path, model, optimizer, 0, best_summary)
    return losses, history, best_summary


def predict(model, loader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    preds: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(prob)
            preds.append(prob.argmax(axis=1))
            labels.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(probs)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    require_torch()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def flatten_metrics(summary: dict[str, object], prefix: str = "") -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in summary.items():
        name = f"{prefix}{key}" if prefix else key
        if key == "classification_report" and isinstance(value, dict):
            for label, metrics in value.items():
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        row[f"{prefix}{label}_{metric_name}"] = metric_value
                else:
                    row[f"{prefix}{label}"] = metrics
        elif key != "confusion_matrix" and not isinstance(value, (list, dict)):
            row[name] = value
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_history_row(
    summary: dict[str, object],
    fold: int,
    held_out_record: str,
    stage: str,
) -> dict[str, object]:
    row = {
        "fold": fold,
        "held_out_record": held_out_record,
        "stage": stage,
        "step": summary.get("step"),
        "accuracy": summary.get("accuracy"),
        "train_loss": summary.get("train_loss"),
        "lr": summary.get("lr"),
    }
    report = summary.get("classification_report")
    if isinstance(report, dict):
        macro_avg = report.get("macro avg")
        weighted_avg = report.get("weighted avg")
        n1_metrics = report.get("N1")
        rem_metrics = report.get("REM")
        if isinstance(macro_avg, dict):
            row["macro_f1"] = macro_avg.get("f1-score")
            row["macro_recall"] = macro_avg.get("recall")
            row["macro_precision"] = macro_avg.get("precision")
        if isinstance(weighted_avg, dict):
            row["weighted_f1"] = weighted_avg.get("f1-score")
        if isinstance(n1_metrics, dict):
            row["n1_recall"] = n1_metrics.get("recall")
            row["n1_f1"] = n1_metrics.get("f1-score")
        if isinstance(rem_metrics, dict):
            row["rem_recall"] = rem_metrics.get("recall")
            row["rem_f1"] = rem_metrics.get("f1-score")
    for prefix, key in (
        ("pred", "prediction_distribution"),
        ("true", "target_distribution"),
    ):
        distribution = summary.get(key)
        if isinstance(distribution, dict):
            for label, count in distribution.items():
                row[f"{prefix}_{label}"] = count
    train_sample = summary.get("train_sample")
    if isinstance(train_sample, dict):
        row["train_sample_size"] = train_sample.get("sample_size")
        row["train_sample_accuracy"] = train_sample.get("accuracy")
        pred_dist = train_sample.get("pred_distribution")
        if isinstance(pred_dist, dict):
            for label, count in pred_dist.items():
                row[f"train_sample_pred_{label}"] = count
    return row


def export_history_csv(
    path: Path,
    history: list[dict[str, object]],
    fold: int,
    held_out_record: str,
    stage: str,
) -> None:
    rows = [flatten_history_row(item, fold, held_out_record, stage) for item in history]
    write_csv(path, rows)


def save_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [INDEX_TO_STAGE[i] for i in range(len(matrix))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix, strict=False):
            writer.writerow([label, *row])


def export_run_tables(run_dir: Path, summaries: list[dict[str, object]]) -> None:
    cnn_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    combined_rows: list[dict[str, object]] = []
    for fold_summary in summaries:
        fold = fold_summary["fold"]
        held_out = fold_summary["held_out_record"]
        fold_dir = run_dir / f"fold_{int(fold):02d}_{held_out}"
        cnn = fold_summary.get("cnn", {})
        context = fold_summary.get("context_bilstm", {})
        if isinstance(cnn, dict) and "confusion_matrix" in cnn:
            save_confusion_matrix(fold_dir / "cnn_confusion_matrix.csv", cnn["confusion_matrix"])
            cnn_rows.append({"fold": fold, **flatten_metrics(cnn, prefix="cnn_")})
        if isinstance(context, dict) and "confusion_matrix" in context:
            save_confusion_matrix(
                fold_dir / "context_bilstm_confusion_matrix.csv",
                context["confusion_matrix"],
            )
            context_rows.append({"fold": fold, **flatten_metrics(context, prefix="context_")})
        row = {"fold": fold, "held_out_record": held_out}
        if isinstance(cnn, dict):
            row.update(flatten_metrics(cnn, prefix="cnn_"))
        if isinstance(context, dict):
            row.update(flatten_metrics(context, prefix="context_"))
        combined_rows.append(row)
    write_csv(run_dir / "cnn_metrics.csv", cnn_rows)
    write_csv(run_dir / "context_bilstm_metrics.csv", context_rows)
    write_csv(run_dir / "fold_metrics.csv", combined_rows)


def evaluate_predictions(
    true: np.ndarray,
    pred: np.ndarray,
    cfg: ReproductionConfig,
    held_out_record: str,
) -> dict[str, object]:
    matrix = confusion_matrix(true, pred, labels=list(range(cfg.n_classes)))
    return {
        "held_out_record": held_out_record,
        "accuracy": float(accuracy_score(true, pred)),
        "confusion_matrix": matrix.tolist(),
        "target_distribution": {
            INDEX_TO_STAGE[i]: int(matrix[i, :].sum()) for i in range(cfg.n_classes)
        },
        "prediction_distribution": {
            INDEX_TO_STAGE[i]: int(matrix[:, i].sum()) for i in range(cfg.n_classes)
        },
        "classification_report": classification_report(
            true,
            pred,
            labels=list(range(cfg.n_classes)),
            target_names=[INDEX_TO_STAGE[i] for i in range(cfg.n_classes)],
            zero_division=0,
            output_dict=True,
        ),
    }


def extract_features(model, x: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    require_torch()
    import torch

    model.eval()
    loader = make_loader(x, y, batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            chunks.append(model.encode(xb.to(device)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def run_cnn_leave_one_record(
    cfg: ReproductionConfig,
    dataset_path: Path,
    run_dir: Path,
    max_folds: int | None,
    steps: int,
    batch_size: int,
    device: str,
    eval_every: int,
    class_weights: str,
    group_key: str,
) -> Path:
    require_torch()
    import torch

    x, y, groups = load_npz_dataset(dataset_path, group_key=group_key)
    x_model = x.reshape(len(x), 3, -1)
    logo = LeaveOneGroupOut()
    fold_summaries = []
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(x_model, y, groups=groups), start=1):
        if max_folds is not None and fold_idx > max_folds:
            break
        held_out = str(groups[test_idx][0])
        fold_dir = run_dir / f"cnn_fold_{fold_idx:02d}_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model = PaperCNN(n_classes=cfg.n_classes, feature_dim=cfg.cnn_feature_dim).to(device)
        criterion = make_criterion(y[train_idx], device, class_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=reference_decay_step_count(len(train_idx)),
            gamma=0.95,
        )
        test_loader = make_loader(x_model[test_idx], y[test_idx], batch_size, shuffle=False)

        losses, history, best_summary = train_steps_with_validation(
            model,
            x_model[train_idx],
            y[train_idx],
            optimizer,
            criterion,
            device,
            steps=steps,
            seed=cfg.seed + fold_idx,
            validation_loader=test_loader,
            cfg=cfg,
            held_out_record=held_out,
            checkpoint_path=fold_dir / "cnn_best.pt",
            batch_size=batch_size,
            lr_scheduler=scheduler,
            eval_every=eval_every,
        )

        cnn_best_checkpoint = fold_dir / "cnn_best.pt"
        load_model_checkpoint(cnn_best_checkpoint, model, device)
        pred, true, _ = predict(model, test_loader, device)
        final_summary = evaluate_predictions(true, pred, cfg, held_out)
        report = {
            "fold": fold_idx,
            "test_record": held_out,
            "group_key": group_key,
            "accuracy": final_summary["accuracy"],
            "confusion_matrix": final_summary["confusion_matrix"],
            "target_distribution": final_summary["target_distribution"],
            "prediction_distribution": final_summary["prediction_distribution"],
            "train_loss": losses,
            "final_lr": float(optimizer.param_groups[0]["lr"]),
            "validation_history": history,
            "best_validation": best_summary,
            "feature_checkpoint": str(cnn_best_checkpoint),
            "class_weights": class_weights,
        }
        export_history_csv(fold_dir / "cnn_validation_history.csv", history, fold_idx, held_out, "cnn")
        fold_summaries.append(report)
    (run_dir / "cnn_leave_one_record_summary.json").write_text(
        json.dumps(fold_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir / "cnn_leave_one_record_summary.json"


def run_fold_reproduction(
    cfg: ReproductionConfig,
    dataset_path: Path,
    run_dir: Path,
    max_folds: int | None,
    cnn_steps: int,
    context_steps: int,
    batch_size: int,
    device: str,
    eval_every: int,
    class_weights: str,
    group_key: str,
) -> list[dict[str, object]]:
    require_torch()
    import torch

    x, y, groups = load_npz_dataset(dataset_path, group_key=group_key)
    x_model = x.reshape(len(x), 3, -1)
    logo = LeaveOneGroupOut()
    summaries: list[dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(x_model, y, groups=groups), start=1):
        if max_folds is not None and fold_idx > max_folds:
            break
        held_out = str(groups[test_idx][0])
        fold_dir = run_dir / f"fold_{fold_idx:02d}_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        cnn = PaperCNN(n_classes=cfg.n_classes, feature_dim=cfg.cnn_feature_dim).to(device)
        criterion = make_criterion(y[train_idx], device, class_weights)
        optimizer = torch.optim.SGD(cnn.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=reference_decay_step_count(len(train_idx)),
            gamma=0.95,
        )
        test_loader = make_loader(x_model[test_idx], y[test_idx], batch_size, shuffle=False)

        cnn_losses, cnn_history, cnn_best = train_steps_with_validation(
            cnn,
            x_model[train_idx],
            y[train_idx],
            optimizer,
            criterion,
            device,
            steps=cnn_steps,
            seed=cfg.seed + fold_idx,
            validation_loader=test_loader,
            cfg=cfg,
            held_out_record=held_out,
            checkpoint_path=fold_dir / "cnn_best.pt",
            batch_size=batch_size,
            lr_scheduler=scheduler,
            eval_every=eval_every,
        )
        cnn_best_checkpoint = fold_dir / "cnn_best.pt"
        load_model_checkpoint(cnn_best_checkpoint, cnn, device)
        cnn_pred, cnn_true, _ = predict(cnn, test_loader, device)
        cnn_summary = evaluate_predictions(cnn_true, cnn_pred, cfg, held_out)
        cnn_summary["train_loss"] = cnn_losses
        cnn_summary["final_lr"] = float(optimizer.param_groups[0]["lr"])
        cnn_summary["validation_history"] = cnn_history
        cnn_summary["best_validation"] = cnn_best
        cnn_summary["feature_checkpoint"] = str(cnn_best_checkpoint)
        cnn_summary["class_weights"] = class_weights
        export_history_csv(fold_dir / "cnn_validation_history.csv", cnn_history, fold_idx, held_out, "cnn")

        train_features = extract_features(cnn, x_model[train_idx], y[train_idx], batch_size, device)
        test_features = extract_features(cnn, x_model[test_idx], y[test_idx], batch_size, device)
        feature_groups = np.concatenate([groups[train_idx], groups[test_idx]])
        feature_labels = np.concatenate([y[train_idx], y[test_idx]])
        features = np.concatenate([train_features, test_features])

        windows, labels, window_groups = make_context_windows(
            features,
            feature_labels,
            feature_groups,
            radius=cfg.context_radius,
        )
        context_train_idx = np.flatnonzero(window_groups != held_out)
        context_test_idx = np.flatnonzero(window_groups == held_out)

        context_summary: dict[str, object]
        if len(context_train_idx) == 0 or len(context_test_idx) == 0:
            context_summary = {
                "held_out_record": held_out,
                "error": "Not enough context windows for this fold.",
            }
        else:
            context = ContextBiLSTM(
                feature_dim=cfg.cnn_feature_dim,
                hidden_dim=100,
                n_classes=cfg.n_classes,
            ).to(device)
            context_criterion = make_criterion(labels[context_train_idx], device, class_weights)
            context_optimizer = torch.optim.Adam(context.parameters(), lr=1e-4, weight_decay=1e-3)
            context_test_loader = make_loader(
                windows[context_test_idx],
                labels[context_test_idx],
                batch_size,
                shuffle=False,
            )
            context_losses, context_history, context_best = train_steps_with_validation(
                context,
                windows[context_train_idx],
                labels[context_train_idx],
                context_optimizer,
                context_criterion,
                device,
                steps=context_steps,
                seed=cfg.seed + 10_000 + fold_idx,
                validation_loader=context_test_loader,
                cfg=cfg,
                held_out_record=held_out,
                checkpoint_path=fold_dir / "context_bilstm_best.pt",
                batch_size=batch_size,
                eval_every=eval_every,
            )
            context_best_checkpoint = fold_dir / "context_bilstm_best.pt"
            load_model_checkpoint(context_best_checkpoint, context, device)
            context_pred, context_true, _ = predict(context, context_test_loader, device)
            context_summary = evaluate_predictions(context_true, context_pred, cfg, held_out)
            context_summary["train_loss"] = context_losses
            context_summary["validation_history"] = context_history
            context_summary["best_validation"] = context_best
            context_summary["feature_checkpoint"] = str(context_best_checkpoint)
            context_summary["class_weights"] = class_weights
            export_history_csv(
                fold_dir / "context_bilstm_validation_history.csv",
                context_history,
                fold_idx,
                held_out,
                "context_bilstm",
            )

        fold_summary = {
            "fold": fold_idx,
            "held_out_record": held_out,
            "group_key": group_key,
            "cnn": cnn_summary,
            "context_bilstm": context_summary,
        }
        summaries.append(fold_summary)
        (fold_dir / "summary.json").write_text(
            json.dumps(fold_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (run_dir / "fold_reproduction_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export_run_tables(run_dir, summaries)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sleep staging reproduction models.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Use an existing NPZ dataset directly. When omitted, the MIT-BIH cache "
            "under --cache-dir is used or generated with --prepare."
        ),
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--max-folds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1200, help="CNN optimization steps per fold.")
    parser.add_argument(
        "--context-steps",
        type=int,
        default=10000,
        help="BiLSTM optimization steps per fold.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Deprecated alias for --steps, kept for previous smoke-test commands.",
    )
    parser.add_argument(
        "--context-epochs",
        type=int,
        default=None,
        help="Deprecated alias for --context-steps.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Evaluate held-out fold and checkpoint best model every N training steps.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--class-weights",
        choices=CLASS_WEIGHT_CHOICES,
        default="none",
        help=(
            "Loss class weighting mode. 'none' is closer to the reference code because "
            "the mini-batch sampler is already class-balanced."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["fold", "cnn"],
        default="fold",
        help="Run full CNN->BiLSTM fold reproduction or CNN-only leave-one-record evaluation.",
    )
    parser.add_argument(
        "--group-key",
        default="record",
        help=(
            "NPZ key used for LeaveOneGroupOut splitting. Use 'record' for leave-one-night "
            "or 'subject' for leave-one-subject on Sleep-EDF caches."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_kwargs = {"cache_dir": args.cache_dir, "run_dir": args.run_dir}
    if args.data_dir is not None:
        cfg_kwargs["data_dir"] = args.data_dir
    cfg = ReproductionConfig(**cfg_kwargs)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    require_torch()
    import torch

    set_seed(cfg.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    if args.dataset_path is not None:
        dataset_path = args.dataset_path
        if args.prepare:
            raise ValueError("--prepare cannot be combined with --dataset-path.")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    else:
        dataset_path = cfg.cache_dir / "mitbih_sleep_eeg_epochs.npz"
        if args.prepare or not dataset_path.exists():
            dataset_path = build_npz_dataset(cfg, overwrite=args.prepare)
    cnn_steps = args.epochs if args.epochs is not None else args.steps
    context_steps = args.context_epochs if args.context_epochs is not None else args.context_steps
    if args.stage == "cnn":
        summary_path = run_cnn_leave_one_record(
            cfg,
            dataset_path,
            args.run_dir,
            max_folds=args.max_folds,
            steps=cnn_steps,
            batch_size=args.batch_size,
            device=device,
            eval_every=args.eval_every,
            class_weights=args.class_weights,
            group_key=args.group_key,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = run_fold_reproduction(
            cfg,
            dataset_path,
            args.run_dir,
            max_folds=args.max_folds,
            cnn_steps=cnn_steps,
            context_steps=context_steps,
            batch_size=args.batch_size,
            device=device,
            eval_every=args.eval_every,
            class_weights=args.class_weights,
            group_key=args.group_key,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create locked statistical figures and tables for the Sleep Medicine package."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sleep_medicine_submission_figures"
STAGES = ["W", "N1", "N2", "N3", "REM"]


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def savefig(fig, name: str):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def external_validation():
    data = load("outputs/external_validation/pre_submission/pre_submission_statistics.json")
    source = load(
        "outputs/attnsleep_multichannel_baseline_train/"
        "full20_quality_gate_4ch_emg_30epoch/aggregate_metrics.json"
    )
    source_metrics = {
        "accuracy": source["final_test_accuracy_mean"],
        "macro_f1": source["final_test_macro_f1_mean"],
        "kappa": source["final_test_kappa_mean"],
    }
    order = ["SHHS1", "SHHS2", "SleepEDF_ST", "ISRUC_Cohort_I"]
    labels = ["SHHS1", "SHHS2\n(follow-up)", "Sleep-EDF ST", "ISRUC I"]
    rows = []
    for key, label in zip(order, labels):
        cohort = data["cohorts"][key]
        for metric in ["accuracy", "macro_f1", "kappa"]:
            ci = cohort["subject_bootstrap_95ci"][metric]
            rows.append({
                "cohort": key,
                "label": label.replace("\n", " "),
                "metric": metric,
                "estimate": cohort["point_estimate"][metric],
                "ci_low": ci[0],
                "ci_high": ci[1],
                "n_subjects": cohort["n_subjects"],
                "n_records": cohort["n_records"],
                "n_epochs": cohort["n_epochs"],
            })
    with (OUT / "external_validation_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparison_rows = []
    for key, label in zip(order, labels):
        cohort = data["cohorts"][key]
        for metric in ["accuracy", "macro_f1", "kappa"]:
            value = cohort["point_estimate"][metric]
            comparison_rows.append({
                "evaluation": key,
                "label": label.replace("\n", " "),
                "metric": metric,
                "source_domain_reference": source_metrics[metric],
                "external_frozen": value,
                "absolute_change_vs_source": value - source_metrics[metric],
            })
    with (OUT / "source_to_external_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    source_rows = []
    source_std = {
        "accuracy": source["final_test_accuracy_std"],
        "macro_f1": source["final_test_macro_f1_std"],
        "kappa": source["final_test_kappa_std"],
    }
    for metric in ["accuracy", "macro_f1", "kappa"]:
        source_rows.append({
            "evaluation": "Sleep-EDF-20 source-domain",
            "protocol": "subject-wise 20-fold internal estimate",
            "metric": metric,
            "estimate": source_metrics[metric],
            "fold_sd": source_std[metric],
            "uncertainty_note": "fold-level SD; not subject-bootstrap CI",
        })
    with (OUT / "source_domain_reference.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    colors = ["#244B74", "#5C7EA5", "#7A9E9F", "#B66D45"]
    for ax, metric, title in zip(
        axes,
        ["accuracy", "macro_f1", "kappa"],
        ["Accuracy", "Macro-F1", "Cohen's kappa"],
    ):
        values = [data["cohorts"][k]["point_estimate"][metric] for k in order]
        lows = [data["cohorts"][k]["subject_bootstrap_95ci"][metric][0] for k in order]
        highs = [data["cohorts"][k]["subject_bootstrap_95ci"][metric][1] for k in order]
        x = np.arange(len(order))
        ax.bar(x, values, color=colors, width=0.68, edgecolor="white", linewidth=0.6)
        ax.errorbar(x, values, yerr=[np.array(values) - lows, np.array(highs) - values],
                    fmt="none", ecolor="#222222", capsize=3, linewidth=0.9)
        ax.axhline(source_metrics[metric], color="#111827", linestyle="--", linewidth=1.0,
                   label="Sleep-EDF-20 source-domain reference")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score; error bars are subject-level 95% CI")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Source-domain reference and frozen external validation", fontsize=12, y=1.02)
    fig.tight_layout()
    savefig(fig, "figure1_external_validation")


def shhs_budget():
    data = load("outputs/external_validation/shhs/adaptation/shhs1_budget_curve_all_splits.json")
    grouped = {}
    for row in data["rows"]:
        grouped.setdefault(int(row["budget_subjects"]), []).append(row)
    budgets = sorted(grouped)
    summary = []
    for budget in budgets:
        vals = grouped[budget]
        frozen = np.array([r["frozen"]["macro_f1"] for r in vals])
        adapted = np.array([r["adapted"]["macro_f1"] for r in vals])
        delta = np.array([r["delta"]["macro_f1"] for r in vals])
        summary.append({
            "budget_subjects": budget,
            "n_splits": len(vals),
            "frozen_mean": float(frozen.mean()),
            "adapted_mean": float(adapted.mean()),
            "adapted_sd": float(adapted.std(ddof=1)),
            "delta_mean": float(delta.mean()),
            "delta_sd": float(delta.std(ddof=1)),
            "positive_splits": int((delta > 0).sum()),
        })
    with (OUT / "shhs1_budget_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    x = np.array(budgets)
    frozen = np.array([r["frozen_mean"] for r in summary])
    adapted = np.array([r["adapted_mean"] for r in summary])
    adapted_sd = np.array([r["adapted_sd"] for r in summary])
    delta = np.array([r["delta_mean"] for r in summary])
    delta_sd = np.array([r["delta_sd"] for r in summary])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    axes[0].plot(x, frozen, "o--", color="#6B7280", label="Frozen source model")
    axes[0].errorbar(x, adapted, yerr=adapted_sd, fmt="o-", color="#1F6F8B", capsize=3,
                     label="Full-model adaptation")
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xticks(x, [str(v) for v in x])
    axes[0].set_xlabel("Labeled calibration subjects")
    axes[0].set_ylabel("Held-out Macro-F1")
    axes[0].set_title("SHHS1 annotation budget")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].errorbar(x, delta, yerr=delta_sd, fmt="o-", color="#B45309", capsize=3)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xticks(x, [str(v) for v in x])
    axes[1].set_xlabel("Labeled calibration subjects")
    axes[1].set_ylabel("Macro-F1 change vs frozen")
    axes[1].set_title("Paired adaptation gain")
    axes[1].grid(alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, "figure2_shhs1_budget_curve")


def clinical_mae():
    data = load("outputs/external_validation/sleep_medicine_clinical_bootstrap.json")
    endpoints = [
        ("tst_hours", "TST (h)"),
        ("sleep_efficiency_pct", "Sleep efficiency (%)"),
        ("waso_minutes", "WASO (min)"),
        ("rem_latency_minutes", "REM latency (min)"),
    ]
    rows = []
    for cohort in ["shhs1", "shhs2"]:
        for split, payload in data["cohorts"][cohort].items():
            for key, label in endpoints:
                for regime in ["frozen", "adapted"]:
                    point = payload[regime][key]["point"]
                    ci = payload[regime][key]["ci95"]["mae"]
                    rows.append({
                        "cohort": cohort,
                        "split": split,
                        "endpoint": key,
                        "label": label,
                        "regime": regime,
                        "n_subjects": point["n_subjects"],
                        "mae": point["mae"],
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                    })
    with (OUT / "clinical_mae_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=False)
    for ax, cohort in zip(axes, ["shhs1", "shhs2"]):
        sub = [r for r in rows if r["cohort"] == cohort]
        means = []
        errors = []
        for _, label in endpoints:
            for regime in ["frozen", "adapted"]:
                q = [r for r in sub if r["label"] == label and r["regime"] == regime]
                vals = np.array([r["mae"] for r in q])
                means.append(vals.mean())
                errors.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        x = np.arange(len(endpoints))
        width = 0.34
        frozen = np.array(means[::2]); adapted = np.array(means[1::2])
        frozen_sd = np.array(errors[::2]); adapted_sd = np.array(errors[1::2])
        ax.bar(x - width / 2, frozen, width, yerr=frozen_sd, capsize=3,
               color="#9CA3AF", label="Frozen")
        ax.bar(x + width / 2, adapted, width, yerr=adapted_sd, capsize=3,
               color="#1F6F8B", label="Adapted")
        ax.set_xticks(x, [label for _, label in endpoints], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Mean absolute error (mean across locked splits)")
        ax.set_title("SHHS1" if cohort == "shhs1" else "SHHS2 follow-up")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sleep-structure error before and after target-domain adaptation", fontsize=11, y=1.02)
    fig.tight_layout()
    savefig(fig, "figure3_clinical_mae")


def confusion_heatmaps():
    data = load("outputs/external_validation/pre_submission/pre_submission_statistics.json")
    cohorts = ["SHHS1", "SHHS2", "ISRUC_Cohort_I"]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2), constrained_layout=True)
    for ax, cohort in zip(axes, cohorts):
        matrix = np.asarray(data["cohorts"][cohort]["confusion_matrix"], dtype=float)
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i, j] > 0.5 else "#111111")
        ax.set_title(cohort.replace("_Cohort_I", " I"), fontsize=9)
        ax.set_xticks(range(5), STAGES, fontsize=7)
        ax.set_yticks(range(5), STAGES, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Reference", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Row-normalized proportion")
    savefig(fig, "figure4_external_confusion_matrices")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.labelsize": 9})
    external_validation()
    shhs_budget()
    clinical_mae()
    confusion_heatmaps()
    print("WROTE", OUT)


if __name__ == "__main__":
    main()

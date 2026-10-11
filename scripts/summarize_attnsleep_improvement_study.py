"""Summarize the three locked stages without selecting any further settings."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_attnsleep_generalization_gap import metrics_from_confusion
from scripts.run_attnsleep_selection_study import write_json

KEYS = ("accuracy", "macro_f1", "n1_precision", "n1_recall", "n1_f1", "n3_f1", "kappa")


def from_cm(cm):
    value = metrics_from_confusion(cm)
    value.update({"n1_precision": value["class_precision"][1], "n1_recall": value["class_recall"][1],
                  "n1_f1": value["class_f1"][1], "n3_f1": value["class_f1"][3]})
    return value


def subject_pairs(results, before, after):
    rows = []
    for result in results:
        groups = {}
        for mode in (before, after):
            records = result[mode]["records"]
            for record in records:
                subject = record["subject"]
                group = groups.setdefault(subject, {})
                group[mode] = group.get(mode, np.zeros((5, 5), dtype=np.int64)) + np.array(record["confusion_matrix"])
        for subject, modes in groups.items():
            a, b = from_cm(modes[before]), from_cm(modes[after])
            rows.append({"seed": result["seed"], "subject": subject,
                         **{key: b[key] - a[key] for key in KEYS}})
    # Average repeated seeds first, then resample the ten distinct subjects.
    subjects = sorted({row["subject"] for row in rows})
    rng = np.random.default_rng(20261010)
    indices = rng.integers(0, len(subjects), size=(5000, len(subjects)))
    statistics = {}
    for key in KEYS:
        values = np.array([np.mean([row[key] for row in rows if row["subject"] == subject]) for subject in subjects])
        draws = values[indices].mean(1)
        statistics[key] = {"mean_subject_delta": float(values.mean()),
                           "descriptive_bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist()}
    return {"subject_count": len(subjects), "rows": rows, "statistics": statistics,
            "interpretation": "Descriptive paired subject bootstrap after averaging repeated seeds. Overlapping fits, prior inspection and reused development subjects prevent confirmatory population inference."}


def main():
    output = ROOT / "outputs/attnsleep_improvement_pipeline_20261010"
    output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "outputs" / name / "summary.json" for name in (
        "attnsleep_selection_study_20261010", "attnsleep_emg_study_20261010", "attnsleep_context_residual_study_20261010")]
    stages = [json.loads(path.read_text()) for path in paths]
    first, second, third = stages
    stage1 = [{"seed": row["seed"], **row["rules"]} for row in first["results"]]
    stage2 = [{"seed": row["seed"], "four": row["four"], "three": row["three"]} for row in second["results"]]
    stage3 = [{"seed": row["seed"], **row["outcomes"]} for row in third["results"]]
    comparisons = {
        "stage1_guarded_minus_accuracy": subject_pairs(stage1, "accuracy", "guarded_macro"),
        "stage2_three_minus_four": subject_pairs(stage2, "four", "three"),
        "stage3_selective_minus_baseline": subject_pairs(stage3, "baseline", "selective_context"),
        "stage3_selective_minus_raw": subject_pairs(stage3, "raw_residual", "selective_context"),
        "stage3_selective_minus_fixed": subject_pairs(stage3, "fixed_context", "selective_context"),
        "stage3_fixed_minus_baseline": subject_pairs(stage3, "baseline", "fixed_context"),
        "stage3_fixed_minus_raw": subject_pairs(stage3, "raw_residual", "fixed_context"),
    }
    table = []
    for stage_name, rows, modes in (
        ("stage1", stage1, ("accuracy", "guarded_macro")),
        ("stage2", stage2, ("four", "three")),
        ("stage3", stage3, ("baseline", "raw_residual", "fixed_context", "selective_context")),
    ):
        for mode in modes:
            table.append({"stage": stage_name, "mode": mode,
                          **{key: float(np.mean([row[mode]["pooled"][key] for row in rows])) for key in KEYS}})
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["stage", "mode", *KEYS])
        writer.writeheader()
        writer.writerows(table)
    write_json(output / "analysis.json", {"tables": table, "paired_subject_comparisons": comparisons,
               "stage2_selector": first["stage2_selector"], "stage3_channels": second["stage3_channels"],
               "context_candidate_for_external_confirmation": third["candidate_for_external_confirmation"],
               "stage_summaries": [str(path) for path in paths]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    plot_keys = ("accuracy", "macro_f1", "n1_f1", "n3_f1")
    plotted = (
        ("stage1_guarded_minus_accuracy", "Checkpoint selection"),
        ("stage2_three_minus_four", "Remove EMG"),
        ("stage3_fixed_minus_baseline", "Fixed context vs baseline"),
        ("stage3_selective_minus_fixed", "Selective vs fixed context"),
    )
    for axis, (comparison_name, title) in zip(axes, plotted):
        comparison = comparisons[comparison_name]
        values = np.array([comparison["statistics"][key]["mean_subject_delta"] for key in plot_keys]) * 100
        ci = np.array([comparison["statistics"][key]["descriptive_bootstrap_95ci"] for key in plot_keys]) * 100
        axis.errorbar(np.arange(4), values, yerr=np.maximum(np.stack((values - ci[:, 0], ci[:, 1] - values)), 0),
                      fmt="o", capsize=4)
        axis.axhline(0, color="gray", linewidth=1)
        axis.set_xticks(np.arange(4), ["Accuracy", "Macro-F1", "N1 F1", "N3 F1"], rotation=25)
        axis.set_title(title, fontsize=10)
    axes[0].set_ylabel("Mean paired subject change (percentage points)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"paired_subject_changes.{extension}", dpi=250)
    plt.close(fig)
    lines = ["# 主线顺序实验结果（2026-10-10）", "",
             "完整执行三阶段，GPU 训练和推理；每阶段五组受试者划分、三随机种子，30 epoch。",
             "每组额外评估受试者独立于本组拟合及 checkpoint 选择，跨组共十人。",
             "额外评估者用于研究决策，属于内部开发评估；不是全新独立测试。", "",
             "| 阶段 | 模型/选型规则 | Accuracy | Macro-F1 | N1 precision | N1 recall | N1 F1 | N3 F1 | Kappa |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| " + " | ".join([row["stage"], row["mode"], *[f"{row[key]:.4f}" for key in KEYS]]) + " |")
    lines += ["", "表中指标是每个 fold-seed 的本组 pooled 指标，再取15个描述性均值；不是15个独立样本。",
              f"第一阶段锁定第二阶段 selector=`{first['stage2_selector']}`；第二阶段锁定第三阶段 channels=`{second['stage3_channels']}`。",
              f"第三阶段选择性上下文满足全部内部升级条件：`{third['candidate_for_external_confirmation']}`。即使通过，仍需独立外部确认，不能立即替换投稿主线。", "",
              "## 第三阶段解释与种子复核", "",
              "以下差值为15组 fold-seed pooled 指标的均值，单位为百分点；种子列为各种子五组平均 Macro-F1 差值。", "",
              "| 比较 | Accuracy变化 | Macro-F1变化 | N1 recall变化 | N1 F1变化 | N3 F1变化 | seed 7777/8888/9999 Macro-F1变化 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for before, after, label in (
        ("baseline", "raw_residual", "普通残差 − 基线"),
        ("baseline", "fixed_context", "固定上下文 − 基线"),
        ("baseline", "selective_context", "选择性上下文 − 基线"),
        ("fixed_context", "selective_context", "选择性上下文 − 固定上下文"),
    ):
        deltas = {key: float(np.mean([row[after]["pooled"][key] - row[before]["pooled"][key]
                                     for row in stage3])) for key in KEYS}
        seed_deltas = [float(np.mean([row[after]["pooled"]["macro_f1"] - row[before]["pooled"]["macro_f1"]
                                     for row in stage3 if row["seed"] == seed])) for seed in (7777, 8888, 9999)]
        lines.append("| " + " | ".join([label, *[f"{100 * deltas[key]:+.3f}" for key in
                     ("accuracy", "macro_f1", "n1_recall", "n1_f1", "n3_f1")],
                     "/".join(f"{100 * value:+.3f}" for value in seed_deltas)]) + " |")
    lines += ["", "固定上下文与选择性上下文均优于原基线，三个种子的 Macro-F1 平均差值均为正；",
              "普通单epoch残差没有平均收益，提示改善与增加历史上下文有关，而非仅增加残差头容量。",
              "选择性门控相对固定融合的平均 Macro-F1 和 N1 F1 略低，且只有一个种子的 Macro-F1 差值为正；",
              "因此不能把收益归因于自适应门控。选择性方案未满足预设的超越全部对照条件，不升级投稿主线。",
              "固定上下文是有希望的后续方向，但本轮未预设其单独升级规则；不能事后将其宣布为已确认优胜模型。", "",
              "## 受试者配对描述性区间", "",
              "受试者配对 bootstrap 在每名受试者上先平均三种子差值，再重采样十名受试者5000次。",
              "这里每名受试者等权，与上表两名受试者 pooled 后再按 fold-seed 平均的估计量不同。",
              "训练集重叠、历史查看和开发集复用限制推断，区间仅作描述性不确定性诊断；未进行确认性显著性检验。", "",
              "| 比较 | Macro-F1平均变化及95%描述性区间（百分点） | N1 F1平均变化及区间 | N3 F1平均变化及区间 |",
              "|---|---|---|---|"]
    for comparison_name, label in (
        ("stage3_fixed_minus_baseline", "固定上下文 − 基线"),
        ("stage3_selective_minus_baseline", "选择性上下文 − 基线"),
        ("stage3_selective_minus_fixed", "选择性上下文 − 固定上下文"),
    ):
        cells = [label]
        for key in ("macro_f1", "n1_f1", "n3_f1"):
            statistic = comparisons[comparison_name]["statistics"][key]
            low, high = statistic["descriptive_bootstrap_95ci"]
            cells.append(f"{100 * statistic['mean_subject_delta']:+.3f} [{100 * low:+.3f}, {100 * high:+.3f}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["",
              "详细 JSON、CSV、PNG/PDF 图位于 `outputs/attnsleep_improvement_pipeline_20261010/`。",
              "此次没有运行外部队列最终确认，不支持外部零样本收益、临床部署或新 SOTA 声明。"]
    (ROOT / "docs/mainline_improvement_results_20261010.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Locked GPU-only ST replication of the already selected fixed-context candidate."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_attnsleep_context_residual_study import (
    ResidualHead, encode_records, tensors, KEYS,
)
from scripts.run_attnsleep_selection_study import write_json
from scripts.audit_attnsleep_generalization_gap import metrics_from_confusion

MODES = ("baseline", "raw_residual", "fixed_context")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cm_metrics(cm):
    value = metrics_from_confusion(cm)
    return {**{k: value[k] for k in ("accuracy", "macro_f1", "kappa")},
            "n1_precision": value["class_precision"][1], "n1_recall": value["class_recall"][1],
            "n1_f1": value["class_f1"][1], "n3_f1": value["class_f1"][3]}


def confusion(y, p):
    return np.bincount(5 * y.astype(int) + p.astype(int), minlength=25).reshape(5, 5)


def lock(args):
    source = json.loads((args.source_root / "protocol.json").read_text())
    jobs = []
    for seed in source["seeds"]:
        for panel in source["panels"]:
            folder = args.source_root / f"seed{seed}" / f"fold_{panel['fold_id']:02d}"
            selected = json.loads((folder / "locked_selection.json").read_text())
            paths = [Path(selected["source_checkpoint"]), folder / "feature_standardizer.npz",
                     folder / "locked_selection.json"]
            paths += [folder / f"{m}_epoch{selected['choices'][m]['epoch']:02d}.pt" for m in MODES[1:]]
            jobs.append({"seed": seed, "fold": panel["fold_id"], "folder": str(folder),
                         "source_checkpoint": selected["source_checkpoint"],
                         "choices": {m: selected["choices"][m]["epoch"] for m in MODES[1:]},
                         "artifacts_sha256": {str(p): sha(p) for p in paths}})
    paths = sorted(args.data_dir.glob("*.npz"))
    if len(paths) != 44 or len({p.name[3:5] for p in paths}) != 22:
        raise RuntimeError("Expected all 44 ST records and 22 subjects")
    summary = json.loads((args.data_dir / "summary.json").read_text())
    raw_dir = ROOT / summary["raw_dir"]
    annotations = [raw_dir / row["hypnogram_file"] for row in summary["records"]]
    protocol = {
        "study": "Source-locked fixed causal context replication on the complete Sleep-EDF ST cohort",
        "date": "2026-10-10", "cohort": "Sleep-EDF ST", "records": 44, "subjects": 22,
        "data_dir": str(args.data_dir.resolve()), "modes": MODES,
        "data_sha256": {str(p): sha(p) for p in [args.data_dir / "summary.json", *paths, *annotations]},
        "source_protocol_sha256": sha(args.source_root / "protocol.json"), "jobs": jobs,
        "inference": "CUDA only; identity inputs; no TTA, target fitting, BN updates, calibration or model selection. Original-timeline gap reset; source-fit normalization only.",
        "primary_estimator": "Equal-weight mean of complete-cohort pooled metrics across 15 locked model pairs; five-fold means within each seed for consistency checks.",
        "secondary_estimator": "Mean logits across five models per seed and across all 15; explicitly separate from historical 20-fold ensemble.",
        "bootstrap": "5000 paired subject draws, both nights together, same draw for all frozen models; average model metrics after pooling resampled subject confusion matrices. Conditional descriptive uncertainty for this locked model set.",
        "decision": {"macro_gain_vs_baseline_and_raw": 0.005, "positive_seed_means_vs_each": 2,
                     "accuracy_min_vs_baseline": -0.005, "n1_recall_min": -0.02,
                     "n1_f1_min": 0.0, "n3_f1_min": -0.01},
        "limitations": ["ST labels and previous models' outcomes were inspected historically; not pristine confirmation.",
                        "ST is an independent subset, not a separate acquisition database; ISRUC/SHHS raw disk currently unavailable.",
                        "Inherited annotation-based removal/cropping is used only for benchmark comparability, not a label-free deployment pipeline.",
                        "ST waveform EMG differs from source Sleep-EDF-20 low-rate EMG representation; no target-specific harmonization.",
                        "Source fitting sets overlap, seeds are not independent subjects; no population significance claim.",
                        "No automatic manuscript model replacement or target-driven tuning after these outcomes."],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "protocol.json"
    # JSON canonicalization makes repeat invocation comparable (tuple becomes list).
    protocol = json.loads(json.dumps(protocol))
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("Refusing changed external protocol")
    else:
        write_json(path, protocol)
    return protocol, sha(path)


def worker(args, protocol, digest, index):
    from model.model import AttnSleep
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    job = protocol["jobs"][index]
    folder = Path(job["folder"])
    output = args.output_dir / f"seed{job['seed']}" / f"fold_{job['fold']:02d}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists():
        saved = json.loads((output / "comparison.json").read_text())
        assert saved["protocol_sha256"] == digest
        return
    for path, expected in job["artifacts_sha256"].items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f"Source artifact changed: {path}")
    device = torch.device("cuda")
    ck = torch.load(job["source_checkpoint"], map_location=device, weights_only=False)
    model = AttnSleep(input_channels=4, channel_fusion="quality-gated", channel_dropout=0.15).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    with np.load(folder / "feature_standardizer.npz") as data:
        mean, std = data["mean"], data["std"]
    heads = {}
    for mode in MODES[1:]:
        head = ResidualHead(len(mean), mode).to(device)
        head.load_state_dict(torch.load(folder / f"{mode}_epoch{job['choices'][mode]:02d}.pt", map_location=device, weights_only=True))
        heads[mode] = head.eval()
    summary = json.loads((args.data_dir / "summary.json").read_text())
    records = []
    for number, path in enumerate(sorted(args.data_dir.glob("*.npz")), 1):
        record = encode_records(model, [path], [0, 1, 2, 3], device, summary)[0]
        data = tensors([record], mean, std, device)
        values = {"baseline": record["logits"]}
        with torch.no_grad():
            for mode, head in heads.items():
                values[mode] = torch.cat([head(*(v[start:start + 256] for v in data[:3]))
                                          for start in range(0, len(record["y"]), 256)]).cpu().numpy()
        np.savez_compressed(output / path.name, y=record["y"], indices=record["indices"], **values)
        records.append({"record": path.name, "subject": record["subject"], "n_epochs": len(record["y"]),
                        "confusion": {mode: confusion(record["y"], logits.argmax(1)).tolist() for mode, logits in values.items()}})
        print(index, number, path.name, "complete", flush=True)
    pooled = {mode: cm_metrics(sum(np.array(r["confusion"][mode]) for r in records)) for mode in MODES}
    write_json(output / "comparison.json", {"protocol_sha256": digest, "seed": job["seed"], "fold": job["fold"],
               "device": torch.cuda.get_device_name(0), "records": records, "pooled": pooled})


def metric_arrays(cm):
    """Vectorized equivalents of metrics_from_confusion, last axes are truth/prediction."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    truth, predicted = cm.sum(-1), cm.sum(-2)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall = div(tp, predicted), div(tp, truth)
    f1 = div(2 * tp, truth + predicted)
    total = cm.sum((-2, -1)); accuracy = div(tp.sum(-1), total)
    chance = div((truth * predicted).sum(-1), total ** 2)
    return {"accuracy": accuracy, "macro_f1": f1.mean(-1), "kappa": div(accuracy - chance, 1 - chance),
            "n1_precision": precision[..., 1], "n1_recall": recall[..., 1], "n1_f1": f1[..., 1], "n3_f1": f1[..., 3]}


def summarize(args, protocol, digest):
    rows = [json.loads((args.output_dir / f"seed{j['seed']}" / f"fold_{j['fold']:02d}" / "comparison.json").read_text()) for j in protocol["jobs"]]
    assert len(rows) == 15 and all(r["protocol_sha256"] == digest and len(r["records"]) == 44 for r in rows)
    subjects = sorted({r["subject"] for r in rows[0]["records"]})
    matrices = {m: np.array([[sum(np.array(r["confusion"][m]) for r in row["records"] if r["subject"] == s)
                            for s in subjects] for row in rows]) for m in MODES}
    means = {m: {key: float(np.mean([r["pooled"][m][key] for r in rows])) for key in KEYS} for m in MODES}
    comparisons = {}
    draws = np.random.default_rng(20261010).integers(0, len(subjects), size=(5000, len(subjects)))
    sampled = {m: {k: [] for k in KEYS} for m in MODES}
    for start in range(0, len(draws), 100):
        for mode in MODES:
            values = metric_arrays(matrices[mode][:, draws[start:start + 100]].sum(2))
            for key in KEYS:
                sampled[mode][key].extend(values[key].mean(0).tolist())
    for control in MODES[:2]:
        comparisons[control] = {
            "mean_delta": {k: means['fixed_context'][k] - means[control][k] for k in KEYS},
            "paired_bootstrap_95ci": {k: np.quantile(np.array(sampled['fixed_context'][k]) - sampled[control][k], [.025, .975]).tolist() for k in KEYS},
            "seed_macro_delta": {str(seed): float(np.mean([r['pooled']['fixed_context']['macro_f1'] - r['pooled'][control]['macro_f1'] for r in rows if r['seed'] == seed])) for seed in (7777, 8888, 9999)},
        }
    checks = {"above_both_controls": all(c['mean_delta']['macro_f1'] >= .005 for c in comparisons.values()),
              "seed_consistency": all(sum(v > 0 for v in c['seed_macro_delta'].values()) >= 2 for c in comparisons.values())}
    for key, tolerance in (("accuracy", -.005), ("n1_recall", -.02), ("n1_f1", 0), ("n3_f1", -.01)):
        checks[key] = comparisons['baseline']['mean_delta'][key] >= tolerance
    ensembles = {}
    for name, indices in [(f"seed{s}_5model", [i for i, r in enumerate(rows) if r['seed'] == s]) for s in (7777, 8888, 9999)] + [("all15", list(range(15)))]:
        totals = {m: np.zeros((5, 5), dtype=int) for m in MODES}
        for record in rows[0]['records']:
            archives = []
            for i in indices:
                row = rows[i]
                with np.load(args.output_dir / f"seed{row['seed']}" / f"fold_{row['fold']:02d}" / record['record']) as data:
                    archives.append({k: data[k] for k in ('y', *MODES)})
            assert all(np.array_equal(a['y'], archives[0]['y']) for a in archives)
            for m in MODES:
                totals[m] += confusion(archives[0]['y'], np.mean([a[m] for a in archives], axis=0).argmax(1))
        ensembles[name] = {m: cm_metrics(cm) for m, cm in totals.items()}
    result = {"protocol_sha256": digest, "n_models": 15, "n_subjects": len(subjects), "n_records": 44,
              "n_epochs": sum(r['n_epochs'] for r in rows[0]['records']), "mean_model_metrics": means,
              "fixed_minus_controls": comparisons, "decision_checks": checks, "passes_locked_replication_checks": all(checks.values()),
              "secondary_ensembles": ensembles, "limitations": protocol['limitations']}
    write_json(args.output_dir / 'summary.json', result)
    with (args.output_dir / 'model_metrics.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['seed', 'fold', 'mode', *KEYS]); writer.writeheader()
        for row in rows:
            for mode in MODES:
                writer.writerow({'seed': row['seed'], 'fold': row['fold'], 'mode': mode, **row['pooled'][mode]})
    lines = ['# 固定上下文残差外部复核（2026-10-10）', '',
             '完整Sleep-EDF ST：22名受试者、44晚；15组冻结模型，GPU推理，无目标域训练/调参。',
             '主估计量：每组模型先在全队列pool，再对15组模型等权平均；不与历史20折集成绝对值作直接改进比较。', '',
             '| 方案 | Accuracy | Macro-F1 | N1 precision | N1 recall | N1 F1 | N3 F1 | Kappa |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in MODES:
        display_keys = ('accuracy', 'macro_f1', 'n1_precision', 'n1_recall', 'n1_f1', 'n3_f1', 'kappa')
        lines.append('| ' + ' | '.join([m, *[f'{means[m][k]:.4f}' for k in display_keys]]) + ' |')
    lines += ['', '## 配对变化（百分点）', '', '| 固定上下文减对照 | Macro-F1变化 [95%描述性区间] | N1 F1变化 [区间] | N3 F1变化 [区间] | 三种子Macro-F1变化 |', '|---|---|---|---|---|']
    for control, c in comparisons.items():
        cells = [control]
        for k in ('macro_f1', 'n1_f1', 'n3_f1'):
            low, high = c['paired_bootstrap_95ci'][k]
            cells.append(f"{100*c['mean_delta'][k]:+.3f} [{100*low:+.3f}, {100*high:+.3f}]")
        cells.append('/'.join(f'{100*v:+.3f}' for v in c['seed_macro_delta'].values()))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f"锁定复核条件全部通过：{all(checks.values())}；逐项检查：`{json.dumps(checks)}`。", '',
              '区间按受试者配对重采样5000次、两晚一起重采样；在每次draw内先pool再平均模型指标。',
              '这些是条件于现有模型集合的描述性区间，不包括整个研发选型不确定性；种子/模型不是独立人群样本。', '',
              '## 次要集成分析', '',
              '与上面的逐模型平均不同，以下对模型logits先取均值再评分。此分析训练人数、划分、模型数与历史20折主线不同，不应直接比较绝对值声称改进。', '',
              '| 集成 | 基线Macro-F1 | 固定上下文Macro-F1 | 变化（百分点） | 基线N1 F1 | 固定上下文N1 F1 | 基线N3 F1 | 固定上下文N3 F1 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, values in ensembles.items():
        b, f = values['baseline'], values['fixed_context']
        lines.append('| ' + ' | '.join([name, f"{b['macro_f1']:.4f}", f"{f['macro_f1']:.4f}",
                     f"{100*(f['macro_f1']-b['macro_f1']):+.3f}", f"{b['n1_f1']:.4f}", f"{f['n1_f1']:.4f}",
                     f"{b['n3_f1']:.4f}", f"{f['n3_f1']:.4f}"]) + ' |')
    lines += ['', '## 结论边界', '',
              '本轮预设主分析未复现内部整体收益：固定上下文相对基线的Macro-F1三个种子均下降，未达到复核条件。',
              'N1召回增加而精确率下降；N1 F1小幅增加但N3 F1下降。不能把N1召回增加等同于整体泛化改善。',
              '15模型集成的Macro-F1有小幅正向差值，但每种子集成结果不一致；这是次要估计量，不替代主分析。',
              '这些结果与源域时间规律或错误补偿不完全可迁移相容，但尚不能确定是EMG表示、队列特征还是模型误差分布所致。',
              '不根据这些外部结果继续搜索融合权重或checkpoint。暂不升级投稿主线。', '',
              '## 限制', '', *['- ' + v for v in protocol['limitations']], '',
              'ISRUC与SHHS原始数据盘未挂载，未完成其新方案验证。现有投稿主线不自动替换。',
              f'完整预测、逐模型CSV及次要集成指标：`{args.output_dir.relative_to(ROOT)}`。']
    (ROOT / 'docs/fixed_context_external_replication_20261010.md').write_text('\n'.join(lines) + '\n')
    write_json(args.output_dir / 'status.json', {'status': 'complete', 'protocol_sha256': digest})
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source-root', type=Path, default=ROOT / 'outputs/attnsleep_context_residual_study_20261010')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/sleepedf_st_multichannel_npz/eeg2_eog_emg_physical_volts')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/fixed_context_external_replication_20261010')
    p.add_argument('--prepare-only', action='store_true'); p.add_argument('--job', type=int)
    a = p.parse_args()
    protocol, digest = lock(a)
    if a.prepare_only:
        print('Locked', digest); return
    if a.job is not None:
        worker(a, protocol, digest, a.job); return
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two GPUs required; refusing CPU fallback')
    pending = queue.Queue()
    for i in range(15): pending.put(i)
    def run_queue(gpu):
        while True:
            try: i = pending.get_nowait()
            except queue.Empty: return
            logdir = a.output_dir / 'logs'; logdir.mkdir(exist_ok=True)
            with (logdir / f'job{i:02d}.log').open('a') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--source-root', str(a.source_root),
                                '--data-dir', str(a.data_dir), '--output-dir', str(a.output_dir), '--job', str(i)],
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1'},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print('completed job', i, flush=True)
    write_json(a.output_dir / 'status.json', {'status': 'running', 'pid': os.getpid(), 'protocol_sha256': digest})
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as executor:
            futures = [executor.submit(run_queue, gpu) for gpu in (0, 1)]
            for f in futures: f.result()
        summarize(a, protocol, digest)
    except Exception as e:
        write_json(a.output_dir / 'status.json', {'status': 'failed', 'error': repr(e), 'protocol_sha256': digest})
        raise


if __name__ == '__main__':
    main()

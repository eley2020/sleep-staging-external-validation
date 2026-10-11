"""Locked GPU-only minimal/fixed/progressive end-to-end growth diagnostic.

No architecture search. Three paired arms, 90 epochs, three seeds, five existing
development panels. Progressive insertion never depends on validation outcomes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'external_baselines/AttnSleep'))
from scripts.search_minimal_attnsleep_growth import DecomposedAttnSleep
from scripts.run_multipath_growth_study import Atom
from scripts.train_attnsleep_multichannel_baseline import load_arrays, set_random_seed
from scripts.evaluate_fixed_context_external import sha, confusion, cm_metrics
from scripts.summarize_attnsleep_improvement_study import subject_pairs, KEYS
from utils.util import calc_class_weight

OUT = ROOT/'outputs/end_to_end_growth_study_20261010'
SOURCE = ROOT/'outputs/attnsleep_selection_study_20261010/protocol.json'
METHODS = ('small', 'fixed', 'progressive')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


class MinimalGrowingSleep(DecomposedAttnSleep):
    """Historical minimal stem, with ordered identity-initialized atom slots."""
    def __init__(self):
        super().__init__(input_channels=4, atoms=())
        self.blocks = nn.ModuleList()

    def forward(self, x):
        z = deterministic_pool(self.features1(x), 30)
        z = self.proj(z).transpose(1, 2)
        for block in self.blocks:
            z = block(z)
        return self.fc(z.contiguous().flatten(1))

    def insert(self, seed):
        # Atom initialization cannot perturb the training/dropout RNG stream.
        device = next(self.parameters()).device
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(seed)
            atom = Atom('large').to(device)
        self.blocks.append(atom)


def deterministic_pool(x, output_size):
    # Same floor/ceil windows as adaptive_avg_pool1d. Its CUDA backward is
    # nondeterministic in this torch build; explicit slice means avoid atomics.
    length = x.shape[-1]
    return torch.stack([x[..., (i*length)//output_size:
                         ((i+1)*length+output_size-1)//output_size].mean(-1)
                        for i in range(output_size)], -1)


def depth_at(method, epoch):
    if not 1 <= epoch <= 90:
        raise ValueError(epoch)
    return 0 if method == 'small' else 2 if method == 'fixed' else (epoch-1)//30


def training_seed(seed, fold):
    return seed + 1009*fold


def atom_seed(seed, fold, index):
    return training_seed(seed, fold) + 100000*(index+1)


def make_model(seed, fold, depth):
    set_random_seed(training_seed(seed, fold))
    model = MinimalGrowingSleep().cuda()
    for index in range(depth):
        model.insert(atom_seed(seed, fold, index))
    return model


def choose_epoch(rows):
    eligible = [row for row in rows if 61 <= row['epoch'] <= 90]
    if not eligible:
        raise ValueError('No final-phase checkpoint')
    # Retain the previously chosen accuracy rule, not a new Macro-F1 search.
    return max(eligible, key=lambda r: (r['validation']['accuracy'], -r['epoch']))


def decision(table, seed_deltas):
    controls = ('small', 'fixed')
    checks = {'macro_vs_both': all(table['progressive']['macro_f1']-table[c]['macro_f1'] >= .005 for c in controls),
              'two_positive_seeds_vs_each': all(sum(v > 0 for v in seed_deltas[c].values()) >= 2 for c in controls)}
    for key, minimum in (('accuracy', -.005), ('n1_recall', -.02), ('n1_f1', 0.), ('n3_f1', -.01)):
        checks[key] = all(table['progressive'][key]-table[c][key] >= minimum for c in controls)
    return checks


def prepare():
    source = json.loads(SOURCE.read_text())
    code_paths = [Path(__file__), ROOT/'scripts/search_minimal_attnsleep_growth.py',
                  ROOT/'scripts/run_multipath_growth_study.py', ROOT/'scripts/train_attnsleep_multichannel_baseline.py',
                  ROOT/'external_baselines/AttnSleep/model/model.py', ROOT/'external_baselines/AttnSleep/utils/util.py',
                  ROOT/'scripts/evaluate_fixed_context_external.py', ROOT/'scripts/audit_attnsleep_generalization_gap.py',
                  ROOT/'scripts/summarize_attnsleep_improvement_study.py']
    data_dir = Path(source['data_dir'])
    protocol = {
        'study': 'From-minimal end-to-end fixed versus progressive atomic growth diagnostic',
        'date': '2026-10-10', 'source_protocol_sha256': sha(SOURCE),
        'code_sha256': {str(p): sha(p) for p in code_paths},
        'data_dir': str(data_dir), 'data_sha256': {p.name: sha(p) for p in sorted(data_dir.glob('*.npz'))},
        'panels': source['panels'], 'seeds': source['seeds'], 'methods': list(METHODS),
        'jobs': [{'seed': s, 'panel': p} for s in source['seeds'] for p in source['panels']],
        'base': 'Historical minimal DecomposedAttnSleep: four-channel early input fusion, features1 only, pool30, projection128->80, flatten2400->5; no quality gate/AFR/TCE/multiscale branch. Adaptive pooling uses mathematically equivalent explicit floor/ceil window means for deterministic CUDA backward, with FP32 rounding differences possible. Not the manuscript mainline.',
        'atom': 'Two identical ordered large-kernel31 depthwise residual blocks, LayerNorm80, GELU, linear80->80 zero-init output projection; no atom search or gate tuning.',
        'arms': {'small': 'Base90 epochs', 'fixed': 'Base plus two identity-init atoms from epoch1, all layers90 epochs',
                 'progressive': 'Base1-30; append atom1 before31; append atom2 before61; all current layers trained throughout. Inherit LAST epoch30/60 weights, never best-validation parent.'},
        'training': {'epochs': 90, 'phase_epochs': 30, 'batch_size': 64, 'optimizer': 'AdamW', 'lr': .0003,
                     'weight_decay': .001, 'clip': 5., 'precision': 'FP32', 'class_weights': 'Inherited calc_class_weight on fit subjects ONLY',
                     'optimizer_reset_before': [1,31,61], 'lr_schedule': 'constant same lr for every arm/phase',
                     'normalization': 'No added normalization; identical inherited physical-volts arrays; no augmentation/oversampling',
                     'rng': 'seed+1009*fold; paired shared base weights, epoch-keyed permutations; insertion fork_rng isolates dropout RNG; deterministic CUDA algorithms'},
        'selection': 'Accuracy, earliest tie, eligible epoch61-90 only for ALL arms. Equal30 candidate checkpoints and fixed/progressive final-depth2. All three decisions locked before assessment loading.',
        'budget': 'Equal90 epochs/optimizer updates/sample passes, not equal exact FLOPs. Record parameters/phase, measured synchronized train seconds, peak GPU memory, Conv1d/Linear forward MACs only (not total FLOPs). No search trajectories.',
        'assessment': 'Reuse locked13-fit/4-selection/2-assessment/1-deferred splits, five panels, three seeds, ten distinct assessment subjects. No outer-test loading. Reused internal development assessment, not untouched test.',
        'decision': {'macro_gain_vs_small_and_fixed': .005, 'positive_seed_means_each': 2,
                     'accuracy_min': -.005, 'n1_recall_min': -.02, 'n1_f1_min': 0., 'n3_f1_min': -.01},
        'statistics': 'Primary equal-weight15 panel-seed pooled metric means; separate descriptive5000 subject-paired bootstrap after averaging seed deltas; no confirmatory p-value.',
        'resume': 'Atomic epoch-end model/optimizer and Python/NumPy/CPU/CUDA RNG state. Exact same-machine algorithmic continuation; cumulative physical wall-time excludes interrupted unfinished epochs.',
        'limitations': ['Small early-fusion base is a mechanistic diagnostic, not manuscript quality-gated mainline.',
                        'One fixed atom type, two insertion points, and one fixed optimizer policy; no universal growth conclusion.',
                        'All arms reset AdamW moments at phase boundaries, a controlled training regime, not every conventional fixed-model recipe.',
                        'Equal sample passes are not equal FLOPs; reported forward MACs omit normalization/activation/pooling/backward operations.',
                        'Assessment subjects reused during development, fits overlap; three seeds are not independent population samples.',
                        'No external cohort validation or automatic manuscript replacement authorized by this diagnostic.']}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError('Locked code/data/protocol changed; refusing overwrite')
    else:
        save_json(path, protocol)
    return protocol, sha(path)


def verify(protocol):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU training/inference forbidden')
    for path, expected in protocol['code_sha256'].items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f'Locked implementation changed: {path}')
    for name, expected in protocol['data_sha256'].items():
        if sha(Path(protocol['data_dir'])/name) != expected:
            raise RuntimeError(f'Input changed: {name}')


def gpu_arrays(paths):
    x, y, *_ = load_arrays([str(p) for p in paths], [0,1,2,3])
    return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()


def evaluate(model, data):
    model.eval()
    x, y = data
    with torch.inference_mode():
        predictions = torch.cat([model(x[i:i+128]).argmax(1) for i in range(0,len(y),128)])
    cm = confusion(y.cpu().numpy(), predictions.cpu().numpy())
    return cm_metrics(cm), cm


def forward_macs(model, length):
    total = [0]
    def hook(module, args, output):
        if isinstance(module, nn.Conv1d):
            total[0] += output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
        elif isinstance(module, nn.Linear):
            total[0] += output.numel()*module.in_features
    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m,(nn.Conv1d,nn.Linear))]
    model.eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(1,4,length,device='cuda'))
    finally:
        for h in hooks:
            h.remove()
    return int(total[0])


def rng_snapshot():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda']])


def checkpoint_payload(model, optimizer, epoch, rows, digest, best, insertions, accounting):
    return {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'epoch': epoch, 'depth': len(model.blocks), 'rows': rows, 'rng': rng_snapshot(),
            'protocol_sha256': digest, 'best_epoch': best, 'insertions': insertions, 'accounting': accounting}


def train_arm(folder, method, job, fit, validation, weights, digest):
    arm = folder/method
    arm.mkdir(exist_ok=True)
    if (arm/'complete.json').exists():
        info = json.loads((arm/'complete.json').read_text())
        if info['protocol_sha256'] != digest or sha(arm/'selected.pt') != info['selected_sha256']:
            raise RuntimeError('Completed arm changed')
        return info
    last_path = arm/'last.pt'
    ck = torch.load(last_path, map_location='cuda', weights_only=False) if last_path.exists() else None
    completed = ck['epoch'] if ck else 0
    if ck and ck['protocol_sha256'] != digest:
        raise RuntimeError('Checkpoint mismatch')
    model = make_model(job['seed'], job['panel']['fold_id'], ck['depth'] if ck else depth_at(method,1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    rows, insertions, accounting, best = [], [], {}, None
    if ck:
        model.load_state_dict(ck['model_state']); optimizer.load_state_dict(ck['optimizer_state'])
        rows, insertions, accounting, best = ck['rows'], ck['insertions'], ck['accounting'], ck['best_epoch']
        restore_rng(ck['rng'])
        if completed in (30,60,90) and not (arm/f'phase_end_{completed:02d}.pt').exists():
            save_torch(arm/f'phase_end_{completed:02d}.pt',ck)
        del ck
    criterion = nn.CrossEntropyLoss(weight=weights)
    x, y = fit
    for epoch in range(completed+1,91):
        while len(model.blocks) < depth_at(method,epoch):
            model.eval()
            with torch.inference_mode():
                before = model(x[:8]).clone()
            index = len(model.blocks)
            model.insert(atom_seed(job['seed'],job['panel']['fold_id'],index))
            with torch.inference_mode():
                after = model(x[:8])
            if not torch.equal(before,after):
                raise RuntimeError('Insertion did not preserve parent output exactly')
            insertions.append({'before_epoch': epoch, 'index': index, 'max_abs_delta': 0.,
                               'parent_epoch': epoch-1, 'parent_last_sha256': sha(last_path)})
        if epoch in (1,31,61):
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
        phase = str((epoch-1)//30)
        if phase not in accounting:
            accounting[phase] = {'depth': len(model.blocks), 'parameters': sum(p.numel() for p in model.parameters()),
                                 'conv_linear_forward_macs_per_sample': forward_macs(model,x.shape[-1])}
        generator = torch.Generator(device='cuda').manual_seed(training_seed(job['seed'],job['panel']['fold_id'])+epoch*7919)
        order = torch.randperm(len(y), generator=generator, device='cuda')
        order_sha = hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.perf_counter()
        model.train(); loss_sum=torch.zeros((),device='cuda'); steps=0
        for start in range(0,len(y),64):
            indices=order[start:start+64]; optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x[indices]),y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            loss_sum += loss.detach()*len(indices); steps+=1
        torch.cuda.synchronize(); train_seconds=time.perf_counter()-started
        val_started=time.perf_counter(); metrics, cm=evaluate(model,validation)
        torch.cuda.synchronize(); validation_seconds=time.perf_counter()-val_started
        row={'epoch': epoch, 'phase': int(phase), 'depth': len(model.blocks), 'train_loss': float(loss_sum/len(y)),
             'steps': steps, 'sample_passes': len(y), 'order_sha256': order_sha,
             'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
             'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
             'validation': metrics, 'validation_confusion_matrix': cm.tolist()}
        rows.append(row)
        if epoch>=61 and choose_epoch(rows)['epoch']==epoch:
            best=epoch
            save_torch(arm/'selected.pt',{'model_state': model.state_dict(), 'epoch': epoch,
                                        'depth': len(model.blocks), 'protocol_sha256': digest, 'validation': metrics})
        payload=checkpoint_payload(model,optimizer,epoch,rows,digest,best,insertions,accounting)
        save_torch(last_path,payload)
        if epoch in (30,60,90):
            save_torch(arm/f'phase_end_{epoch:02d}.pt',payload)
        save_json(arm/'trajectory.json',rows)
        save_json(folder/'progress.json',{'status':'training', 'method':method, 'epoch':epoch, 'protocol_sha256':digest})
        print(f"{method} epoch{epoch:02d}/90 depth{len(model.blocks)} train={train_seconds:.1f}s acc={metrics['accuracy']:.4f} macro={metrics['macro_f1']:.4f} N1={metrics['n1_f1']:.4f} N3={metrics['n3_f1']:.4f}",flush=True)
    choice=choose_epoch(rows)
    if best!=choice['epoch']:
        raise RuntimeError('Saved checkpoint differs from locked selection rule')
    info={'protocol_sha256':digest,'method':method,'epoch':best,'depth':len(model.blocks),
          'selected_sha256':sha(arm/'selected.pt'),'validation':choice['validation'],
          'accounting':accounting,'insertions':insertions,'completed_epochs':90,
          'optimizer_steps':sum(r['steps'] for r in rows),'sample_passes':sum(r['sample_passes'] for r in rows),
          'train_seconds':sum(r['train_seconds'] for r in rows),'validation_seconds':sum(r['validation_seconds'] for r in rows)}
    save_json(arm/'complete.json',info)
    del model,optimizer; torch.cuda.empty_cache()
    return info


def run_job(protocol,digest,index):
    verify(protocol); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    job=protocol['jobs'][index]; panel=job['panel']
    folder=OUT/f"seed{job['seed']}"/f"fold_{panel['fold_id']:02d}"
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'job.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'comparison.json').exists():
            if json.loads((folder/'comparison.json').read_text())['protocol_sha256']!=digest:
                raise RuntimeError('Comparison digest changed')
            return
        paths=sorted(Path(protocol['data_dir']).glob('*.npz'))
        fit=gpu_arrays([p for p in paths if p.name[3:5] in panel['fit']])
        validation=gpu_arrays([p for p in paths if p.name[3:5] in panel['selection']])
        counts=np.bincount(fit[1].cpu().numpy(),minlength=5).tolist()
        weights=torch.tensor(calc_class_weight(counts),device='cuda',dtype=torch.float32)
        choices={m:train_arm(folder,m,job,fit,validation,weights,digest) for m in METHODS}
        if choices['fixed']['accounting']['2']!=choices['progressive']['accounting']['2']:
            raise RuntimeError('Final architectures/cost differ')
        if len({choices[m]['optimizer_steps'] for m in METHODS})!=1:
            raise RuntimeError('Optimizer budgets differ')
        save_json(folder/'locked_selection.json',{'protocol_sha256':digest,'choices':choices,
                  'assessment_not_loaded_at_lock':True,'fit_subjects':panel['fit'],'selection_subjects':panel['selection'],
                  'fit_class_counts':counts,'class_weights':weights.cpu().tolist()})
        del fit,validation; torch.cuda.empty_cache()
        outcomes={m:[] for m in METHODS}
        models={}
        for m in METHODS:
            model=make_model(job['seed'],panel['fold_id'],choices[m]['depth'])
            ck=torch.load(folder/m/'selected.pt',map_location='cuda',weights_only=False)
            model.load_state_dict(ck['model_state']); models[m]=model.eval()
        for path in paths:
            if path.name[3:5] not in panel['assessment']:
                continue
            data=gpu_arrays([path]); archive={'y':data[1].cpu().numpy()}
            for m,model in models.items():
                metrics,cm=evaluate(model,data)
                outcomes[m].append({'subject':path.name[3:5],'record':path.name,'confusion_matrix':cm.tolist()})
                archive[m+'_confusion_matrix']=cm
            np.savez_compressed(folder/(path.stem+'_assessment.npz'),**archive)
        results={m:{'records':r,'pooled':cm_metrics(sum(np.array(v['confusion_matrix']) for v in r))} for m,r in outcomes.items()}
        save_json(folder/'comparison.json',{'protocol_sha256':digest,'seed':job['seed'],'panel':panel,'choices':choices,
                  'device':torch.cuda.get_device_name(0),'outcomes':results})
        save_json(folder/'progress.json',{'status':'complete','protocol_sha256':digest})


def summarize(protocol,digest):
    rows=[json.loads((OUT/f"seed{j['seed']}"/f"fold_{j['panel']['fold_id']:02d}"/'comparison.json').read_text()) for j in protocol['jobs']]
    if len(rows)!=15 or any(r['protocol_sha256']!=digest for r in rows):
        raise RuntimeError('Missing/mismatched jobs')
    table={m:{k:float(np.mean([r['outcomes'][m]['pooled'][k] for r in rows])) for k in KEYS} for m in METHODS}
    flat=[{'seed':r['seed'],**r['outcomes']} for r in rows]
    pairs={c:subject_pairs(flat,c,'progressive') for c in ('small','fixed')}
    deltas={c:{str(s):float(np.mean([r['outcomes']['progressive']['pooled']['macro_f1']-r['outcomes'][c]['pooled']['macro_f1'] for r in rows if r['seed']==s])) for s in protocol['seeds']} for c in pairs}
    checks=decision(table,deltas)
    save_json(OUT/'summary.json',{'protocol_sha256':digest,'table':table,'results':rows,
              'subject_paired_comparisons':pairs,'seed_macro_deltas':deltas,'decision_checks':checks,
              'passes_internal_checks':all(checks.values()),'limitations':protocol['limitations']})
    with (OUT/'metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['method',*KEYS]); writer.writeheader()
        writer.writerows({'method':m,**v} for m,v in table.items())
    lines=['# 端到端渐进生长：三组内部开发对照','',
           '3种子×5划分；每组90 epoch，30+30+30。渐进组在31/61 epoch插入大核31原子块，继续训练全部层。',
           '所有组仅从61–90 epoch按Accuracy/最早平局选检查点。固定/渐进组最终结构一致。','',
           '| 模型 | Accuracy | Macro-F1 | N1 F1 | N3 F1 |','|---|---:|---:|---:|---:|']
    for m,v in table.items():
        lines.append('| '+' | '.join([m,*[f'{v[k]:.5f}' for k in ('accuracy','macro_f1','n1_f1','n3_f1')]])+' |')
    lines+=['',f'预设内部条件全部通过：{all(checks.values())}；逐项：`{json.dumps(checks)}`。','',
            '渐进组必须优于相同最终结构fixed组，才能支持此设置下的生长过程收益；仅优于small不足以支持。',
            '阴性结果不证明所有生长方法无效，也不能确定上一轮失败是冻结主干导致。',
            '受试者配对区间和逐种子差异见summary.json。主表是15组pooled指标等权均值；bootstrap是先平均种子后受试者等权，估计量不同。','',
            '| 模型 | 累计实测训练秒 | 每组样本遍历epoch |','|---|---:|---:|']
    for m in METHODS:
        lines.append(f"| {m} | {sum(r['choices'][m]['train_seconds'] for r in rows):.1f} | 90 |")
    lines+=['','## 限制','',*['- '+v for v in protocol['limitations']], '', '不自动替换稿件主线。']
    (ROOT/'docs/end_to_end_growth_results_20261010.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
    keys=('accuracy','macro_f1','n1_f1','n3_f1')
    for ax,(control,result) in zip(axes,pairs.items()):
        values=np.array([result['statistics'][k]['mean_subject_delta'] for k in keys])*100
        ci=np.array([result['statistics'][k]['descriptive_bootstrap_95ci'] for k in keys])*100
        ax.errorbar(range(4),values,yerr=np.maximum(np.stack([values-ci[:,0],ci[:,1]-values]),0),fmt='o',capsize=4)
        ax.axhline(0,color='gray'); ax.set_xticks(range(4),['Accuracy','Macro-F1','N1 F1','N3 F1'],rotation=20)
        ax.set_title('Progressive minus '+control)
    axes[0].set_ylabel('Subject-paired change (percentage points)'); fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'paired_changes.{ext}',dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--job',type=int)
    args=parser.parse_args()
    if args.job is not None:
        protocol=json.loads((OUT/'protocol.json').read_text())
        run_job(protocol,sha(OUT/'protocol.json'),args.job); return
    protocol,digest=prepare()
    print('Protocol locked:',digest,flush=True)
    if args.prepare_only:
        return
    if torch.cuda.device_count()<2:
        raise RuntimeError('Two GPUs required, no CPU fallback')
    with (OUT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=queue.Queue()
        for index in range(len(protocol['jobs'])):
            pending.put(index)
        def work(gpu):
            while True:
                try: index=pending.get_nowait()
                except queue.Empty: return
                logs=OUT/'logs'; logs.mkdir(exist_ok=True)
                with (logs/f'job{index:02d}.log').open('a') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',str(index)],cwd=ROOT,
                                   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1',
                                        'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
                print('job',index,'complete',flush=True)
        save_json(OUT/'status.json',{'status':'running','pid':os.getpid(),'protocol_sha256':digest})
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(work,gpu) for gpu in (0,1)]
                for future in futures:
                    future.result()
            summarize(protocol,digest)
            save_json(OUT/'status.json',{'status':'complete','protocol_sha256':digest})
        except Exception as error:
            save_json(OUT/'status.json',{'status':'failed','error':repr(error),'protocol_sha256':digest})
            raise


if __name__=='__main__':
    main()

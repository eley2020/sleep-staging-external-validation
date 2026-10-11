"""Budget-matched greedy/beam/random atomic growth on a frozen CUDA backbone.

This is a finite, ordered-state search, not exact dynamic programming. Each
algorithm is charged three first-stage and six second-stage 30-epoch trials;
identical trials may be physically cached, never counted as free evaluations.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'external_baselines/AttnSleep'))
from scripts.evaluate_fixed_context_external import sha, confusion, cm_metrics
from scripts.run_attnsleep_selection_study import write_json
from scripts.train_attnsleep_multichannel_baseline import set_random_seed, load_arrays

OUT = ROOT/'outputs/multipath_growth_study_20261010'
SOURCE = ROOT/'outputs/attnsleep_selection_study_20261010'
ATOMS = ('local', 'large', 'geglu')
METHODS = ('no_growth', 'greedy', 'beam', 'random')
GUARDS = {'accuracy':.005, 'n1_recall':.02, 'n1_f1':.005, 'n3_f1':.01, 'kappa':.005}


class Atom(nn.Module):
    """Dimension-preserving residual; zero output projection gives identity."""
    def __init__(self, kind, dim=80):
        super().__init__()
        if kind not in ATOMS:
            raise ValueError(kind)
        self.kind = kind
        self.norm = nn.LayerNorm(dim)
        if kind in ('local','large'):
            kernel = 7 if kind=='local' else 31
            self.temporal = nn.Conv1d(dim, dim, kernel, padding=kernel//2, groups=dim)
            self.output = nn.Linear(dim,dim)
        else:
            self.input = nn.Linear(dim,80)
            self.output = nn.Linear(40,dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x):
        z = self.norm(x)
        if self.kind=='geglu':
            value, gate = self.input(z).chunk(2,-1)
            z = value * torch.nn.functional.gelu(gate)
        else:
            z = torch.nn.functional.gelu(self.temporal(z.transpose(1,2)).transpose(1,2))
        return x + self.output(z)


class GrowingHead(nn.Module):
    def __init__(self, atoms=()):
        super().__init__()
        self.atoms = tuple(atoms)
        self.blocks = nn.ModuleList([Atom(a) for a in atoms])
        self.hidden = nn.Sequential(nn.Linear(165,32),nn.GELU(),nn.Dropout(.2))
        self.output = nn.Linear(32,5)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, tokens, logits):
        for block in self.blocks: tokens=block(tokens)
        features=torch.cat((tokens.mean(1),tokens.std(1,unbiased=False),logits),-1)
        return logits + torch.tanh(self.output(self.hidden(features)))

    def grow(self, atom):
        result=copy.deepcopy(self)
        result.blocks.append(Atom(atom).to(next(self.parameters()).device))
        result.atoms=self.atoms+(atom,)
        return result


def safe(metrics, reference):
    return all(metrics[k]>=reference[k]-v for k,v in GUARDS.items())


def rank(node):
    return (node['validation']['macro_f1'], -node['parameters'], node['validation']['accuracy'], -node['order'])


def best(nodes, reference):
    eligible=[n for n in nodes if safe(n['validation'],reference)]
    if not eligible: raise RuntimeError('Missing safe fallback')
    return max(eligible,key=rank)


def beam_parents(nodes, reference):
    eligible=[n for n in nodes if safe(n['validation'],reference)]
    frontier=[]
    for node in eligible:
        score=node['validation']['macro_f1']; cost=node['parameters']
        dominated=any((other['validation']['macro_f1']>=score and other['parameters']<=cost)
                      and (other['validation']['macro_f1']>score or other['parameters']<cost) for other in eligible)
        if not dominated: frontier.append(node)
    ranked=sorted(frontier,key=rank,reverse=True)
    selected=ranked[:2]
    # A near-best dominated route is exploratory, explicitly not Pareto optimal.
    if len(selected)<2:
        threshold=best(eligible,reference)['validation']['macro_f1']-.005
        fill=sorted([n for n in eligible if n not in selected and n['validation']['macro_f1']>=threshold],key=rank,reverse=True)
        selected+=fill[:2-len(selected)]
    return selected or [best(nodes,reference)]


def expansion_plan(parents):
    if not 1<=len(parents)<=2 or len(set(parents))!=len(parents):
        raise ValueError('One or two distinct parent states required')
    assignments=[(parents[0],0),(parents[0],1)] if len(parents)==1 else [(p,0) for p in parents]
    return [(parent,a,repeat) for parent,repeat in assignments for a in ATOMS]


def prepare():
    source=json.loads((SOURCE/'protocol.json').read_text())
    jobs=[]
    for seed in source['seeds']:
        for panel in source['panels']:
            folder=SOURCE/f'seed{seed}'/f"fold_{panel['fold_id']:02d}"
            locked=json.loads((folder/'locked_selection.json').read_text())
            ep=locked['rules']['accuracy']['epoch']
            checkpoint=folder/'selection_trajectory'/f'epoch_{ep:02d}.pt'
            jobs.append({'seed':seed,'panel':panel,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint)})
    protocol={
        'study':'Finite multi-path atomic growth on a frozen full mainline; internal development pilot',
        'date':'2026-10-10','source_protocol_sha256':sha(SOURCE/'protocol.json'),
        'data_dir':source['data_dir'],'input_inventory':source['input_inventory'],'jobs':jobs,
        'seeds':source['seeds'],'methods':list(METHODS),'atoms':list(ATOMS),'max_added_depth':2,
        'state':'ordered atom tuple AND parent weight checkpoint AND trial RNG; repeated atoms allowed; no merging by unordered atom sets',
        'backbone':'four-channel quality-gated AttnSleep, existing 30-epoch source checkpoint, frozen/eval; original 2400-D classifier retained',
        'representation':'30 intra-epoch TCE tokens x 80 dimensions; train-fit-only per-channel token normalization; no inter-epoch context',
        'head':'token mean/std plus original5 logits -> hidden32/dropout0.2 -> zero-init5; original logits+tanh(residual)',
        'atom':'local depthwise7, large depthwise31, or GEGLU hidden40; LayerNorm and zero-init output projection; identity at insertion',
        'training':{'epochs_per_trial':30,'root_epochs':30,'batch_size':256,'optimizer':'AdamW','lr':.0003,'weight_decay':.001,'clip':5.0,'loss':'source-fit class-weighted cross-entropy'},
        'budget':'Each method: common root30, three stage1 trials30, six stage2 trials30. All shared/cache trials charged. Equal optimizer-update/data-pass budgets, NOT exact FLOPs/wall-time budgets; measured cost reported.',
        'greedy':'common three first atoms, pick best safe parent among root+first stage, then three children x two training RNG replicates; retain all fallback nodes',
        'beam':'width2 Pareto Macro-F1 vs trainable parameters, fill with safe routes <=0.005 below best if necessary; expand all3 atoms per parent. If only1 eligible parent, second allocation is independent replicates, never phantom new paths',
        'random':'same three first atoms, choose two distinct eligible parents by seeded random permutation independent of ranking, expand all3; one-parent case same replicate policy',
        'no_growth':'three stage1 continuations of root without adding atoms, choose best safe parent, six stage2 no-atom continuations; equal9 trial budget and epoch trajectory selection',
        'selection':{'guards_vs_frozen_validation':GUARDS,'checkpoint':'include epoch0, highest Macro-F1 among safe epochs; then accuracy/earliest; final architecture tie favors fewer parameters. All method locks written before assessment access'},
        'assessment':'same historical five panels:13 fitting,4 selection,2 assessment,1 outer deferred; ten distinct assessment subjects. Previously inspected development data, not pristine test',
        'decision':{'macro_gain_beam_vs_greedy_random_no_growth':.005,'positive_seed_means_each':2,'accuracy_min':-.005,'n1_recall_min':-.02,'n1_f1_min':0,'n3_f1_min':-.01},
        'statistics':'descriptive subject-paired bootstrap5000, average seed deltas within subject; no confirmatory p-value, overlapping fits and reused assessment',
        'limitations':['Approximate beam/Pareto search, no Bellman/global-optimum guarantee.','Frozen-backbone pilot, not from-scratch minimal network or end-to-end growth.','Equal trial/data budgets do not make exact FLOP budgets equal.','Existing Sleep-EDF development reuse and limited subjects constrain inference.','No external labels used, outer subjects deferred; no automatic manuscript replacement.'],
    }
    OUT.mkdir(exist_ok=True,parents=True); path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol: raise RuntimeError('Changed locked growth protocol')
    else: write_json(path,protocol)
    return protocol,sha(path)


def encode(model, paths):
    records=[]; model.eval()
    with torch.inference_mode():
        for path in paths:
            x,y,*_=load_arrays([str(path)],[0,1,2,3])
            tokens=[]; logits=[]
            for start in range(0,len(y),128):
                batch=torch.as_tensor(x[start:start+128],device='cuda')
                z=model.tce(model.mrcnn(model.channel_gate(batch)))
                l=model.fc(z.contiguous().flatten(1))
                tokens.append(z.cpu().numpy()); logits.append(l.cpu().numpy())
            records.append({'record':path.name,'subject':path.name[3:5],'tokens':np.concatenate(tokens),
                            'logits':np.concatenate(logits),'y':y})
    return records


def tensor_data(records,mean,std):
    return tuple(torch.as_tensor(np.concatenate([r[key] for r in records]),device='cuda') for key in ('tokens','logits','y')) if mean is None else (
        torch.as_tensor((np.concatenate([r['tokens'] for r in records])-mean)/std,device='cuda'),
        torch.as_tensor(np.concatenate([r['logits'] for r in records]),device='cuda'),
        torch.as_tensor(np.concatenate([r['y'] for r in records]),device='cuda'))


def evaluate(head,data):
    head.eval(); tokens,logits,y=data
    with torch.no_grad():
        predicted=torch.cat([head(tokens[i:i+256],logits[i:i+256]).argmax(1) for i in range(0,len(y),256)]).cpu().numpy()
    return cm_metrics(confusion(y.cpu().numpy(),predicted))


def rng_seed(job,key):
    digest=hashlib.sha256(f"{job['seed']}:{job['panel']['fold_id']}:{key}".encode()).digest()
    return int.from_bytes(digest[:4],'little') % (2**31-1)


def run_job(protocol,digest,index):
    from model.model import AttnSleep
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
    job=protocol['jobs'][index]; panel=job['panel']
    folder=OUT/f"seed{job['seed']}"/f"fold_{panel['fold_id']:02d}"; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'comparison.json').exists():
        assert json.loads((folder/'comparison.json').read_text())['protocol_sha256']==digest
        return
    # No arbitrary continuation of unfinished stochastic training.
    if (folder/'started.json').exists(): raise RuntimeError('Partial growth trajectory; refusing stochastic resume/overwrite')
    write_json(folder/'started.json',{'protocol_sha256':digest,'pid':os.getpid()})
    if sha(Path(job['checkpoint']))!=job['checkpoint_sha256']: raise RuntimeError('Source checkpoint changed')
    for row in protocol['input_inventory']:
        path=Path(protocol['data_dir'])/row['name']; stat=path.stat()
        if stat.st_size!=row['bytes'] or stat.st_mtime_ns!=row['mtime_ns']:
            raise RuntimeError('Source data inventory changed: '+str(path))
    ck=torch.load(job['checkpoint'],map_location='cuda',weights_only=False)
    assert ck['metadata']['model_train_subjects']==panel['fit']
    assert ck['metadata']['checkpoint_selection_subjects']==panel['selection']
    encoder=AttnSleep(input_channels=4,channel_fusion='quality-gated',channel_dropout=.15).cuda().eval()
    encoder.load_state_dict(ck['model_state']); backbone_parameters=sum(p.numel() for p in encoder.parameters())
    paths=sorted(Path(protocol['data_dir']).glob('*.npz'))
    fit=encode(encoder,[p for p in paths if p.name[3:5] in panel['fit']])
    validation=encode(encoder,[p for p in paths if p.name[3:5] in panel['selection']])
    all_fit=np.concatenate([r['tokens'] for r in fit]); mean=all_fit.mean((0,1)); std=np.maximum(all_fit.std((0,1)),1e-6)
    del all_fit
    np.savez_compressed(folder/'token_normalization.npz',mean=mean,std=std)
    train=tensor_data(fit,mean,std); val=tensor_data(validation,mean,std)
    del fit,validation
    baseline=cm_metrics(confusion(val[-1].cpu().numpy(),val[1].argmax(1).cpu().numpy()))
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(ck['metadata']['class_weights'],device='cuda'))
    cache={}; nodes={}; accounting={}; started=time.time()

    def trial(parent,atom,key):
        if key in cache: return cache[key]
        set_random_seed(rng_seed(job,key+':initialization'))
        model=GrowingHead().cuda() if parent is None else copy.deepcopy(nodes[parent]['model'])
        if atom is not None: model=model.grow(atom)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
        initial=evaluate(model,val); history=[{'epoch':0,'validation':initial}]
        if not safe(initial,baseline): raise RuntimeError('Insertion violated safe identity fallback')
        chosen=history[0]; state=copy.deepcopy(model.state_dict()); updates=0; t0=time.time()
        for epoch in range(1,31):
            seed=rng_seed(job,key+f':epoch{epoch}')
            set_random_seed(seed)
            order=torch.randperm(len(train[-1]),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
            model.train(); losses=[]
            for start in range(0,len(order),256):
                ix=order[start:start+256]; optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(train[0][ix],train[1][ix]),train[2][ix]); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); updates+=1
                losses.append(loss.detach())
            outcome=evaluate(model,val); row={'epoch':epoch,'validation':outcome}; history.append(row)
            if safe(outcome,baseline) and (outcome['macro_f1'],outcome['accuracy'],-epoch)>(chosen['validation']['macro_f1'],chosen['validation']['accuracy'],-chosen['epoch']):
                chosen=row; state=copy.deepcopy(model.state_dict())
            print(index,key,epoch,round(float(torch.stack(losses).mean()),5),round(outcome['macro_f1'],5),flush=True)
            progress={'status':'training','protocol_sha256':digest,'trial':key,'epoch':epoch,
                      'completed_unique_trials':len(nodes),'elapsed_seconds':time.time()-started}
            write_json(folder/'progress.partial.json',progress); (folder/'progress.partial.json').replace(folder/'progress.json')
        model.load_state_dict(state); model.eval()
        node={'key':key,'parent':parent,'atom':atom,'atoms':list(model.atoms),'epoch':chosen['epoch'],
              'validation':chosen['validation'],'parameters':sum(p.numel() for p in model.parameters()),
              'total_parameters':backbone_parameters+sum(p.numel() for p in model.parameters()),
              'optimizer_updates':updates,'epochs':30,'seconds':time.time()-t0,'order':len(nodes),'model':model}
        nodes[key]=node; cache[key]=key
        write_json(folder/(key+'_trajectory.json'),history)
        torch.save({'model_state':state,'atoms':model.atoms,'selected_epoch':chosen['epoch'],'protocol_sha256':digest},folder/(key+'.pt'))
        return key

    root=trial(None,None,'root')
    growth_first=[trial(root,a,'first_'+a) for a in ATOMS]
    first_nodes=[nodes[k] for k in [root,*growth_first]]
    no_first=[trial(root,None,'no_first_'+str(i)) for i in range(3)]
    greedy_parent=best(first_nodes,baseline)['key']
    beam=[n['key'] for n in beam_parents(first_nodes,baseline)]
    eligible=[n['key'] for n in first_nodes if safe(n['validation'],baseline)]
    random=list(np.random.default_rng(rng_seed(job,'random_parents')).permutation(eligible))[:2]
    no_parent=best([nodes[k] for k in [root,*no_first]],baseline)['key']

    def expand(parents):
        # Exactly six trials, including independent repeats for one-parent routes.
        return [trial(parent,a,'child_'+parent+'_'+a+'_r'+str(repeat)) for parent,a,repeat in expansion_plan(parents)]

    children={'greedy':expand([greedy_parent]),'beam':expand(beam),'random':expand(random),
              'no_growth':[trial(no_parent,None,'no_child_'+str(i)) for i in range(6)]}
    selected={}
    for method in METHODS:
        first=no_first if method=='no_growth' else growth_first
        pool=[nodes[k] for k in [root,*first,*children[method]]]
        selected[method]=best(pool,baseline)['key']
        charged=[root,*first,*children[method]]
        accounting[method]={'root_trials':1,'search_trials':9,'charged_epochs':300,
            'charged_optimizer_updates':sum(nodes[k]['optimizer_updates'] for k in charged),
            'charged_seconds_sum':sum(nodes[k]['seconds'] for k in charged),'trial_keys':charged,
            'parent_routes': [no_parent] if method=='no_growth' else {'greedy':[greedy_parent],'beam':beam,'random':random}[method]}
    if len({v['charged_optimizer_updates'] for v in accounting.values()})!=1: raise RuntimeError('Unmatched data-pass budget')
    serial={k:{kk:vv for kk,vv in v.items() if kk!='model'} for k,v in nodes.items()}
    write_json(folder/'locked_selection.json',{'protocol_sha256':digest,'selected':selected,'nodes':serial,'accounting':accounting})
    # No assessment signals/labels read above this lock.
    assessment=encode(encoder,[p for p in paths if p.name[3:5] in panel['assessment']])
    outcomes={m:[] for m in ('frozen_baseline','root',*METHODS)}
    for record in assessment:
        data=tensor_data([record],mean,std)
        for method in outcomes:
            if method=='frozen_baseline': predicted=record['logits'].argmax(1)
            else:
                key=root if method=='root' else selected[method]; head=nodes[key]['model'].eval()
                with torch.no_grad(): predicted=torch.cat([head(data[0][i:i+256],data[1][i:i+256]).argmax(1) for i in range(0,len(data[-1]),256)]).cpu().numpy()
            outcomes[method].append({'record':record['record'],'subject':record['subject'],
                'confusion_matrix':confusion(record['y'],predicted).tolist()})
    results={m:{'records':rs,'pooled':cm_metrics(sum(np.array(r['confusion_matrix']) for r in rs))} for m,rs in outcomes.items()}
    write_json(folder/'comparison.json',{'protocol_sha256':digest,'seed':job['seed'],'panel':panel,
        'device':torch.cuda.get_device_name(0),'selected':selected,'accounting':accounting,'nodes':serial,
        'unique_physical_trials':len(nodes),'wall_seconds':time.time()-started,'outcomes':results})
    write_json(folder/'progress.json',{'status':'complete','completed_unique_trials':len(nodes),'protocol_sha256':digest})


def summarize(protocol,digest):
    from scripts.summarize_attnsleep_improvement_study import subject_pairs,KEYS
    rows=[json.loads((OUT/f"seed{j['seed']}"/f"fold_{j['panel']['fold_id']:02d}"/'comparison.json').read_text()) for j in protocol['jobs']]
    assert len(rows)==15 and all(r['protocol_sha256']==digest for r in rows)
    flattened=[{'seed':r['seed'],**r['outcomes']} for r in rows]
    modes=('frozen_baseline','root',*METHODS)
    table={m:{k:float(np.mean([r['outcomes'][m]['pooled'][k] for r in rows])) for k in KEYS} for m in modes}
    comparisons={c:subject_pairs(flattened,c,'beam') for c in ('greedy','random','no_growth')}
    seed_deltas={c:{str(s):float(np.mean([r['outcomes']['beam']['pooled']['macro_f1']-r['outcomes'][c]['pooled']['macro_f1'] for r in rows if r['seed']==s])) for s in (7777,8888,9999)} for c in comparisons}
    checks={'above_all_controls':all(table['beam']['macro_f1']-table[c]['macro_f1']>=.005 for c in comparisons),
            'seed_consistency':all(sum(v>0 for v in d.values())>=2 for d in seed_deltas.values())}
    for key,tolerance in (('accuracy',-.005),('n1_recall',-.02),('n1_f1',0),('n3_f1',-.01)):
        checks[key]=all(table['beam'][key]-table[c][key]>=tolerance for c in comparisons)
    selection_counts={m:{} for m in METHODS}
    for row in rows:
        for method in METHODS:
            label='>'.join(row['nodes'][row['selected'][method]]['atoms']) or 'no atoms'
            selection_counts[method][label]=selection_counts[method].get(label,0)+1
    write_json(OUT/'summary.json',{'protocol_sha256':digest,'results':rows,'table':table,'paired_subject_comparisons':comparisons,
        'selected_structure_counts':selection_counts,
        'seed_macro_deltas':seed_deltas,'decision_checks':checks,'passes_internal_pilot_checks':all(checks.values()),'limitations':protocol['limitations']})
    with (OUT/'metrics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['method',*KEYS]); writer.writeheader()
        for method,values in table.items(): writer.writerow({'method':method,**values})
    lines=['# 多路径原子网络生长：内部开发实验','',
        '双GPU、3种子、5组受试者划分。原子块/残差头每条训练30 epochs；主干冻结。每方法9条搜索试验及共同root30，预算按共享候选也计费。','',
        '| 方法 | Accuracy | Macro-F1 | N1 F1 | N3 F1 |','|---|---:|---:|---:|---:|']
    for m,v in table.items(): lines.append('| '+' | '.join([m,*[f'{v[k]:.4f}' for k in ('accuracy','macro_f1','n1_f1','n3_f1')]])+' |')
    lines+=['',f'预设内部条件全部通过：{all(checks.values())}。条件：`{json.dumps(checks)}`。','',
        '| beam减对照 | Macro-F1变化（百分点） | N1 F1变化 | N3 F1变化 | 3种子Macro-F1变化 |','|---|---:|---:|---:|---|']
    for control in comparisons:
        lines.append('| '+' | '.join([control,*[f"{100*(table['beam'][k]-table[control][k]):+.3f}" for k in ('macro_f1','n1_f1','n3_f1')],
                     '/'.join(f'{100*v:+.3f}' for v in seed_deltas[control].values())])+' |')
    lines+=['','## 受试者配对描述性区间','',
        '| beam减对照 | Macro-F1受试者等权平均变化及95%区间（百分点） | N1 F1平均及区间 | N3 F1平均及区间 |',
        '|---|---|---|---|']
    for control,comparison in comparisons.items():
        cells=[control]
        for key in ('macro_f1','n1_f1','n3_f1'):
            statistic=comparison['statistics'][key]; lo,hi=statistic['descriptive_bootstrap_95ci']
            cells.append(f"{100*statistic['mean_subject_delta']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}]")
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','区间先在各受试者平均3种子差值，再重采样10名受试者；受试者等权估计量不同于上表fold-seed pooled均值。','',
        '## 结构选择与预算','',f'各方法最终有序结构计数：`{json.dumps(selection_counts)}`。','',
        '| 方法 | 每组计费试验数（含root） | 每组计费epoch | 15组计费累计训练耗时（秒） | 平均选定头参数量 |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        seconds=sum(r['accounting'][method]['charged_seconds_sum'] for r in rows)
        parameters=np.mean([r['nodes'][r['selected'][method]]['parameters'] for r in rows])
        lines.append(f'| {method} | 10 | 300 | {seconds:.1f} | {parameters:.0f} |')
    lines+=['','计费时间为包含共享候选的各试验时间之和，不是并行流水线墙钟时间；相同数据更新次数不代表相同FLOPs。',
        f"实际唯一物理训练轨迹共{sum(r['unique_physical_trials'] for r in rows)}条，均30 epochs；共享缓存只减少执行成本，不增大策略可见候选预算。",'',
        '完整有序结构、选型epoch、带计费候选清单、时间/参数量及受试者配对区间见summary.json；总体指标另存metrics.csv。','',
        '## 方法学结论','',
        ('beam通过预设内部条件，但只作为后续独立确认候选，不升级稿件。' if all(checks.values()) else
         'beam未通过预设内部条件。不能将它优于某个单一对照或某个种子的微小变化解释为多路径机制有效；暂不继续扩大本轮候选池，也不升级稿件。'),
        f"beam在{selection_counts['beam'].get('no atoms',0)}/15组选择不保留原子块，表明回退机制确实被使用。该选择来自validation，不是根据assessment事后删块。",
        '本轮不能确定是候选表达力、冻结主干、优化过程还是选型噪声限制了收益；也不能据此证明所有生长网络或动态规划启发搜索无效。','',
        '## 限制','',*['- '+v for v in protocol['limitations']], '',
        '受试者重复开发使用，训练集重叠，种子不是独立人口样本；区间仅描述性。无新外部结果，稿件不自动替换。']
    (ROOT/'docs/multipath_growth_results_20261010.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    keys=('accuracy','macro_f1','n1_f1','n3_f1')
    for axis,(control,comparison) in zip(axes,comparisons.items()):
        values=np.array([comparison['statistics'][k]['mean_subject_delta'] for k in keys])*100
        ci=np.array([comparison['statistics'][k]['descriptive_bootstrap_95ci'] for k in keys])*100
        axis.errorbar(range(4),values,yerr=np.maximum(np.stack((values-ci[:,0],ci[:,1]-values)),0),fmt='o',capsize=4)
        axis.axhline(0,color='gray',linewidth=1); axis.set_title('Beam minus '+control)
        axis.set_xticks(range(4),['Accuracy','Macro-F1','N1 F1','N3 F1'],rotation=25)
    axes[0].set_ylabel('Subject-paired change (percentage points)'); fig.tight_layout()
    for extension in ('png','pdf'): fig.savefig(OUT/f'paired_growth_changes.{extension}',dpi=220)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--prepare-only',action='store_true'); p.add_argument('--job',type=int); a=p.parse_args()
    if a.job is not None:
        protocol=json.loads((OUT/'protocol.json').read_text()); run_job(protocol,sha(OUT/'protocol.json'),a.job); return
    protocol,digest=prepare()
    if a.prepare_only: print('Locked',digest); return
    if torch.cuda.device_count()<2: raise RuntimeError('Two GPUs required, no CPU fallback')
    with (OUT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=queue.Queue()
        for i in range(len(protocol['jobs'])): pending.put(i)
        def work(gpu):
            while True:
                try: i=pending.get_nowait()
                except queue.Empty: return
                log_dir=OUT/'logs'; log_dir.mkdir(exist_ok=True)
                with (log_dir/f'job{i:02d}.log').open('a') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',str(i)],
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'},
                        stdout=log,stderr=subprocess.STDOUT,check=True)
                print('job',i,'complete',flush=True)
        write_json(OUT/'status.json',{'status':'running','pid':os.getpid(),'protocol_sha256':digest})
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(work,gpu) for gpu in (0,1)]
                for f in futures:f.result()
            summarize(protocol,digest); write_json(OUT/'status.json',{'status':'complete','protocol_sha256':digest})
        except Exception as e:
            write_json(OUT/'status.json',{'status':'failed','protocol_sha256':digest,'error':repr(e)}); raise


if __name__=='__main__': main()

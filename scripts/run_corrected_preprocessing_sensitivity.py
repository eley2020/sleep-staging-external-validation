"""New locked 2x2 pilot after ST repair; preserve the halted original pilot."""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.run_mainline_preprocessing_sensitivity as old
from scripts.resume_st_repair_cuda import check_parity
from scripts.audit_attnsleep_generalization_gap import metrics_from_confusion

OUT=ROOT/'outputs/mainline_preprocessing_sensitivity_corrected_20261011'
REPAIR=ROOT/'outputs/st_baseline_repair_20261011'
DATA=ROOT/'data/sleepedf_st_multichannel_npz/eeg2_eog_emg_physical_volts_repaired_20261011'
REPORT=ROOT/'docs/mainline_preprocessing_sensitivity_corrected_20261011.md'
COHORTS=('st','isruc','shhs1','shhs2')
MODES=old.MODES
KEYS=(*old.KEYS,'n3_precision','n3_recall')
save_json=old.save_json
sha=old.sha

def metrics(cm):
    value=metrics_from_confusion(cm)
    return {**old.cm_metrics(cm),'n3_precision':value['class_precision'][3],'n3_recall':value['class_recall'][3]}

def prepare():
    audit=json.loads((REPAIR/'audit_status.json').read_text())
    assert audit['status']=='passed' and audit['all44_raw_parity_verified']
    previous=ROOT/'outputs/mainline_preprocessing_sensitivity_20261011/protocol.json'
    p=copy.deepcopy(json.loads(previous.read_text()))
    p['study']='Corrected ST baseline, frozen original20-fold 2x2 preprocessing sensitivity; new post-hoc pilot'
    p['prior_halted_protocol_sha256']=sha(previous)
    p['repair_audit_sha256']=sha(REPAIR/'audit_status.json')
    p['repair_protocol_sha256']=sha(REPAIR/'protocol.json')
    p['repair_amendment_sha256']=sha(REPAIR/'cuda_parity_amendment.json')
    p['repaired_data_summary_sha256']=sha(DATA/'summary.json')
    for row in p['records']:
        if row['cohort']=='st':
            row['npz']=str(DATA/(row['record']+'.npz'))
            row['historical']=str(REPAIR/'predictions'/(row['record']+'_predictions.npz'))
        row['input_inventory']={};row['small_artifact_sha256']={}
        for key in ('signal','annotation','npz','historical'):
            if key not in row:continue
            path=Path(row[key])
            if not path.exists():
                if key=='historical':row['historical_available']=False;continue
                raise FileNotFoundError(path)
            st=path.stat();row['input_inventory'][key]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
            if key!='signal':row['small_artifact_sha256'][key]=sha(path)
    code=[Path(__file__),Path(old.__file__),ROOT/'scripts/resume_st_repair_cuda.py',
          ROOT/'scripts/audit_attnsleep_generalization_gap.py',ROOT/'scripts/evaluate_fixed_context_external.py',
          *map(Path,p['code_sha256'])]
    p['code_sha256']={str(f.resolve()):sha(f) for f in code}
    p['output_dir']=str(OUT);p['report_path']=str(REPORT)
    p['parity']='ST original x/y/indices exactly match repaired cache; original logits allclose atol=rtol=1e-5 vs sameCUDA repaired baseline, predictions exact. Other cohorts original x/y exactly match inherited readers and historical classes must match; no tolerance relaxation without separate explicit amendment.'
    p['input_guard']='Reject nonfinite or entirely zero native channels and prepared epochs. Only missing-channel KeyError is an exclusion; other input/parity failure stops study. Preserve requested records, never replace missing records.'
    p['summary_metrics']=list(KEYS)
    p['baseline_scope']='Only repaired ST historical original scores; never compare a modified input with invalid zero-filled ST cache. No reuse of incomplete old pilot outcomes.'
    p['limits'].append('Specific ST input-cache integrity incident repaired and independently audited before pilot; this does not establish universal data correctness.')
    OUT.mkdir(exist_ok=True,parents=True);path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=p:raise RuntimeError('New locked protocol changed')
    else:save_json(path,p)
    return p,sha(path)

def verify(p):
    for kind in ('code_sha256','model_sha256'):
        for path,expected in p[kind].items():
            if sha(Path(path))!=expected:raise RuntimeError('Locked file changed '+path)
    for key,path in (('repair_audit_sha256',REPAIR/'audit_status.json'),('repair_protocol_sha256',REPAIR/'protocol.json'),
                     ('repair_amendment_sha256',REPAIR/'cuda_parity_amendment.json'),('repaired_data_summary_sha256',DATA/'summary.json')):
        assert sha(path)==p[key]

def load_inputs(row):
    # Original reader reads and closes PSG before opening annotations.
    arrays,y,indices,meta=old.load_inputs(row)
    for mode,x in arrays.items():
        if not np.isfinite(x).all() or np.any(np.all(x==0,axis=(1,2))):raise RuntimeError('Invalid prepared signals '+mode)
        if np.any(np.all(x==0,axis=(0,2))):raise RuntimeError('Entire native/prepared channel zero '+mode)
    if row['cohort']=='st':
        with np.load(row['npz']) as d:assert np.array_equal(indices,d['indices'])
        assert np.array_equal(arrays['emg_envelope'],arrays['both'])
    return arrays,y,indices,meta

def worker(p,digest,shard):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
    verify(p);torch.set_num_threads(2);model,params,buffers=old.models(p)
    for row in p['records'][shard::2]:
        folder=OUT/row['cohort'];folder.mkdir(exist_ok=True)
        target=folder/(row['record']+'.json');archive=folder/(row['record']+'.npz')
        if target.exists():
            value=json.loads(target.read_text());assert value['protocol_sha256']==digest
            if 'error' in value or (archive.exists() and sha(archive)==value['archive_sha256']):continue
            raise RuntimeError('Invalid resume artifact')
        started=time.time()
        save_json(OUT/f'worker{shard}_status.json',{'status':'processing','cohort':row['cohort'],'record':row['record']})
        try:arrays,y,indices,meta=load_inputs(row)
        except KeyError as error:
            if not str(error).startswith("\"Missing channels ") and not str(error).startswith("'Missing channels "):raise
            save_json(target,{'protocol_sha256':digest,'record':row['record'],'subject':row['subject'],'error':repr(error),'exclusion':'missing channel, no substitute'})
            print('Missing channel exclusion',row['cohort'],row['record'],error,flush=True);continue
        logits={};input_same={}
        for mode in MODES:
            equivalent=next((m for m in logits if np.array_equal(arrays[mode],arrays[m])),None)
            input_same[mode]=equivalent
            logits[mode]=logits[equivalent] if equivalent else old.predict_ensemble_logits(model,params,buffers,arrays[mode],64,[{'kind':'identity'}],torch.device('cuda'))
            if not np.isfinite(logits[mode]).all():raise RuntimeError('Nonfinite logits')
        mismatch=None;logit_parity=None
        if Path(row['historical']).exists():
            with np.load(row['historical']) as hist:
                if not np.array_equal(y,hist['y_true']):raise RuntimeError('Historical labels differ')
                mismatch=int((logits['original'].argmax(1)!=hist['y_pred']).sum())
                if mismatch:raise RuntimeError(f'Historical original class mismatch {row["record"]}: {mismatch}')
                if row['cohort']=='st':logit_parity=check_parity(logits['original'],hist['logits'])
        cms={m:old.confusion(y,v.argmax(1)).tolist() for m,v in logits.items()}
        temp=archive.with_suffix('.partial.npz');np.savez_compressed(temp,y=y.astype(np.int8),indices=indices,**logits);temp.replace(archive)
        save_json(target,{'protocol_sha256':digest,'record':row['record'],'subject':row['subject'],'n_epochs':len(y),'metadata':meta,
            'reader_exact_parity':True,'historical_prediction_mismatches':mismatch,'historical_logit_parity':logit_parity,
            'input_equivalences':input_same,'confusion':cms,'archive_sha256':sha(archive),'seconds':time.time()-started,'device':torch.cuda.get_device_name(0)})
        print(row['cohort'],row['record'],len(y),'done',round(time.time()-started,1),flush=True)
    save_json(OUT/f'worker{shard}_status.json',{'status':'complete'})

def bootstrap(cms):
    rng=np.random.default_rng(20261011);n=len(cms['original']);draws={m:{k:[] for k in KEYS} for m in MODES[1:]}
    for _ in range(5000):
        ids=rng.integers(0,n,size=n);base=metrics(cms['original'][ids].sum(0))
        for mode in draws:
            value=metrics(cms[mode][ids].sum(0))
            for k in KEYS:draws[mode][k].append(value[k]-base[k])
    return {m:{k:np.quantile(v,[.025,.975]).tolist() for k,v in data.items()} for m,data in draws.items()}

def summarize(p,digest):
    verify(p);cohorts={};csvrows=[]
    for cohort in COHORTS:
        expected=[r for r in p['records'] if r['cohort']==cohort]
        rows=[json.loads((OUT/cohort/(r['record']+'.json')).read_text()) for r in expected]
        valid=[r for r in rows if 'error' not in r];assert valid and all(r['protocol_sha256']==digest for r in rows)
        if cohort in ('st','isruc'):assert len(valid)==len(expected)
        subjects=sorted({r['subject'] for r in valid})
        cms={m:np.stack([sum(np.asarray(r['confusion'][m]) for r in valid if r['subject']==s) for s in subjects]) for m in MODES}
        table={m:metrics(v.sum(0)) for m,v in cms.items()};deltas={m:{k:table[m][k]-table['original'][k] for k in KEYS} for m in MODES[1:]}
        ci=bootstrap(cms)
        cohorts[cohort]={'requested':len(expected),'records':len(valid),'subjects':len(subjects),'n_epochs':sum(r['n_epochs'] for r in valid),
            'exclusions':[r for r in rows if 'error' in r],'table':table,'deltas':deltas,'paired_subject_bootstrap_delta_95ci':ci,
            'confusion':{m:cm.sum(0).tolist() for m,cm in cms.items()},
            'subject_gain_counts':{m:{k:sum(metrics(cms[m][i])[k]>metrics(cms['original'][i])[k] for i in range(len(subjects))) for k in ('macro_f1','n1_f1','n3_f1')} for m in MODES[1:]},
            'records_with_historical_parity':sum(r['historical_prediction_mismatches']==0 for r in valid)}
        for mode,data in table.items():
            for k in KEYS:
                csvrows.append({'cohort':cohort,'mode':mode,'metric':k,'estimate':data[k],'delta_vs_corrected_original':0. if mode=='original' else deltas[mode][k],
                    'paired_delta_ci_low':None if mode=='original' else ci[mode][k][0],'paired_delta_ci_high':None if mode=='original' else ci[mode][k][1]})
    gates={m:old.decision(cohorts,m) for m in MODES[1:]}
    save_json(OUT/'summary.json',{'protocol_sha256':digest,'cohorts':cohorts,'resource_gates':gates,'limitations':p['limits']})
    with (OUT/'metrics_and_paired_deltas.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    lines=['# 修正ST基线后的冻结主线2×2预处理敏感性','',
        '原20-fold冻结模型，双TITAN V/FP32，无训练/调参。原始ST模式只与已修正基线比较，不与失效的zero-filled缓存比较。全部3个修改模式报告，不改稿件主线。','',
        '| 队列 | 模式 | Accuracy | Macro-F1 | N1 precision | N1 recall | N1 F1 | N3 recall | N3 F1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort,value in cohorts.items():
        for mode,data in value['table'].items():lines.append('| '+' | '.join([cohort,mode,*[f'{data[k]:.5f}' for k in ('accuracy','macro_f1','n1_precision','n1_recall','n1_f1','n3_recall','n3_f1')]])+' |')
    lines+=['','## 推进决策','', '预设规则不变：至少两个非随访队列Macro-F1≥+0.005，必须有ST或ISRUC；所有队列Macro-F1/Accuracy≥−0.005，N1 recall≥−0.02，N1/N3 F1≥−0.005。SHHS2只检查安全性，不算独立成功。门槛是探索性资源规则，不是统计确认。','']
    for mode,g in gates.items():lines.append(f'- {mode}: passes_resource_gate={g["passes_resource_gate"]}; {json.dumps(g)}')
    lines+=['','## 请求与实际纳入','']
    for cohort,value in cohorts.items():lines.append(f'- {cohort}: {value["requested"]}请求，{value["records"]}记录/{value["subjects"]}受试者/{value["n_epochs"]}epochs；排除{len(value["exclusions"])}，历史原模式逐epoch一致{value["records_with_historical_parity"]}。')
    lines+=['','ST100Hz original/antialias完全相同，EMG/both完全相同，作为负对照。固定HP16Hz/rectify/LP0.4Hz/1Hzrepeat；抗混叠仅对>100Hz原波形，均full-record滤波，不匹配幅值。',
        '5000次受试者配对bootstrap保持ST两夜一起，区间条件于固定源模型。SHHS每波仅32条预设样本，不能称全库性能。多个模式/队列/终点无确认性多重检验，不做显著性或新盲测声明。',
        '原源域checkpoint来自一个训练种子/20折，不是3个独立种子；零相位滤波仅离线，继承标注裁剪不是无标签实时部署。','',*['- '+v for v in p['limits']],
        '', '不自动启动全SHHS/重新训练，不将此次结果替换稿件主模型。']
    REPORT.write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,mode in zip(axes,MODES[1:]):
        for k in ('macro_f1','n1_f1','n3_f1'):
            estimates=np.array([100*cohorts[c]['deltas'][mode][k] for c in COHORTS])
            bounds=100*np.array([cohorts[c]['paired_subject_bootstrap_delta_95ci'][mode][k] for c in COHORTS])
            ax.plot(range(4),estimates,'o-',label=k)
            ax.vlines(range(4),bounds[:,0],bounds[:,1],alpha=.6)
        ax.axhline(0,color='gray');ax.set_xticks(range(4),['ST','ISRUC','SHHS1 sample','SHHS2 sample'],rotation=25);ax.set_title(mode);ax.legend()
    axes[0].set_ylabel('Change vs corrected baseline (percentage points)');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'sensitivity_changes.{ext}',dpi=220)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--worker',type=int);a=parser.parse_args()
    if a.worker is not None:
        p=json.loads((OUT/'protocol.json').read_text());worker(p,sha(OUT/'protocol.json'),a.worker);return
    p,digest=prepare();verify(p);print('Locked corrected pilot',digest,flush=True)
    if a.prepare_only:return
    if torch.cuda.device_count()<2:raise RuntimeError('Two GPUs required')
    with (OUT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save_json(OUT/'status.json',{'status':'running','pid':os.getpid(),'protocol_sha256':digest})
        def launch(g):
            with (OUT/f'worker{g}.log').open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(g)],cwd=ROOT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(g),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(launch,g) for g in (0,1)]
                for f in futures:f.result()
            save_json(OUT/'status.json',{'status':'summarizing','pid':os.getpid(),'protocol_sha256':digest})
            summarize(p,digest);save_json(OUT/'status.json',{'status':'complete_pending_audit','protocol_sha256':digest})
        except Exception as error:
            save_json(OUT/'status.json',{'status':'failed','error':repr(error),'protocol_sha256':digest});raise

if __name__=='__main__':main()

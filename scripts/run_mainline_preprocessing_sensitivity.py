"""Frozen original20-fold ensemble, fixed2x2 preprocessing sensitivity on CUDA.

Post-hoc external development analysis; no training or target-driven tuning.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyedflib
from scipy import signal
import torch
from torch.func import stack_module_state

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'external_baselines/AttnSleep'))
from model.model import AttnSleep
from scripts.prepare_sleepedf20_multichannel import physical_unit_scale_to_volts,resample_epoch,read_labels,UNKNOWN
from scripts.evaluate_attnsleep_shhs_ensemble import read_labels as shhs_labels,load_record as load_shhs
from scripts.evaluate_attnsleep_isruc import resolve_channel,labels_from_txt,load_record as load_isruc
from scripts.evaluate_attnsleep_external_ensemble import predict_ensemble_logits
from scripts.evaluate_fixed_context_external import confusion,cm_metrics
from scripts.run_end_to_end_growth_study import save_json,sha

OUT=ROOT/'outputs/mainline_preprocessing_sensitivity_20261011'
MODEL=ROOT/'outputs/attnsleep_multichannel_baseline_train/full20_quality_gate_4ch_emg_30epoch'
STDATA=ROOT/'data/sleepedf_st_multichannel_npz/eeg2_eog_emg_physical_volts'
MODES=('original','antialias','emg_envelope','both')
KEYS=('accuracy','macro_f1','kappa','n1_precision','n1_recall','n1_f1','n3_f1')


def antialias_epochs(native,sfreq):
    if sfreq<=100:
        return np.stack([resample_epoch(x,3000) for x in native]).astype(np.float32)
    frac=Fraction(100/sfreq).limit_denominator(10000)
    # Complete native record avoids introducing a new filter boundary each30s.
    out=signal.resample_poly(native.reshape(-1),frac.numerator,frac.denominator,
                             window=('kaiser',5.0),padtype='constant')
    if len(out)!=len(native)*3000: raise ValueError('Unexpected resample length')
    return out.reshape(len(native),3000).astype(np.float32)


def envelope_epochs(native,sfreq):
    # Not a source reconstruction: original acquisition transfer is unknown.
    if sfreq/2<=16: raise ValueError('Waveform EMG cannot support fixed16Hz highpass')
    flat=native.reshape(-1).astype(np.float64)
    filtered=signal.sosfiltfilt(signal.butter(4,16,btype='highpass',fs=sfreq,output='sos'),flat)
    low=signal.sosfiltfilt(signal.butter(4,.4,btype='lowpass',fs=sfreq,output='sos'),np.abs(filtered))
    low=np.maximum(low,0.)
    ratio=Fraction(1/sfreq).limit_denominator(10000)
    onehz=signal.resample_poly(low,ratio.numerator,ratio.denominator,window=('kaiser',5.0),padtype='constant')
    if len(onehz)!=len(native)*30: raise ValueError('Unexpected envelope length')
    return np.repeat(np.maximum(onehz,0.).reshape(len(native),30),100,axis=1).astype(np.float32)


def modes_from_native(native_channels,rates):
    existing=[np.stack([resample_epoch(x,3000) for x in c]).astype(np.float32) for c in native_channels]
    aa=[antialias_epochs(c,fs) for c,fs in zip(native_channels,rates)]
    envelope=envelope_epochs(native_channels[3],rates[3])
    return {'original':existing,'antialias':aa,'emg_envelope':[*existing[:3],envelope],
            'both':[*aa[:3],envelope]}


def prepare():
    audit=json.loads((ROOT/'outputs/mainline_domain_input_audit_20261011/protocol.json').read_text())
    raw=json.loads((ROOT/'outputs/fixed_context_raw_external_20261010/protocol.json').read_text())
    summary=json.loads((STDATA/'summary.json').read_text())
    records=[]
    for row in summary['records']:
        stem=row['psg_file'].replace('-PSG.edf','')
        records.append({'cohort':'st','record':stem,'subject':stem[3:5],
            'signal':str((ROOT/summary['raw_dir']/row['psg_file']).resolve()),
            'annotation':str((ROOT/summary['raw_dir']/row['hypnogram_file']).resolve()),
            'npz':str(STDATA/(stem+'.npz')),
            'historical':str(ROOT/'outputs/external_validation/sleepedf_st/quality_gated_attnsleep_20fold_frozen_ensemble_no_tta'/f'{stem}_predictions.npz')})
    for cohort in ('isruc','shhs1','shhs2'):
        wanted={r['record'] for r in audit['jobs'] if r['cohort']==cohort}
        for row in raw['cohorts'][cohort]['requested']:
            if row['record'] not in wanted: continue
            directory=ROOT/'outputs/external_validation'
            history=(directory/'isruc_cohort1_zero_shot_x1_final'/f"{row['subject']}.npz" if cohort=='isruc'
                else directory/'shhs'/f'frozen_ensemble_identity_{cohort}'/'record_predictions'/f"{row['record']}.npz")
            records.append({'cohort':cohort,'record':row['record'],'subject':row['subject'],
                            'signal':row['signal']['path'],'annotation':row['annotation']['path'],
                            'historical':str(history)})
    checkpoint_paths=[MODEL/f'fold_{i:02d}/checkpoint_best.pt' for i in range(20)]
    code=[Path(__file__),ROOT/'scripts/prepare_sleepedf20_multichannel.py',ROOT/'scripts/evaluate_attnsleep_isruc.py',
          ROOT/'scripts/evaluate_attnsleep_shhs_ensemble.py',ROOT/'scripts/evaluate_attnsleep_external_ensemble.py',
          ROOT/'external_baselines/AttnSleep/model/model.py']
    for row in records:
        row['input_inventory']={}
        row['small_artifact_sha256']={}
        for key in ('signal','annotation','npz','historical'):
            if key not in row: continue
            path=Path(row[key])
            if not path.exists():
                if key=='historical': row['historical_available']=False; continue
                raise FileNotFoundError(path)
            stat=path.stat(); row['input_inventory'][key]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            if key!='signal': row['small_artifact_sha256'][key]=sha(path)
    p={'date':'2026-10-11','study':'Original manuscript frozen20-fold2x2 preprocessing sensitivity; bounded post-hoc pilot',
       'model_sha256':{str(f):sha(f) for f in checkpoint_paths},'code_sha256':{str(f):sha(f) for f in code},
       'audit_protocol_sha256':sha(ROOT/'outputs/mainline_domain_input_audit_20261011/protocol.json'),
       'records':records,'modes':list(MODES),
       'scope':'All44 ST/22subjects,100ISRUC,32predeclared evenly spaced matched SHHS EDFs each wave. Missing channels recorded and never replaced. No source input/model changes.',
       'antialias':'Only waveform channels native fs>100Hz: full-record resample_poly target100, Kaiser5, constant boundary. No changes when fs<=100; existing repeat/interp retained.',
       'emg':'Native full-record fourth-channel: Butterworth4 highpass16Hz sosfiltfilt, abs rectification, Butterworth4 lowpass0.4Hz sosfiltfilt, polyphase to1Hz Kaiser5, clip negative numerical/filter ripple, repeat100. Source envelope unchanged. Offline zero-phase, not causal deployment.',
       'emg_limits':'Approximate amplitude representation, not exact source acquisition reconstruction.0.4Hz chosen below1Hz Nyquist; header LP0.7Hz cannot be replicated without alias. No target RMS scaling, fitting or cutoff search.',
       'both':'Antialias EEG1/EEG2/EOG; EMG envelope directly from native waveform (NOT from already downsampled EMG).',
       'labels':'Same fixed inherited mapping/removal/crop as original. Filters use full native timelines before annotation crop; target labels only used for inherited benchmark crop and scoring. No label-driven filtering.',
       'parity':'Original waveform arrays exactly compared with inherited readers, ST npz x/y identical; original predictions compared with archived20-fold frozen predictions where available. Any changed predicted epoch is fatal, not a new result.',
       'inference':'2record-sharded TITAN V workers;20fixed eval-mode checkpoints;mean logits,FP32,batch64;noTTA/BNupdates/newheads/tuning/CPUfallback.',
       'decision':{'each_variant_predeclared':True,'minimum_macro_gain_per_cohort':.005,'minimum_cohorts_passing':2,
                   'required_pass_cohort':'st or isruc','other_cohort_macro_floor':-.005,'accuracy_floor':-.005,
                   'n1_recall_floor':-.02,'n1_f1_floor':-.005,'n3_f1_floor':-.005,
                   'interpretation':'All3modifiedvariants reported; no best variant selection. Thresholds are exploratory resource-decision rules, not confirmatory tests. SHHS2 never counts as independent passing cohort, but safety checked.'},
       'bootstrap':'5000 descriptive paired subject resamples within each cohort;bothST nights together. Same draws for all modes,conditional on fixed20-model ensemble;no cohort pooling or p-value.',
       'limits':['Historically inspected external labels; not pristine validation.','SHHS32record sample each wave, not full-cohort performance.',
                 'Original20 models are one source seed, not3independent training seeds; source fits overlap.',
                 'Offline filters and existing annotation crops are not label-free causal clinical deployment.',
                 'No automatic manuscript replacement, no new training or fullSHHS launch.']}
    OUT.mkdir(parents=True,exist_ok=True); path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=p: raise RuntimeError('Locked protocol changed')
    if not path.exists(): save_json(path,p)
    return p,sha(path)


def load_inputs(row):
    for k,inventory in row['input_inventory'].items():
        stat=Path(row[k]).stat()
        if inventory!={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}: raise RuntimeError('Input changed')
    for k,expected in row['small_artifact_sha256'].items():
        if sha(Path(row[k]))!=expected: raise RuntimeError('Input checksum changed')
    reader=pyedflib.EdfReader(row['signal'])
    try:
        if row['cohort']=='st': names=['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental']
        elif row['cohort']=='isruc': names=[resolve_channel(reader,k,'X1','auto') for k in ('eeg_c3','eeg_c4','eog','emg')]
        else: names=['EEG(sec)','EEG','EOG(L)','EMG']
        missing=[name for name in names if name not in reader.getSignalLabels()]
        if missing: raise KeyError('Missing channels '+repr(missing))
        indices=[reader.getSignalLabels().index(n) for n in names]
        rates=[float(reader.getSampleFrequency(i)) for i in indices]
        n_epochs=min(int(reader.getNSamples()[i]//(fs*30)) for i,fs in zip(indices,rates))
        native=[]
        for i,fs in zip(indices,rates):
            n=int(round(fs*30)); values=reader.readSignal(i).astype(np.float32)*physical_unit_scale_to_volts(reader.getPhysicalDimension(i))
            native.append(values[:n_epochs*n].reshape(n_epochs,n))
    finally: reader.close()
    if any(not np.isfinite(c).all() for c in native): raise ValueError('Nonfinite input')
    if row['cohort']=='st': y_all=read_labels(Path(row['annotation']),n_epochs)
    elif row['cohort']=='isruc': y_all=labels_from_txt(Path(row['annotation']),n_epochs)
    else: y_all=shhs_labels(Path(row['annotation']),n_epochs)
    valid=np.flatnonzero(y_all!=UNKNOWN)
    if row['cohort']!='isruc':
        sleep=np.flatnonzero(y_all[valid]!=0)
        if not len(sleep): raise ValueError('No sleep')
        valid=valid[max(int(sleep[0])-60,0):min(int(sleep[-1])+61,len(valid))]
    modes=modes_from_native(native,rates)
    arrays={m:np.stack([c[valid] for c in channels],1).astype(np.float32) for m,channels in modes.items()}
    y=y_all[valid].astype(np.int64)
    if row['cohort']=='st':
        with np.load(row['npz'],allow_pickle=True) as old:
            if not np.array_equal(old['x'],arrays['original']) or not np.array_equal(old['y'],y): raise RuntimeError('ST reader parity failed')
    elif row['cohort']=='isruc':
        old_x,old_y,_=load_isruc(Path(row['signal']),Path(row['annotation']),'X1','auto','physical-volts')
        if not np.array_equal(old_x,arrays['original']) or not np.array_equal(old_y,y): raise RuntimeError('ISRUC reader parity failed')
    else:
        old_x,old_y,_=load_shhs(Path(row['signal']),Path(row['annotation']),30)
        if not np.array_equal(old_x,arrays['original']) or not np.array_equal(old_y,y): raise RuntimeError('SHHS reader parity failed')
    if row['cohort']=='st' and not np.array_equal(arrays['original'],arrays['antialias']): raise RuntimeError('100HzST antialias must be identity')
    return arrays,y,valid,{'channels':names,'native_hz':rates,'n_signal_epochs':n_epochs}


def models(p):
    for path,expected in p['model_sha256'].items():
        if sha(Path(path))!=expected: raise RuntimeError('Frozen model changed')
    instances=[]
    for path in p['model_sha256']:
        ck=torch.load(path,map_location='cuda',weights_only=False)
        instance=AttnSleep(input_channels=4,channel_fusion='quality-gated',channel_dropout=.15).cuda().eval()
        instance.load_state_dict(ck['model_state']); instances.append(instance)
    params,buffers=stack_module_state(instances)
    return instances[0],params,buffers


def worker(p,digest,shard):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA only')
    torch.set_num_threads(2)
    for path,expected in p['code_sha256'].items():
        if sha(Path(path))!=expected: raise RuntimeError('Code changed')
    base,params,buffers=models(p)
    for row in p['records'][shard::2]:
        folder=OUT/row['cohort']; folder.mkdir(exist_ok=True)
        target=folder/(row['record']+'.json'); archive=folder/(row['record']+'.npz')
        if target.exists():
            value=json.loads(target.read_text()); assert value['protocol_sha256']==digest
            if 'error' in value or (archive.exists() and sha(archive)==value['archive_sha256']): continue
            raise RuntimeError('Invalid resume artifacts')
        started=time.time()
        try: arrays,y,indices,meta=load_inputs(row)
        except (KeyError,ValueError,OSError) as e:
            save_json(target,{'protocol_sha256':digest,'record':row['record'],'subject':row['subject'],'error':repr(e)})
            print('Input exclusion',row['cohort'],row['record'],repr(e),flush=True); continue
        logits={}
        for mode in MODES:
            equivalent=next((m for m in logits if np.array_equal(arrays[mode],arrays[m])),None)
            logits[mode]=(logits[equivalent] if equivalent else predict_ensemble_logits(base,params,buffers,arrays[mode],64,[{'kind':'identity'}],torch.device('cuda')))
            if not np.isfinite(logits[mode]).all(): raise RuntimeError('Nonfinite logits')
        mismatch=None
        if Path(row['historical']).exists():
            with np.load(row['historical']) as old:
                if not np.array_equal(y,old['y_true']): raise RuntimeError('Historical labels differ')
                mismatch=int((logits['original'].argmax(1)!=old['y_pred']).sum())
                if mismatch: raise RuntimeError(f'Historical original predictions differ: {mismatch}')
        cms={m:confusion(y,v.argmax(1)).tolist() for m,v in logits.items()}
        temp=archive.with_suffix('.partial.npz'); np.savez_compressed(temp,y=y.astype(np.int8),indices=indices,**logits); temp.replace(archive)
        save_json(target,{'protocol_sha256':digest,'record':row['record'],'subject':row['subject'],'n_epochs':len(y),
                   'metadata':meta,'reader_exact_parity':True,'historical_prediction_mismatches':mismatch,
                   'confusion':cms,'archive_sha256':sha(archive),'seconds':time.time()-started,'device':torch.cuda.get_device_name(0)})
        save_json(OUT/f'worker{shard}_status.json',{'status':'running','cohort':row['cohort'],'record':row['record']})
        print(row['cohort'],row['record'],len(y),'done',round(time.time()-started,1),flush=True)
    save_json(OUT/f'worker{shard}_status.json',{'status':'complete'})


def paired_bootstrap(cms):
    rng=np.random.default_rng(20261011); n=len(cms['original'])
    values={m:{k:[] for k in KEYS} for m in MODES[1:]}
    for _ in range(5000):
        chosen=rng.integers(0,n,size=n); base=cm_metrics(cms['original'][chosen].sum(0))
        for m in values:
            value=cm_metrics(cms[m][chosen].sum(0))
            for k in KEYS: values[m][k].append(value[k]-base[k])
    return {m:{k:np.quantile(v,[.025,.975]).tolist() for k,v in d.items()} for m,d in values.items()}


def decision(cohorts,mode):
    gains={c:v['deltas'][mode]['macro_f1']>=.005 for c,v in cohorts.items()}
    safety={c:all(v['deltas'][mode][k]>=minimum for k,minimum in (('macro_f1',-.005),('accuracy',-.005),('n1_recall',-.02),('n1_f1',-.005),('n3_f1',-.005))) for c,v in cohorts.items()}
    nonfollowup=sum(gains[c] for c in ('st','isruc','shhs1'))
    return {'gain_at_least_two_nonfollowup_cohorts':nonfollowup>=2,'gain_in_st_or_isruc':gains['st'] or gains['isruc'],
            'safety_all_cohorts':all(safety.values()),'gain_per_cohort':gains,'safety_per_cohort':safety,
            'passes_resource_gate':nonfollowup>=2 and (gains['st'] or gains['isruc']) and all(safety.values())}


def summarize(p,digest):
    cohorts={}
    for cohort in ('st','isruc','shhs1','shhs2'):
        expected=[r for r in p['records'] if r['cohort']==cohort]
        rows=[json.loads((OUT/cohort/(r['record']+'.json')).read_text()) for r in expected]
        valid=[r for r in rows if 'error' not in r]
        if not valid or any(r['protocol_sha256']!=digest for r in rows): raise RuntimeError('Incomplete cohort')
        subjects=sorted({r['subject'] for r in valid})
        cms={m:np.stack([sum(np.asarray(r['confusion'][m]) for r in valid if r['subject']==s) for s in subjects]) for m in MODES}
        table={m:cm_metrics(cm.sum(0)) for m,cm in cms.items()}
        deltas={m:{k:table[m][k]-table['original'][k] for k in KEYS} for m in MODES[1:]}
        changes={m:{k:sum(cm_metrics(cms[m][i])[k]>cm_metrics(cms['original'][i])[k] for i in range(len(subjects))) for k in ('macro_f1','n1_f1','n3_f1')} for m in MODES[1:]}
        cohorts[cohort]={'requested':len(expected),'records':len(valid),'subjects':len(subjects),'n_epochs':sum(r['n_epochs'] for r in valid),
                        'exclusions':[r for r in rows if 'error' in r],'table':table,'deltas':deltas,
                        'paired_subject_bootstrap_delta_95ci':paired_bootstrap(cms),'subject_gain_counts':changes,
                        'records_with_historical_parity':sum(r['historical_prediction_mismatches']==0 for r in valid)}
    gates={m:decision(cohorts,m) for m in MODES[1:]}
    save_json(OUT/'summary.json',{'protocol_sha256':digest,'cohorts':cohorts,'resource_gates':gates,'limitations':p['limits']})
    lines=['# 冻结主线：2×2预处理敏感性实验','',
           '原稿20-fold source ensemble；双TITAN V，FP32推理，无训练。标签/裁剪/纳入清单与历史不变，所有配置预设且全部报告。不称新盲测，也不重新挑选稿件主结果。','',
           '| 队列 | 模式 | Accuracy | Macro-F1 | N1 recall | N1 F1 | N3 F1 |','|---|---|---:|---:|---:|---:|---:|']
    for c,v in cohorts.items():
        for m,d in v['table'].items(): lines.append('| '+' | '.join([c,m,*[f'{d[k]:.5f}' for k in ('accuracy','macro_f1','n1_recall','n1_f1','n3_f1')]])+' |')
    lines+=['','## 预设推进规则','', '至少两个非随访队列Macro-F1≥+0.005，其中必须包含ST或ISRUC；其他队列Macro-F1/Accuracy≥−0.005，N1 recall≥−0.02，N1/N3 F1≥−0.005。SHHS2不计作新的独立成功队列，但检查安全性。门槛是探索性资源决策，不是确认性显著性。','']
    for m,g in gates.items(): lines.append(f"- {m}: passes_resource_gate={g['passes_resource_gate']}；`{json.dumps(g)}`")
    lines+=['','## 规模与核对','']
    for c,v in cohorts.items(): lines.append(f"- {c}: 请求{v['requested']}，完成{v['records']}记录/{v['subjects']}受试者/{v['n_epochs']}epochs，原处理历史预测完全一致{v['records_with_historical_parity']}条；排除{len(v['exclusions'])}条。")
    lines+=['','ST原采样100Hz，original/antialias严格一致是负对照；both与emg_envelope也严格一致。EMG处理固定HP16Hz、rectify、LP0.4Hz、1Hz polyphase后repeat100，不进行幅值匹配。源域已1Hz EMG不改变。',
            '抗混叠full-record polyphase仅用于原始>100Hz信号；没有30s边界滤波。零相位EMG处理是离线敏感性，不可称因果实时部署。','',
            '5000次受试者配对重采样差值区间、各受试者增益计数和逐记录结果见summary.json。ST两晚共同抽样；SHHS只有预设32记录样本，不能冒充完整库结果。','',
            '## 限制','',*['- '+v for v in p['limits']], '', '不自动修改稿件或启动全SHHS/重新训练。']
    (ROOT/'docs/mainline_preprocessing_sensitivity_20261011.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,m in zip(axes,MODES[1:]):
        for k in ('macro_f1','n1_f1','n3_f1'):
            ax.plot(range(4),[100*cohorts[c]['deltas'][m][k] for c in cohorts],'o-',label=k)
        ax.axhline(0,color='gray'); ax.set_xticks(range(4),['ST','ISRUC','SHHS1sample','SHHS2sample'],rotation=25); ax.set_title(m); ax.legend()
    axes[0].set_ylabel('Paired cohort change (percentage points)'); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(OUT/f'sensitivity_changes.{ext}',dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true'); parser.add_argument('--worker',type=int); a=parser.parse_args()
    if a.worker is not None:
        p=json.loads((OUT/'protocol.json').read_text()); worker(p,sha(OUT/'protocol.json'),a.worker); return
    p,digest=prepare(); print('Locked',digest,flush=True)
    if a.prepare_only: return
    if torch.cuda.device_count()<2: raise RuntimeError('Two GPUs required, no CPU fallback')
    with (OUT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save_json(OUT/'status.json',{'status':'running','pid':os.getpid(),'protocol_sha256':digest})
        def launch(g):
            with (OUT/f'worker{g}.log').open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(g)],cwd=ROOT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(g),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'},
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(launch,g) for g in (0,1)]
                for f in futures: f.result()
            summarize(p,digest); save_json(OUT/'status.json',{'status':'complete','protocol_sha256':digest})
        except Exception as e:
            save_json(OUT/'status.json',{'status':'failed','error':repr(e),'protocol_sha256':digest}); raise


if __name__=='__main__': main()

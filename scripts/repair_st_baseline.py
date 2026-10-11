"""Rebuild ST to new directory and re-evaluate frozen manuscript ensemble.

No model training, no resampling-method change, no replacement of old files.
"""
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

import numpy as np
import pyedflib
import torch
from torch.func import stack_module_state

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'external_baselines/AttnSleep'))
from model.model import AttnSleep
from scripts.prepare_sleepedf20_multichannel import prepare_one,read_labels,UNKNOWN,physical_unit_scale_to_volts
from scripts.evaluate_attnsleep_external_ensemble import predict_ensemble_logits
from scripts.run_end_to_end_growth_study import save_json,sha
from scripts.pre_submission_statistics import score_rows,stage_sensitivity_specificity
from scripts.audit_attnsleep_generalization_gap import metrics_from_confusion
from scripts.analyze_n1_clinical import clinical,CLINICAL

OLD=ROOT/'data/sleepedf_st_multichannel_npz/eeg2_eog_emg_physical_volts'
NEW=ROOT/'data/sleepedf_st_multichannel_npz/eeg2_eog_emg_physical_volts_repaired_20261011'
OUT=ROOT/'outputs/st_baseline_repair_20261011'
MODELS=ROOT/'outputs/attnsleep_multichannel_baseline_train/full20_quality_gate_4ch_emg_30epoch'
HISTORY=ROOT/'outputs/external_validation/sleepedf_st/quality_gated_attnsleep_20fold_frozen_ensemble_no_tta'
OLDSTATS=ROOT/'outputs/external_validation/pre_submission/pre_submission_statistics.json'


def prepare():
    summary=json.loads((OLD/'summary.json').read_text())
    incident=json.loads((ROOT/'outputs/st_input_integrity_incident_20261011/audit.json').read_text())
    affected=[r['record'] for r in incident['records'] if not r['raw_matches_cache']]
    records=[]
    for meta in summary['records']:
        stem=meta['psg_file'].replace('-PSG.edf','')
        raw=ROOT/summary['raw_dir']/meta['psg_file'];ann=ROOT/summary['raw_dir']/meta['hypnogram_file']
        st=raw.stat()
        records.append({'record':stem,'subject':stem[3:5],'meta':meta,'signal':str(raw.resolve()),'annotation':str(ann.resolve()),
                        'signal_inventory':{'bytes':st.st_size,'mtime_ns':st.st_mtime_ns},'annotation_sha256':sha(ann),
                        'old_npz_sha256':sha(OLD/(stem+'.npz')),'old_prediction_sha256':sha(HISTORY/(stem+'_predictions.npz')),
                        'affected':stem in affected})
    codes=[Path(__file__),ROOT/'scripts/prepare_sleepedf20_multichannel.py',ROOT/'external_baselines/AttnSleep/model/model.py',
           ROOT/'scripts/evaluate_attnsleep_external_ensemble.py',ROOT/'scripts/pre_submission_statistics.py']
    p={'date':'2026-10-11','study':'ST input integrity repair and identical frozen20-fold baseline rescore, not model improvement',
       'records':records,'code_sha256':{str(f):sha(f) for f in codes},
       'model_sha256':{str(MODELS/f'fold_{i:02d}/checkpoint_best.pt'):sha(MODELS/f'fold_{i:02d}/checkpoint_best.pt') for i in range(20)},
       'old_stats_sha256':sha(OLDSTATS),'old_summary_sha256':sha(OLD/'summary.json'),
       'new_data_dir':str(NEW),'old_data_dir':str(OLD),'channels':summary['channels'],
       'preprocessing':'Same physical-volts,100Hz,3000samples,30s/30min benchmark crop,label mapping,no extra software filters.Only closePSG before annotation fallback and reject invalid/all-zero channel.',
       'verification':'44records,22subjects,42471epochs,alllabels/shapes unchanged;37unaffected x EXACT match;7affected x EXACT match independent label-first raw reconstruction;none allzero. Preserve old data/predictions/models/statistics.',
       'inference':'2TITAN V workers, original20fold meanlogits,FP32,batch64,evalBN,noTTA/newheads/CPUfallback;37unaffected predictions must match history exactly.',
       'statistics':'5000subjectpairedbootstrap old/corrected, bothSTnights together; separate corrected22subjectbootstrap pooled score. Improvement is input repair, not method gain.',
       'clinical':'Secondary record endpoints on retained benchmark timeline and retained valid-epoch denominator, not clinical time-in-bed.Original indices used forWASO/REMlatency, gap flags,STnights clustered subjectbootstrap5000.',
       'publication':'New corrected statistics/figures in new output directory; old archives preserved. STadaptation/context/TTA dependencies flagged not automatically refit. No publish/upload.'}
    OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=p:raise RuntimeError('Locked repair protocol changed')
    if not path.exists():save_json(path,p)
    return p,sha(path)


def verify(p):
    for paths in ('code_sha256','model_sha256'):
        for f,v in p[paths].items():
            if sha(Path(f))!=v:raise RuntimeError('Locked file changed '+f)
    if sha(OLDSTATS)!=p['old_stats_sha256'] or sha(OLD/'summary.json')!=p['old_summary_sha256']:raise RuntimeError('Old aggregate artifacts changed')


def original_indices(row,y):
    y_all=read_labels(Path(row['annotation']),row['meta']['n_all_epochs'])
    valid=np.flatnonzero(y_all!=UNKNOWN);sleep=np.flatnonzero(y_all[valid]!=0)
    valid=valid[max(int(sleep[0])-60,0):min(int(sleep[-1])+61,len(valid))]
    if not np.array_equal(y_all[valid],y):raise RuntimeError('Original timeline label mismatch')
    return valid


def independent_raw(row,indices):
    # No annotation reader active while raw PSG handle is open.
    r=pyedflib.EdfReader(row['signal']);channels=[]
    try:
        for name in ('EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental'):
            i=r.getSignalLabels().index(name);assert r.getSampleFrequency(i)==100
            raw=r.readSignal(i).astype(np.float32)*physical_unit_scale_to_volts(r.getPhysicalDimension(i))
            channels.append(raw[:row['meta']['n_all_epochs']*3000].reshape(row['meta']['n_all_epochs'],3000)[indices])
    finally:r.close()
    return np.stack(channels,1)


def rebuild(p,digest):
    NEW.mkdir(exist_ok=True,parents=True);audits=[];newsummary=json.loads((OLD/'summary.json').read_text())
    newsummary['output_dir']=str(NEW.relative_to(ROOT));newsummary['repair_protocol_sha256']=digest
    newsummary['filter']={'bandpass_low':0.,'bandpass_high':0.,'notch_freq':0.,'filter_order':4}
    newsummary['records']=[]
    for row in p['records']:
        old_path=OLD/(row['record']+'.npz');target=NEW/(row['record']+'.npz')
        stat=Path(row['signal']).stat()
        assert row['signal_inventory']=={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        assert sha(Path(row['annotation']))==row['annotation_sha256'] and sha(old_path)==row['old_npz_sha256']
        assert sha(HISTORY/(row['record']+'_predictions.npz'))==row['old_prediction_sha256']
        if target.exists():
            audit_path=OUT/'data_audit'/f"{row['record']}.json"
            if not audit_path.exists():raise RuntimeError('Partial data without audit; inspect instead of overwrite')
            saved=json.loads(audit_path.read_text());assert saved['protocol_sha256']==digest and saved['new_sha256']==sha(target)
            with np.load(target) as d:x=d['x'];y=d['y']
            meta=saved['metadata']
        else:
            x,y,meta=prepare_one(Path(row['signal']),Path(row['annotation']),p['channels'],3000,30,'physical-volts',0,0,0,4)
        with np.load(old_path,allow_pickle=True) as old:
            if x.shape!=old['x'].shape or not np.array_equal(y,old['y']):raise RuntimeError('Changed labels or shapes')
            same=np.array_equal(x,old['x'])
            if same==row['affected']:raise RuntimeError('Unexpected changed-record set')
        indices=original_indices(row,y)
        if not np.array_equal(x,independent_raw(row,indices)):raise RuntimeError('Raw reconstruction mismatch')
        if not np.isfinite(x).all() or np.any(np.all(x==0,axis=(1,2))):raise RuntimeError('Invalid corrected data')
        if not target.exists():
            temporary=target.with_suffix('.partial.npz')
            np.savez_compressed(temporary,x=x,y=y,fs=np.float32(100),ch_labels=np.asarray(p['channels']),
                   normalization=np.asarray('physical-volts'),filter_config=np.asarray(json.dumps(newsummary['filter'],sort_keys=True)),
                   source_psg=row['meta']['psg_file'],source_hypnogram=row['meta']['hypnogram_file'],indices=indices)
            temporary.replace(target)
        audit={'protocol_sha256':digest,'record':row['record'],'affected':row['affected'],'shape':list(x.shape),
               'labels_unchanged':True,'raw_reconstruction_exact':True,'zero_epochs':0,'unaffected_x_exact':same,
               'old_sha256':row['old_npz_sha256'],'new_sha256':sha(target),'metadata':meta}
        save_json(OUT/'data_audit'/f"{row['record']}.json",audit);audits.append(audit);newsummary['records'].append(meta)
        print('Rebuilt/verified',row['record'],'changed',not same,flush=True)
    assert len(audits)==44 and sum(a['affected'] for a in audits)==7
    save_json(NEW/'summary.json',newsummary)
    save_json(OUT/'data_integrity.json',{'protocol_sha256':digest,'records':audits,'records_checked':44,
             'changed_records':7,'all_shapes_labels_unchanged':True,'raw_reconstruction_exact_all':True,'zero_epochs':0})


def gpu_worker(p,digest,shard):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);verify(p)
    instances=[]
    for path in p['model_sha256']:
        ck=torch.load(path,map_location='cuda',weights_only=False)
        m=AttnSleep(input_channels=4,channel_fusion='quality-gated',channel_dropout=.15).cuda().eval()
        m.load_state_dict(ck['model_state']);instances.append(m)
    params,buffers=stack_module_state(instances);model=instances[0]
    for row in p['records'][shard::2]:
        folder=OUT/'predictions';folder.mkdir(exist_ok=True);target=folder/(row['record']+'_predictions.npz')
        audit_path=OUT/'prediction_audit'/f"{row['record']}.json"
        if target.exists():
            saved=json.loads(audit_path.read_text());assert saved['archive_sha256']==sha(target) and saved['protocol_sha256']==digest
            continue
        with np.load(NEW/(row['record']+'.npz')) as d:x=d['x'];y=d['y'];indices=d['indices']
        logits=predict_ensemble_logits(model,params,buffers,x,64,[{'kind':'identity'}],torch.device('cuda'))
        if not np.isfinite(logits).all():raise RuntimeError('Nonfinite logits')
        prediction=logits.argmax(1).astype(np.int8)
        with np.load(HISTORY/(row['record']+'_predictions.npz')) as old:
            assert np.array_equal(y,old['y_true']);mismatches=int((prediction!=old['y_pred']).sum())
        if not row['affected'] and mismatches:raise RuntimeError('Unaffected historical predictions changed')
        temp=target.with_suffix('.partial.npz');np.savez_compressed(temp,y_true=y,y_pred=prediction,logits=logits,indices=indices);temp.replace(target)
        save_json(audit_path,{'protocol_sha256':digest,'record':row['record'],'subject':row['subject'],'affected':row['affected'],
                   'n_epochs':len(y),'confusion_matrix':np.bincount(y*5+prediction,minlength=25).reshape(5,5).tolist(),
                   'historical_prediction_changes':mismatches,'archive_sha256':sha(target),'device':torch.cuda.get_device_name(0)})
        print('CUDA scored',row['record'],len(y),'changed predictions',mismatches,flush=True)


def paired_stats(rows):
    subjects=sorted({r['subject'] for r in rows});old=[];new=[]
    for s in subjects:
        chosen=[r for r in rows if r['subject']==s]
        old.append(sum(np.bincount(r['y_true']*5+r['old_pred'],minlength=25).reshape(5,5) for r in chosen))
        new.append(sum(np.bincount(r['y_true']*5+r['y_pred'],minlength=25).reshape(5,5) for r in chosen))
    old,new=np.stack(old),np.stack(new);rng=np.random.default_rng(20261011)
    keys=('accuracy','macro_f1','weighted_f1','kappa');draws={k:[] for k in keys}
    for _ in range(5000):
        idx=rng.integers(0,len(subjects),size=len(subjects));a=metrics_from_confusion(old[idx].sum(0));b=metrics_from_confusion(new[idx].sum(0))
        for k in keys:draws[k].append(b[k]-a[k])
    a=metrics_from_confusion(old.sum(0));b=metrics_from_confusion(new.sum(0))
    return {'old':{k:a[k] for k in keys},'corrected':{k:b[k] for k in keys},
            'corrected_minus_old':{k:b[k]-a[k] for k in keys},'paired_subject_delta_95ci':{k:np.quantile(d,[.025,.975]).tolist() for k,d in draws.items()}}


def timeline_clinical(y,indices):
    value=clinical(y);sleep=np.flatnonzero(y!=0);rem=np.flatnonzero(y==4)
    value['waso_minutes']=float(((y==0)&(indices>=indices[sleep[0]])&(indices<=indices[sleep[-1]])).sum()*.5) if len(sleep) else 0.
    valid_rem=rem[indices[rem]>=indices[sleep[0]]] if len(sleep) else []
    value['rem_latency_minutes']=float((indices[valid_rem[0]]-indices[sleep[0]])*.5) if len(valid_rem) else float('nan')
    return value


def clinical_stats(rows):
    subject_ids=sorted({r['subject'] for r in rows});rng=np.random.default_rng(20261011);output={};records=[]
    for r in rows:
        records.append({'subject':r['subject'],'record':r['record'],'timeline_gaps':int((np.diff(r['indices'])>1).sum()),
                        'truth':timeline_clinical(r['y_true'],r['indices']),'corrected':timeline_clinical(r['y_pred'],r['indices']),
                        'old':timeline_clinical(r['old_pred'],r['indices'])})
    index=np.stack([rng.integers(0,len(subject_ids),len(subject_ids)) for _ in range(5000)])
    for k in CLINICAL:
        results={}
        for mode in ('old','corrected'):
            deltas=np.array([r[mode][k]-r['truth'][k] for r in records])
            valid=np.isfinite(deltas);err=deltas[valid];ids=np.array([r['subject'] for r in records])[valid]
            sums=np.array([err[ids==s].sum() for s in subject_ids]);abs_sums=np.array([np.abs(err[ids==s]).sum() for s in subject_ids]);n=np.array([(ids==s).sum() for s in subject_ids])
            denominator=n[index].sum(1);good=denominator>0
            bias=sums[index].sum(1)[good]/denominator[good];mae=abs_sums[index].sum(1)[good]/denominator[good]
            sd=err.std(ddof=1) if len(err)>1 else 0.
            results[mode]={'n_valid_records':int(valid.sum()),'bias':float(err.mean()),'mae':float(np.abs(err).mean()),
                          'limits_of_agreement':(err.mean()+np.array([-1,1])*1.96*sd).tolist(),
                          'bias_subject_cluster_bootstrap_95ci':np.quantile(bias,[.025,.975]).tolist(),
                          'mae_subject_cluster_bootstrap_95ci':np.quantile(mae,[.025,.975]).tolist()}
        output[k]=results
    return {'protocol':'Retained benchmark window: TST/stage durations from valid scored epochs, SE denominator valid retained epochs not time-in-bed; WASO observedW between first/last sleep, REM latency original timeline;22subject clustered bootstrap, record-pooled point estimates;not clinical readiness.',
            'metrics':output,'records':records}


def summarize(p,digest):
    rows=[]
    for r in p['records']:
        path=OUT/'predictions'/(r['record']+'_predictions.npz')
        with np.load(path) as d:row={'record':r['record'],'subject':r['subject'],'y_true':d['y_true'].astype(int),'y_pred':d['y_pred'].astype(int),'indices':d['indices']}
        with np.load(HISTORY/(r['record']+'_predictions.npz')) as old:row['old_pred']=old['y_pred'].astype(int)
        rows.append(row)
    score=score_rows(rows,5000,20260919);score['stage_sensitivity_specificity']=stage_sensitivity_specificity(score)
    score['source']=str(OUT/'predictions');score['target_labels_used']='historical benchmark crop and scoring, no target tuning';score['repair_protocol_sha256']=digest
    pair=paired_stats(rows);clinc=clinical_stats(rows)
    save_json(OUT/'st_corrected_statistics.json',score);save_json(OUT/'paired_correction_effect.json',pair)
    # JSON-safe missing REM latency; not converted into zero clinical error.
    from scripts.analyze_n1_clinical import json_clean
    save_json(OUT/'st_corrected_clinical.json',json_clean(clinc))
    updated=copy.deepcopy(json.loads(OLDSTATS.read_text()));updated['cohorts']['SleepEDF_ST']=score
    updated['protocol']='Revised ST frozen baseline after input integrity correction; other cohorts unchanged from historical locked statistics'
    updated['st_integrity_repair']={'protocol_sha256':digest,'old_stats_path':str(OLDSTATS),'not_model_innovation':True}
    if 'SleepEDF_ST_adaptation' in updated.get('adaptation',{}):
        updated['adaptation']['SleepEDF_ST_adaptation']={'status':'invalidated_pending_refit_on_repaired_data','reason':'fit/validation/test may containzero-filledSTinputs;not corrected by rescoring'}
    save_json(OUT/'pre_submission_statistics_corrected.json',updated)
    with (OUT/'cohort_summary_corrected.csv').open('w',newline='') as f:
        fields=['cohort','n_subjects','n_records','n_epochs','accuracy','macro_f1','weighted_f1','kappa'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({'cohort':c,**{k:v[k] for k in ('n_subjects','n_records','n_epochs')},**v['point_estimate']} for c,v in updated['cohorts'].items())
    with (OUT/'st_per_subject_corrected.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(score['per_subject'][0]));w.writeheader();w.writerows(score['per_subject'])
    figures(score,pair,rows,updated)
    integrity=json.loads((OUT/'data_integrity.json').read_text())
    lines=['# ST数据修复与冻结主线基线重算','',
            '全部44记录/22受试者/42471epochs重建并评分。旧缓存、预测、源模型和旧统计完整保留。输入制备唯一改动为EDF句柄生命周期与无效信号拒绝，不是方法创新。','',
            '| 指标 | 旧输入结果（失效） | 修正输入结果 | 差值(pp) | 修正95%subject CI |','|---|---:|---:|---:|---|']
    for k in ('accuracy','macro_f1','weighted_f1','kappa'):
        ci=score['subject_bootstrap_95ci'][k];lines.append(f"| {k} | {pair['old'][k]:.5f} | {pair['corrected'][k]:.5f} | {100*pair['corrected_minus_old'][k]:+.3f} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    lines+=['','7条/6106epochs原来是全零，现与独立原波形重建完全一致。37条不受影响的信号逐元素不变，预测必须与历史逐epoch一致。全部标签、shape、裁剪和人数不变。','',
            '## 分期指标','', '| 分期 | Precision | Recall | F1 | Support |','|---|---:|---:|---:|---:|']
    for s in ('W','N1','N2','N3','REM'):
        v=score['classification_report'][s];lines.append(f"| {s} | {v['precision']:.5f} | {v['recall']:.5f} | {v['f1-score']:.5f} | {int(v['support'])} |")
    lines+=['','## 临床端点边界','',
            '只在保留的benchmark窗口定义TST/分期时长，睡眠效率分母为保留有效epoch而非真实time-in-bed；WASO仅观测W，REMlatency用原始时间索引。未知间隙不补标签，不能当作临床部署验证。22subject聚类bootstrap同时包含两夜。旧/修正端点bias/MAE/LoA/CI见st_corrected_clinical.json。','',
            '## 投稿数据状态','',
            '原统计文件保留供追溯，修正统一统计在outputs/st_baseline_repair_20261011/pre_submission_statistics_corrected.json。新的外部表格和图在figures/；旧ST适配/上下文结果不能自动恢复，须另行处理。完整影响清单见docs/st_repair_dependency_status_20261011.md。',
            '不能将此次差值宣传为模型或预处理创新，不能称未查看标签的盲测，不能把SHHS2变为独立人群。没有自动重启2×2敏感性试验，也没有重新训练。']
    (ROOT/'docs/st_baseline_repair_results_20261011.md').write_text('\n'.join(lines)+'\n')
    verify(p)
    save_json(OUT/'summary.json',{'protocol_sha256':digest,'corrected':score,'paired_correction':pair,
              'data_integrity':{k:v for k,v in integrity.items() if k!='records'},'model_innovation':False,'new_data_dir':str(NEW)})


def figures(score,pair,rows,updated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    import scripts.make_sleep_medicine_figures as publication
    old_load=publication.load
    publication.OUT=dest
    publication.load=lambda path:updated if path=='outputs/external_validation/pre_submission/pre_submission_statistics.json' else old_load(path)
    publication.external_validation();publication.confusion_heatmaps()
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    cm=np.array(score['confusion_matrix'],dtype=float);norm=cm/cm.sum(1,keepdims=True)
    axes[0].imshow(norm,vmin=0,vmax=1,cmap='Blues')
    for i in range(5):
        for j in range(5):axes[0].text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center')
    stages=['W','N1','N2','N3','REM'];axes[0].set_xticks(range(5),stages);axes[0].set_yticks(range(5),stages);axes[0].set_title('ST corrected row-normalized confusion')
    x=np.arange(4);keys=['accuracy','macro_f1','weighted_f1','kappa']
    axes[1].bar(x-.17,[pair['old'][k] for k in keys],width=.34,label='Old invalid inputs')
    axes[1].bar(x+.17,[pair['corrected'][k] for k in keys],width=.34,label='Corrected inputs')
    axes[1].set_xticks(x,['Accuracy','MacroF1','WeightedF1','Kappa'],rotation=20);axes[1].set_ylim(0,1);axes[1].legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(dest/f'st_input_correction.{ext}',dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    if args.worker is not None:
        p=json.loads((OUT/'protocol.json').read_text());gpu_worker(p,sha(OUT/'protocol.json'),args.worker);return
    p,digest=prepare();verify(p)
    if torch.cuda.device_count()<2:raise RuntimeError('TwoGPUsrequired')
    with (OUT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save_json(OUT/'status.json',{'status':'rebuilding','pid':os.getpid(),'protocol_sha256':digest})
        try:
            rebuild(p,digest);save_json(OUT/'status.json',{'status':'gpu_rescoring','pid':os.getpid(),'protocol_sha256':digest})
            def work(g):
                with (OUT/f'worker{g}.log').open('a') as log:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(g)],cwd=ROOT,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(g),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(work,g) for g in (0,1)]
                for f in futures:f.result()
            summarize(p,digest);save_json(OUT/'status.json',{'status':'complete_pending_final_audit','protocol_sha256':digest})
        except Exception as e:
            save_json(OUT/'status.json',{'status':'failed','error':repr(e),'protocol_sha256':digest});raise


if __name__=='__main__':main()

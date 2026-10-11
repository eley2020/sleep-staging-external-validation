"""Explicit amendment: historical ST inference was CPU; require FP32 logit parity.

Original repair protocol/source/partial archives remain unchanged. All neural
inference stays on CUDA. An old-input CUDA counterfactual isolates input repair.
"""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from torch.func import stack_module_state

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import scripts.repair_st_baseline as base

AMEND=base.OUT/'cuda_parity_amendment.json'
ATOL=RTOL=1e-5

def check_parity(new,old):
    if not np.allclose(new,old,atol=ATOL,rtol=RTOL):
        raise RuntimeError('CPU/CUDA logits exceed declared FP32 tolerance')
    changed=np.flatnonzero(new.argmax(1)!=old.argmax(1))
    if len(changed):
        top=np.sort(old[changed],axis=1)[:,-2:]
        margin=top[:,1]-top[:,0]
        bound=2*(ATOL+RTOL*np.max(np.abs(old[changed]),axis=1))
        if np.any(margin>bound):raise RuntimeError('Non-near-tie predicted class changed')
    return {'max_absolute_logit_difference':float(np.max(np.abs(new-old))),
        'prediction_changes':int(len(changed)),'changed_indices':changed.tolist(),
        'old_top2_margin_at_changes':(np.sort(old[changed],axis=1)[:,-1]-np.sort(old[changed],axis=1)[:,-2]).tolist(),
        'fp32_logit_parity':True}

def amendment(p):
    value={'original_protocol_sha256':base.sha(base.OUT/'protocol.json'),
        'original_protocol_preserved':True,'reason':'Historical reproduction command used --device cpu --batch-size 128; new inference CUDA. ST7181J0 epoch948 W/N1 historical top2 margin4.76837158203125e-7, max logit difference4.76837158203125e-6. CUDA batch64 and128 identical for diagnostic record.',
        'code_sha256':{str(Path(__file__).resolve()):base.sha(Path(__file__)),str(ROOT/'scripts/audit_st_repair_outputs.py'):base.sha(ROOT/'scripts/audit_st_repair_outputs.py')},
        'unaffected_parity':{'atol':ATOL,'rtol':RTOL,'all_logits_required_close':True,'changes_allowed_only_within_twice_elementwise_tolerance':True,'all_changes_disclosed':True},
        'counterfactual':'Old affected zero-input records separately scored on same20 models CUDA,batch64; unchanged37 use verified CUDA logits. Same-backend old/new paired bootstrap isolates data repair. Historical CPU comparison also preserved.',
        'reuse':'Completed repaired data and individually hashed predictions retained; missing predictions only scored; independent amended audit validates all44.',
        'neural_device':'CUDA only,no CPU neural execution,no retraining/no model/input/label changes.'}
    if AMEND.exists():
        if json.loads(AMEND.read_text())!=value:raise RuntimeError('Locked amendment changed')
    else:base.save_json(AMEND,value)
    return value,base.sha(AMEND)

def verify_amendment():
    value=json.loads(AMEND.read_text());base.verify(json.loads((base.OUT/'protocol.json').read_text()))
    assert value['original_protocol_sha256']==base.sha(base.OUT/'protocol.json')
    for path,digest in value['code_sha256'].items():assert base.sha(Path(path))==digest

def worker(shard):
    verify_amendment();p=json.loads((base.OUT/'protocol.json').read_text())
    digest=base.sha(base.OUT/'protocol.json');adigest=base.sha(AMEND)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);models=[]
    for path in p['model_sha256']:
        m=base.AttnSleep(input_channels=4,channel_fusion='quality-gated',channel_dropout=.15).cuda().eval()
        m.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model_state']);models.append(m)
    params,buffers=stack_module_state(models)
    for row in p['records'][shard::2]:
        stem=row['record'];target=base.OUT/'predictions'/(stem+'_predictions.npz')
        audit_path=base.OUT/'prediction_audit'/(stem+'.json')
        with np.load(base.NEW/(stem+'.npz')) as d:x=d['x'];y=d['y'];indices=d['indices']
        with np.load(base.HISTORY/(stem+'_predictions.npz')) as d:historical=d['mean_logits'];old_pred=d['y_pred']
        if target.exists():
            saved=json.loads(audit_path.read_text());assert saved['archive_sha256']==base.sha(target) and saved['protocol_sha256']==digest
            with np.load(target) as d:logits=d['logits'];pred=d['y_pred']
        else:
            logits=base.predict_ensemble_logits(models[0],params,buffers,x,64,[{'kind':'identity'}],torch.device('cuda'))
            assert np.isfinite(logits).all();pred=logits.argmax(1).astype(np.int8)
            if not row['affected']:check_parity(logits,historical)
            target.parent.mkdir(exist_ok=True);temp=target.with_suffix('.partial.npz')
            np.savez_compressed(temp,y_true=y,y_pred=pred,logits=logits,indices=indices);temp.replace(target)
            base.save_json(audit_path,{'protocol_sha256':digest,'amendment_sha256':adigest,'record':stem,'subject':row['subject'],
                'affected':row['affected'],'n_epochs':len(y),'confusion_matrix':np.bincount(y*5+pred,minlength=25).reshape(5,5).tolist(),
                'historical_prediction_changes':int((pred!=old_pred).sum()),'archive_sha256':base.sha(target),'device':torch.cuda.get_device_name(0)})
        if not row['affected']:check_parity(logits,historical)
        counter=base.OUT/'old_input_cuda'/(stem+'_predictions.npz');caudit=base.OUT/'counterfactual_audit'/(stem+'.json')
        if counter.exists():
            saved=json.loads(caudit.read_text());assert saved['archive_sha256']==base.sha(counter) and saved['amendment_sha256']==adigest
            with np.load(counter) as d:old_logits=d['logits']
        else:
            if row['affected']:
                with np.load(base.OLD/(stem+'.npz')) as d:oldx=d['x']
                old_logits=base.predict_ensemble_logits(models[0],params,buffers,oldx,64,[{'kind':'identity'}],torch.device('cuda'))
            else:old_logits=logits
            parity=check_parity(old_logits,historical)
            counter.parent.mkdir(exist_ok=True);temp=counter.with_suffix('.partial.npz')
            np.savez_compressed(temp,y_true=y,y_pred=old_logits.argmax(1).astype(np.int8),logits=old_logits,indices=indices);temp.replace(counter)
            base.save_json(caudit,{'record':stem,'protocol_sha256':digest,'amendment_sha256':adigest,'archive_sha256':base.sha(counter),'device':torch.cuda.get_device_name(0),'historical_parity':parity})
        check_parity(old_logits,historical)
        print('CUDA completed corrected + old-input control',stem,flush=True)

def finish(p,digest):
    base.summarize(p,digest)
    rows=[]
    for r in p['records']:
        with np.load(base.OUT/'predictions'/(r['record']+'_predictions.npz')) as d:
            row={'record':r['record'],'subject':r['subject'],'y_true':d['y_true'].astype(int),'y_pred':d['y_pred'].astype(int)}
        with np.load(base.OUT/'old_input_cuda'/(r['record']+'_predictions.npz')) as d:row['old_pred']=d['y_pred'].astype(int)
        rows.append(row)
    same=base.paired_stats(rows);base.save_json(base.OUT/'paired_input_repair_cuda.json',same)
    parity={r['record']:json.loads((base.OUT/'counterfactual_audit'/(r['record']+'.json')).read_text())['historical_parity'] for r in p['records']}
    base.save_json(base.OUT/'cuda_historical_parity.json',{'amendment_sha256':base.sha(AMEND),'records':parity,'historical_prediction_changes':sum(v['prediction_changes'] for v in parity.values()),'maximum_logit_error':max(v['max_absolute_logit_difference'] for v in parity.values())})
    verify_amendment()

def main():
    if len(sys.argv)>1:worker(int(sys.argv[1]));return
    p=json.loads((base.OUT/'protocol.json').read_text());base.verify(p);_,adigest=amendment(p)
    with (base.OUT/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.save_json(base.OUT/'status.json',{'status':'gpu_rescoring_amended','amendment_sha256':adigest,'pid':os.getpid()})
        def launch(g):
            with (base.OUT/f'worker{g}_amended.log').open('a') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),str(g)],cwd=ROOT,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(g),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'},stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                for f in [pool.submit(launch,g) for g in (0,1)]:f.result()
            finish(p,base.sha(base.OUT/'protocol.json'))
            base.save_json(base.OUT/'status.json',{'status':'complete_pending_final_audit','amendment_sha256':adigest})
        except Exception as error:
            base.save_json(base.OUT/'status.json',{'status':'failed_amended','error':repr(error),'amendment_sha256':adigest});raise

if __name__=='__main__':main()

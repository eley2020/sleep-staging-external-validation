"""Leakage-safe Viterbi continuity post-processing for current SHHS adaptation."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

def args():
 p=argparse.ArgumentParser(); p.add_argument('--cache-dir',type=Path,default=Path('/path/to/datasets/SHHS/adaptation_cache/shhs1_trainval_210')); p.add_argument('--split-file',type=Path,default=Path('outputs/external_validation/shhs/adaptation/splits/seed20260813.json')); p.add_argument('--checkpoint',type=Path,default=Path('outputs/external_validation/shhs/adaptation/seed20260813_full/checkpoint_best.pt')); p.add_argument('--base-predictions',type=Path,default=Path('outputs/external_validation/shhs/adaptation/seed20260813_full/adapted_test_predictions.npz')); p.add_argument('--shhs-root',type=Path,default=Path('/path/to/datasets/SHHS/polysomnography')); p.add_argument('--cohort',choices=['shhs1','shhs2'],default='shhs1'); p.add_argument('--locked-weight',type=float,default=None); p.add_argument('--output-dir',type=Path,default=Path('outputs/external_validation/shhs/viterbi_seed20260813')); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); return p.parse_args()

def metric(y,p):
 from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score
 return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',labels=[0,1,2,3,4],zero_division=0)),'kappa':float(cohen_kappa_score(y,p,labels=[0,1,2,3,4]))}

def viterbi(prob, log_start, log_trans, weight, temperature=1.0):
 e=np.log(np.clip(prob,1e-8,1.0))/temperature; n=e.shape[0]; dp=np.empty((n,5)); ptr=np.zeros((n,5),dtype=np.int8); dp[0]=log_start+e[0]
 for t in range(1,n):
  z=dp[t-1][:,None]+weight*log_trans; ptr[t]=z.argmax(axis=0).astype(np.int8); dp[t]=e[t]+z.max(axis=0)
 out=np.empty(n,dtype=np.int64); out[-1]=int(dp[-1].argmax())
 for t in range(n-2,-1,-1): out[t]=ptr[t+1,out[t+1]]
 return out

def train_markov(paths):
 start=np.ones(5); trans=np.ones((5,5))
 for path in paths:
  with np.load(path,allow_pickle=True) as d: y=d['y'].astype(int)
  if len(y): start[y[0]]+=1
  if len(y)>1: np.add.at(trans,(y[:-1],y[1:]),1)
 return np.log(start/start.sum()), np.log(trans/trans.sum(axis=1,keepdims=True))

def predict(model,x,device,batch):
 import torch
 out=[]; model.eval()
 with torch.no_grad():
  for i in range(0,len(x),batch): out.append(torch.softmax(model(torch.from_numpy(x[i:i+batch]).to(device)),dim=1).cpu().numpy())
 return np.concatenate(out)

def main():
 a=args(); a.output_dir.mkdir(parents=True,exist_ok=True); split=json.loads(a.split_file.read_text())['split']; paths=sorted(a.cache_dir.glob('*.npz')); train_set=set(split['train_subjects']); val_set=set(split['validation_subjects'])
 def sid(p):
  with np.load(p,allow_pickle=True) as d:return str(d['subject'].item())
 train_paths=[p for p in paths if sid(p) in train_set]; val_paths=[p for p in paths if sid(p) in val_set]
 log_start,log_trans=train_markov(train_paths)
 val_records=[]
 for p in val_paths:
  with np.load(p,allow_pickle=True) as d: x=d['x'].astype('float32'); y=d['y'].astype('int64')
  val_records.append((p.stem,y,x))
 import torch
 sys.path.insert(0,str(Path('external_baselines/AttnSleep').resolve())); from model.model import AttnSleep
 device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else a.device); ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model=AttnSleep(input_channels=4,channel_fusion='quality-gated',channel_dropout=.15).to(device); model.load_state_dict(ck['model_state'])
 val_prob=[predict(model,x,device,a.batch_size) for _,_,x in val_records]; val_y=np.concatenate([y for _,y,_ in val_records]); val_raw=np.concatenate([p.argmax(1) for p in val_prob])
 candidates=[]
 for w in [0,.25,.5,1,1.5,2,3]:
  pred=np.concatenate([viterbi(p,log_start,log_trans,w) for p in val_prob]); candidates.append({'weight':w,'metrics':metric(val_y,pred)})
 selected=max(candidates,key=lambda z:(z['metrics']['macro_f1'],z['metrics']['accuracy']))
 if a.locked_weight is not None:
  selected={'weight':float(a.locked_weight),'metrics':None,'selection':'locked_from_shhs1_validation'}
 d=np.load(a.base_predictions,allow_pickle=True); y_test=d['y_true'].astype('int64'); base=d['y_pred'].astype('int64'); spans=d['spans']; record_map={str(r[0]):r for r in spans}
 try:
  from scripts.evaluate_attnsleep_shhs_ensemble import load_record,xml_for
 except ModuleNotFoundError:
  from evaluate_attnsleep_shhs_ensemble import load_record,xml_for
 edf_dir=a.shhs_root/'edfs'/a.cohort; ann_dir=a.shhs_root/'annotations-events-nsrr'/a.cohort; test_prob=np.empty((len(y_test),5),dtype='float32'); used=0
 for edf in sorted(edf_dir.glob('*.edf')):
  if edf.stem not in record_map: continue
  xml=xml_for(edf,ann_dir)
  if not xml.exists(): continue
  x,_,_=load_record(edf,xml,30); p=predict(model,x,device,a.batch_size); r=record_map[edf.stem]; s,e=int(r[2]),int(r[3]);
  if len(p)!=e-s: raise RuntimeError(f'span mismatch {edf.stem}')
  test_prob[s:e]=p; used+=1
  if used%25==0: print(f'test {used}',flush=True)
 if used!=len(spans): raise RuntimeError(f'processed {used}/{len(spans)} records')
 decoded=np.empty(len(y_test),dtype='int64'); offset=0
 for r in spans:
  s,e=int(r[2]),int(r[3]); decoded[s:e]=viterbi(test_prob[s:e],log_start,log_trans,selected['weight'])
 report={'protocol':('Locked SHHS1 validation-selected Viterbi applied to '+a.cohort+'; transitions estimated from SHHS1 calibration-training subjects; held-out labels used once for scoring'),'cohort':a.cohort,'n_train_subjects':len(train_set),'n_validation_subjects':len(val_set),'n_test_subjects':len(spans),'transition_weight_candidates':candidates,'selected':selected,'validation_raw':metric(val_y,val_raw),'test_raw_adapted':metric(y_test,base),'test_viterbi':metric(y_test,decoded),'test_delta':{k:metric(y_test,decoded)[k]-metric(y_test,base)[k] for k in metric(y_test,base)},'record_boundary_reset':True}
 (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)); np.savez_compressed(a.output_dir/'predictions.npz',y_true=y_test.astype('int8'),adapted_pred=base.astype('int8'),viterbi_pred=decoded.astype('int8'),spans=spans); print(json.dumps(report,indent=2))
if __name__=='__main__':main()

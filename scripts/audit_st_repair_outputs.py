"""Independent read-only repair output audit; no model execution."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.repair_st_baseline import OUT,OLD,NEW,HISTORY,verify,sha,save_json,original_indices,independent_raw
from scripts.audit_attnsleep_generalization_gap import metrics_from_confusion
from scripts.resume_st_repair_cuda import verify_amendment,check_parity,AMEND


def main():
    p=json.loads((OUT/'protocol.json').read_text());digest=sha(OUT/'protocol.json');verify(p);verify_amendment()
    total=np.zeros((5,5),dtype=np.int64);old_cuda_total=total.copy();changed=[];subjects=set();epochs=0;numeric_changes=0
    for row in p['records']:
        stem=row['record'];path=NEW/(stem+'.npz');prediction=OUT/'predictions'/(stem+'_predictions.npz')
        audit=json.loads((OUT/'data_audit'/(stem+'.json')).read_text())
        pa=json.loads((OUT/'prediction_audit'/(stem+'.json')).read_text())
        assert audit['protocol_sha256']==pa['protocol_sha256']==digest
        assert sha(path)==audit['new_sha256'] and sha(prediction)==pa['archive_sha256']
        assert sha(OLD/(stem+'.npz'))==row['old_npz_sha256']
        assert sha(HISTORY/(stem+'_predictions.npz'))==row['old_prediction_sha256']
        with np.load(path) as d:x=d['x'];y=d['y'];indices=d['indices']
        with np.load(OLD/(stem+'.npz')) as old:
            assert x.shape==old['x'].shape;assert np.array_equal(y,old['y'])
            same=np.array_equal(x,old['x']);assert same!=row['affected']
        assert np.isfinite(x).all() and not np.any(np.all(x==0,axis=(1,2)))
        assert np.array_equal(indices,original_indices(row,y))
        assert np.array_equal(x,independent_raw(row,indices))
        with np.load(prediction) as d:
            assert np.array_equal(y,d['y_true']);assert np.array_equal(indices,d['indices'])
            assert np.isfinite(d['logits']).all();assert np.array_equal(d['y_pred'],d['logits'].argmax(1))
            pred=d['y_pred'];logits=d['logits'];cm=np.bincount(y*5+pred,minlength=25).reshape(5,5)
        assert np.array_equal(cm,pa['confusion_matrix'])
        with np.load(HISTORY/(stem+'_predictions.npz')) as old:
            n=int((pred!=old['y_pred']).sum());assert n==pa['historical_prediction_changes']
            if not row['affected']:check_parity(logits,old['mean_logits'])
        counter=OUT/'old_input_cuda'/(stem+'_predictions.npz')
        ca=json.loads((OUT/'counterfactual_audit'/(stem+'.json')).read_text())
        assert ca['amendment_sha256']==sha(AMEND) and ca['archive_sha256']==sha(counter)
        assert ca['device']=='NVIDIA TITAN V'
        with np.load(counter) as d:
            assert np.array_equal(d['y_true'],y) and np.array_equal(d['indices'],indices)
            assert np.array_equal(d['y_pred'],d['logits'].argmax(1))
            with np.load(HISTORY/(stem+'_predictions.npz')) as old:
                parity=check_parity(d['logits'],old['mean_logits']);assert parity==ca['historical_parity']
            if not row['affected']:assert np.array_equal(logits,d['logits'])
            else:assert np.all(d['y_pred']==4)
            old_cuda_total+=np.bincount(y*5+d['y_pred'],minlength=25).reshape(5,5)
            numeric_changes+=parity['prediction_changes']
        if row['affected']:changed.append(stem)
        assert pa['device']=='NVIDIA TITAN V';total+=cm;subjects.add(row['subject']);epochs+=len(y)
        print('Verified',stem,flush=True)
    assert len(changed)==7 and len(subjects)==22 and epochs==42471
    score=json.loads((OUT/'st_corrected_statistics.json').read_text());m=metrics_from_confusion(total)
    assert np.array_equal(total,score['confusion_matrix'])
    for k in ('accuracy','macro_f1','weighted_f1','kappa'):assert abs(m[k]-score['point_estimate'][k])<1e-12
    new=json.loads((OUT/'pre_submission_statistics_corrected.json').read_text())
    old=json.loads((ROOT/'outputs/external_validation/pre_submission/pre_submission_statistics.json').read_text())
    for c in ('SHHS1','SHHS2','ISRUC_Cohort_I'):assert new['cohorts'][c]==old['cohorts'][c]
    assert new['cohorts']['SleepEDF_ST']==score
    pair=json.loads((OUT/'paired_input_repair_cuda.json').read_text())
    oldm=metrics_from_confusion(old_cuda_total)
    for k in ('accuracy','macro_f1','weighted_f1','kappa'):
        assert abs(oldm[k]-pair['old'][k])<1e-12
        assert abs(m[k]-pair['corrected'][k])<1e-12
        assert abs((m[k]-oldm[k])-pair['corrected_minus_old'][k])<1e-12
    save_json(OUT/'audit_status.json',{'status':'passed','protocol_sha256':digest,'records':44,'subjects':22,'epochs':42471,
              'amendment_sha256':sha(AMEND),'all44_raw_parity_verified':True,'old_artifacts_preserved':True,
              'unaffected37_logits_fp32_parity':True,'historical_cpu_to_cuda_near_tie_prediction_changes':numeric_changes,
              'same_cuda_old_new_counterfactual_verified':True,'cpu_model_execution':False})
    save_json(OUT/'status.json',{'status':'complete','protocol_sha256':digest,'amendment_sha256':sha(AMEND),'final_audit':'passed'})
    print('PASS fullSTrepair',44,22,42471,flush=True)


if __name__=='__main__':main()

"""Independent archive/statistics consistency audit; no neural inference."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_corrected_preprocessing_sensitivity import OUT,REPAIR,COHORTS,MODES,KEYS,metrics,verify,sha,save_json
from scripts.run_mainline_preprocessing_sensitivity import decision
from scripts.resume_st_repair_cuda import check_parity

def main():
    p=json.loads((OUT/'protocol.json').read_text());digest=sha(OUT/'protocol.json');verify(p)
    summary=json.loads((OUT/'summary.json').read_text());assert summary['protocol_sha256']==digest
    counts={};excluded=[]
    for cohort in COHORTS:
        expected=[r for r in p['records'] if r['cohort']==cohort]
        total={m:np.zeros((5,5),dtype=np.int64) for m in MODES};subjects=set();epochs=0;records=0;parity_records=0;by_subject={}
        for row in expected:
            stem=row['record'];meta=json.loads((OUT/cohort/(stem+'.json')).read_text())
            assert meta['protocol_sha256']==digest and meta['subject']==row['subject']
            for k,inventory in row['input_inventory'].items():
                stat=Path(row[k]).stat();assert inventory=={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            for k,value in row['small_artifact_sha256'].items():assert sha(Path(row[k]))==value
            if 'error' in meta:
                assert cohort in ('shhs1','shhs2') and meta['exclusion']=='missing channel, no substitute'
                assert meta in summary['cohorts'][cohort]['exclusions'];excluded.append({'cohort':cohort,'record':stem,'error':meta['error']});continue
            archive=OUT/cohort/(stem+'.npz');assert sha(archive)==meta['archive_sha256']
            assert meta['device']=='NVIDIA TITAN V' and meta['reader_exact_parity']
            with np.load(archive) as data:
                y=data['y'].astype(int);indices=data['indices']
                assert len(y)==meta['n_epochs'] and np.all((y>=0)&(y<5)) and np.all(np.diff(indices)>0)
                assert all(data[m].shape==(len(y),5) and np.isfinite(data[m]).all() for m in MODES)
                for m in MODES:
                    cm=np.bincount(5*y+data[m].argmax(1),minlength=25).reshape(5,5)
                    assert np.array_equal(cm,meta['confusion'][m]);total[m]+=cm
                    if row['subject'] not in by_subject:by_subject[row['subject']]={mode:np.zeros((5,5),dtype=np.int64) for mode in MODES}
                    by_subject[row['subject']][m]+=cm
                if cohort=='st':
                    assert np.array_equal(data['original'],data['antialias'])
                    assert np.array_equal(data['emg_envelope'],data['both'])
                    with np.load(row['npz']) as repaired:assert np.array_equal(y,repaired['y']) and np.array_equal(indices,repaired['indices'])
                if Path(row['historical']).exists():
                    with np.load(row['historical']) as hist:
                        assert np.array_equal(y,hist['y_true']) and np.array_equal(data['original'].argmax(1),hist['y_pred'])
                        if cohort=='st':assert check_parity(data['original'],hist['logits'])==meta['historical_logit_parity']
                    assert meta['historical_prediction_mismatches']==0;parity_records+=1
            records+=1;epochs+=len(y);subjects.add(row['subject'])
        value=summary['cohorts'][cohort]
        assert value['requested']==len(expected) and value['records']==records and value['subjects']==len(subjects) and value['n_epochs']==epochs
        assert value['records_with_historical_parity']==parity_records
        for m in MODES:
            assert np.array_equal(total[m],value['confusion'][m]);computed=metrics(total[m])
            for k in KEYS:assert abs(computed[k]-value['table'][m][k])<1e-12
            if m!='original':
                for k in KEYS:assert abs(computed[k]-value['table']['original'][k]-value['deltas'][m][k])<1e-12
        if cohort=='st':
            repaired=json.loads((REPAIR/'st_corrected_statistics.json').read_text())
            assert np.array_equal(total['original'],repaired['confusion_matrix']) and records==44 and epochs==42471 and len(subjects)==22
        # Recalculate all paired intervals from audited archives, not reported CMs.
        matrices={m:np.stack([by_subject[s][m] for s in sorted(subjects)]) for m in MODES}
        rng=np.random.default_rng(20261011);draws={m:{k:[] for k in KEYS} for m in MODES[1:]}
        for _ in range(5000):
            ids=rng.integers(0,len(subjects),size=len(subjects));base=metrics(matrices['original'][ids].sum(0))
            for mode in MODES[1:]:
                result=metrics(matrices[mode][ids].sum(0))
                for k in KEYS:draws[mode][k].append(result[k]-base[k])
        for mode in MODES[1:]:
            for k in KEYS:
                np.testing.assert_allclose(np.quantile(draws[mode][k],[.025,.975]),value['paired_subject_bootstrap_delta_95ci'][mode][k],rtol=0,atol=1e-12)
        counts[cohort]={'requested':len(expected),'records':records,'subjects':len(subjects),'epochs':epochs}
        print('PASS',cohort,counts[cohort],flush=True)
    for m in MODES[1:]:assert decision(summary['cohorts'],m)==summary['resource_gates'][m]
    verify(p)
    save_json(OUT/'audit_status.json',{'status':'passed','protocol_sha256':digest,'audit_code_sha256':sha(Path(__file__)),
        'cohorts':counts,'exclusions':excluded,'all_archives_confusions_metrics_and_gates_verified':True,'all_paired_bootstrap_intervals_recalculated':True,
        'corrected_ST_original_confusion_exact':True,'ST_negative_controls_exact':True,'old_pilot_not_reused':True,'cpu_neural_execution':False})
    save_json(OUT/'status.json',{'status':'complete','protocol_sha256':digest,'audit':'passed'})
    print('PASS complete corrected preprocessing sensitivity',flush=True)

if __name__=='__main__':main()

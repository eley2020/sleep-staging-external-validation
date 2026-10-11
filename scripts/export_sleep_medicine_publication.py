"""Regenerate current publication tables/figures from data-free aggregate inputs.

This entry point never loads raw PSG, labels, subject predictions or checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STAGES=('W','N1','N2','N3','REM')

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def export(data,out):
    out.mkdir(parents=True,exist_ok=True)
    cohorts=data['external'];source=data['source'];order=('SleepEDF_ST','SHHS1','SHHS2','ISRUC_Cohort_I')
    labels=('ST (corrected)','SHHS1','SHHS2\nfollow-up','ISRUC I')
    rows=[];stagerows=[]
    for cohort in order:
        v=cohorts[cohort];row={'cohort':cohort,**{k:v[k] for k in ('n_subjects','n_records','n_epochs')}}
        for k in ('accuracy','macro_f1','weighted_f1','kappa'):
            row[k]=v['point_estimate'][k];row[k+'_ci_low'],row[k+'_ci_high']=v['subject_bootstrap_95ci'][k]
        rows.append(row)
        for stage in STAGES:
            r=v['classification_report'][stage];s=v['stage_sensitivity_specificity'][stage]
            stagerows.append({'cohort':cohort,'stage':stage,'precision':r['precision'],'recall':r['recall'],'f1':r['f1-score'],'specificity':s['specificity'],'support':int(r['support'])})
    write_csv(out/'table1_2_frozen_external.csv',rows);write_csv(out/'tableS2_external_stages.csv',stagerows)
    write_csv(out/'tableS1_available_record_flow.csv',data['flow'])
    fig,ax=plt.subplots(figsize=(11,3.8),constrained_layout=True);ax.axis('off')
    from matplotlib.patches import FancyBboxPatch
    for y,row in zip((.67,.19),data['flow']):
        values=[f'{row["cohort"].upper()}\nAvailable EDF: {row["available_signal_files"]}',
            f'Matched annotation\npairs: {row["matched_signal_annotation_pairs"]}\nUnmatched: {row["unmatched_signal_files"]}',
            f'Excluded: {row["matched_pairs_excluded"]}\n'+('Missing EEG(sec)' if row['matched_pairs_excluded'] else 'No matched exclusions'),
            f'Analyzed records\n{row["evaluated_records"]}']
        for x,text in zip((.02,.27,.52,.77),values):
            ax.add_patch(FancyBboxPatch((x,y),.20,.22,boxstyle='round,pad=.012',facecolor='#E9F0F5',edgecolor='#40627D',transform=ax.transAxes))
            ax.text(x+.10,y+.11,text,ha='center',va='center',transform=ax.transAxes,fontsize=9)
            if x<.7:ax.annotate('',xy=(x+.25,y+.11),xytext=(x+.205,y+.11),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#40627D'})
    ax.set_title('SHHS available-file analysis flow (not original recruitment counts)',fontsize=12)
    save_fig(fig,out,'figureS1_record_flow')
    sourcerows=[{'metric':k,'fold_mean':source['final_test_'+k+'_mean'],'fold_sd':source['final_test_'+k+'_std'],'note':'source20fold means, not external pooled estimator'} for k in ('accuracy','macro_f1','weighted_f1','kappa')]
    write_csv(out/'table_source_reference.csv',sourcerows)
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True,constrained_layout=True)
    for ax,k,title in zip(axes,('accuracy','macro_f1','kappa'),('Accuracy','Macro-F1','Kappa')):
        values=np.array([cohorts[c]['point_estimate'][k] for c in order]);ci=np.array([cohorts[c]['subject_bootstrap_95ci'][k] for c in order]);x=np.arange(4)
        ax.bar(x,values,color=['#7A9E9F','#244B74','#5C7EA5','#B66D45'])
        ax.errorbar(x,values,yerr=[values-ci[:,0],ci[:,1]-values],fmt='none',color='#222222',capsize=3)
        ax.axhline(source['final_test_'+k+'_mean'],color='#111111',linestyle='--',label='Source fold mean')
        ax.set_xticks(x,labels,fontsize=8);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Score; subject-bootstrap 95% CI');axes[0].legend(fontsize=7,frameon=False)
    save_fig(fig,out,'figure1_external_validation')
    fig,axes=plt.subplots(1,3,figsize=(10.4,3.4),constrained_layout=True)
    for ax,c,title in zip(axes,order[1:],('SHHS1','SHHS2 follow-up','ISRUC I')):
        cm=np.array(cohorts[c]['confusion_matrix'],dtype=float);cm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues')
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if cm[i,j]>.5 else '#111111')
        ax.set_xticks(range(5),STAGES,fontsize=8);ax.set_yticks(range(5),STAGES,fontsize=8);ax.set_xlabel('Predicted');ax.set_ylabel('Reference');ax.set_title(title)
    fig.colorbar(im,ax=axes,shrink=.8,label='Reference-row proportion');save_fig(fig,out,'figure4_external_confusion_matrices')
    budget=data['shhs_budget'];write_csv(out/'table3_shhs1_budget.csv',budget)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    x=[r['budget_subjects'] for r in budget]
    axes[0].plot(x,[r['frozen_mean'] for r in budget],'o--',label='Matched frozen')
    axes[0].errorbar(x,[r['adapted_mean'] for r in budget],yerr=[r['adapted_sd'] for r in budget],fmt='o-',capsize=3,label='Adapted')
    axes[1].errorbar(x,[r['delta_mean'] for r in budget],yerr=[r['delta_sd'] for r in budget],fmt='o-',capsize=3)
    for ax in axes:ax.set_xticks(x);ax.set_xlabel('Calibration subjects (+20 validation)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Macro-F1; split SD');axes[1].set_ylabel('Paired gain; split SD');axes[0].legend(frameon=False,fontsize=8)
    save_fig(fig,out,'figure2_shhs1_budget_curve')
    clinicalrows=[]
    endpoints=('tst_hours','sleep_efficiency_pct','waso_minutes','rem_latency_minutes')
    clinical=data['clinical']
    for c,runs in clinical['cohorts'].items():
        for split,run in runs.items():
            for regime in ('frozen','adapted'):
                for endpoint in endpoints:
                    r=run[regime][endpoint]
                    clinicalrows.append({'cohort':c,'split':split,'regime':regime,'endpoint':endpoint,**r['point'],
                        'mae_ci_low':r['ci95']['mae'][0],'mae_ci_high':r['ci95']['mae'][1],
                        'bias_ci_low':r['ci95']['bias'][0],'bias_ci_high':r['ci95']['bias'][1]})
    write_csv(out/'table4_clinical_errors.csv',clinicalrows)
    write_csv(out/'tableS4_clinical_limits.csv',data['clinical_limits'])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,c in zip(axes,('shhs1','shhs2')):
        for mode,offset,color in (('frozen',-.18,'#6B7280'),('adapted',.18,'#1F6F8B')):
            values=np.array([[run[mode][e]['point']['mae'] for e in endpoints] for run in clinical['cohorts'][c].values()])
            ax.bar(np.arange(4)+offset,values.mean(0),width=.36,yerr=values.std(0,ddof=1),capsize=3,label=mode,color=color)
        ax.set_xticks(range(4),['TST(h)','SE(pp)','WASO(min)','REM latency(min)'],rotation=22,ha='right',fontsize=8);ax.set_title(c.upper());ax.set_ylabel('MAE; units differ by endpoint');ax.legend(fontsize=8,frameon=False)
    save_fig(fig,out,'figure3_clinical_mae')
    sensitivity=data['sensitivity'];rows=[]
    for c,v in sensitivity['cohorts'].items():
        for mode,values in v['table'].items():
            for k,value in values.items():
                ci=[None,None] if mode=='original' else v['paired_subject_bootstrap_delta_95ci'][mode][k]
                rows.append({'cohort':c,'mode':mode,'metric':k,'estimate':value,'delta':0. if mode=='original' else v['deltas'][mode][k],'delta_ci_low':ci[0],'delta_ci_high':ci[1]})
    write_csv(out/'tableS8_preprocessing.csv',rows)
    pointrows=[{'cohort':c,'mode':mode,**values} for c,v in sensitivity['cohorts'].items() for mode,values in v['table'].items()]
    write_csv(out/'tableS8a_preprocessing_points.csv',pointrows)
    write_csv(out/'tableS8b_preprocessing_paired_deltas.csv',[r for r in rows if r['mode']!='original'])
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,constrained_layout=True);scope=('st','isruc','shhs1','shhs2')
    for ax,mode in zip(axes,('antialias','emg_envelope','both')):
        for k,color,offset in (('macro_f1','#1f77b4',-.13),('n1_f1','#ff7f0e',0.),('n3_f1','#2ca02c',.13)):
            x=np.arange(4)+offset;values=100*np.array([sensitivity['cohorts'][c]['deltas'][mode][k] for c in scope]);ci=100*np.array([sensitivity['cohorts'][c]['paired_subject_bootstrap_delta_95ci'][mode][k] for c in scope])
            ax.vlines(x,ci[:,0],ci[:,1],color=color,alpha=.7);ax.scatter(x,values,color=color,label=k,s=25)
        ax.axhline(0,color='#777777');ax.set_xticks(range(4),['ST','ISRUC','SHHS1\nsample','SHHS2\nsample']);ax.set_title(mode);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Absolute score delta ×100; paired subject 95% CI');axes[1].legend(fontsize=8,frameon=False)
    save_fig(fig,out,'figureS8_preprocessing')
    # No individual participant values, raw paths or invalid ST adaptation summaries.
    (out/'publication_metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def save_fig(fig,out,name):
    for ext in ('pdf','png'):fig.savefig(out/f'{name}.{ext}',dpi=300)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('publication/publication_metrics.json'));p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise RuntimeError('Use a new empty export directory')
    export(json.loads(a.input.read_text()),a.output_dir);print('Exported aggregate publication tables/figures',a.output_dir)

if __name__=='__main__':main()

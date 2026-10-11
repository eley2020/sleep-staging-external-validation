"""Greedy growth from a minimal, decomposed AttnSleep network.

The minimal model keeps one temporal CNN branch, adaptive temporal pooling, a
projection and the five-class head. Original AttnSleep components are added one
at a time and accepted only by a locked, subject-disjoint validation rule.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if '--deterministic' in sys.argv:
    # Must be set before importing torch/CUDA to make cuBLAS workspace choices reproducible.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scripts.train_attnsleep_multichannel_baseline import (
        build_criterion, evaluate, make_loaders, parse_channel_indices,
        resolve_device, set_random_seed, split_train_validation_files_with_info,
    )
except ModuleNotFoundError:
    from train_attnsleep_multichannel_baseline import (
        build_criterion, evaluate, make_loaders, parse_channel_indices,
        resolve_device, set_random_seed, split_train_validation_files_with_info,
    )


LEGACY_ATOMS = ('multiscale_branch', 'afr_se', 'tce_context', 'quality_gate')
MODERN_ATOMS = ('convnext1d', 'geglu_mlp', 'eca_feature_gate', 'dilated_depthwise', 'unireplk1d', 'mambaout_gated', 'cbramod_lite')
ATOMS = LEGACY_ATOMS + MODERN_ATOMS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--np-data-dir', type=Path, default=Path('data/sleepedf20_multichannel_npz/eeg2_eog_emg_physical_volts'))
    p.add_argument('--attnsleep-dir', type=Path, default=Path('external_baselines/AttnSleep'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--fold-ids', default='3,10,12'); p.add_argument('--num-folds', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128); p.add_argument('--val-subjects-per-fold', type=int, default=2)
    p.add_argument('--base-epochs', type=int, default=5); p.add_argument('--growth-epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--lr-drop-epoch', type=int, default=2); p.add_argument('--lr-after-drop', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-3); p.add_argument('--channel-dropout', type=float, default=.15)
    p.add_argument('--min-delta', type=float, default=.01); p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto'); p.add_argument('--seed', type=int, default=7777)
    p.add_argument('--multi-metric-gate', action='store_true', help='Also require N1 F1/Kappa non-inferiority and a parameter-growth limit.')
    p.add_argument('--relaxed-directional-gate', action='store_true', help='Use positive-direction Macro-F1 and non-degradation gates instead of the fixed effect-size threshold.')
    p.add_argument('--min-n1-delta', type=float, default=-.005)
    p.add_argument('--min-kappa-delta', type=float, default=-.005)
    p.add_argument('--max-parameter-growth-ratio', type=float, default=.25)
    p.add_argument('--atoms', default=','.join(LEGACY_ATOMS)); p.add_argument('--max-rounds', type=int, default=4)
    p.add_argument('--force-all-growth', action='store_true', help='Keep adding the best validation candidate even when the acceptance threshold is not met.')
    p.add_argument('--skip-final-test', action='store_true', help='Run validation-only structure search without evaluating held-out test folds.')
    p.add_argument('--deterministic', action='store_true', help='Request deterministic CUDA/cuDNN behavior for reproducibility checks.')
    return p.parse_args()


class DecomposedAttnSleep(nn.Module):
    """AttnSleep components exposed as independently addable atoms."""

    def __init__(self, input_channels=4, atoms=(), channel_dropout=.15):
        super().__init__()
        from model.model import (ChannelQualityGate, EncoderLayer, MRCNN,
                                 MRCNN_SHHS, MultiHeadedAttention,
                                 PositionwiseFeedForward, SEBasicBlock, TCE,
                                 clones)
        self.input_channels = int(input_channels)
        self.atoms = tuple(atoms)
        self.channel_fusion = 'quality-gated' if 'quality_gate' in self.atoms else 'early'
        self.channel_gate = ChannelQualityGate(input_channels, channel_dropout) if 'quality_gate' in self.atoms else None
        source = MRCNN(30, input_channels=input_channels)
        self.features1 = source.features1
        self.features2 = source.features2 if 'multiscale_branch' in self.atoms else None
        in_channels = 256 if self.features2 is not None else 128
        afr_downsample = None
        if 'afr_se' in self.atoms and in_channels != 30:
            afr_downsample = nn.Sequential(
                nn.Conv1d(in_channels, 30, kernel_size=1, bias=False),
                nn.BatchNorm1d(30),
            )
        self.afr = SEBasicBlock(in_channels, 30, downsample=afr_downsample) if 'afr_se' in self.atoms else None
        proj_in = 30 if self.afr is not None else in_channels
        self.proj = nn.Conv1d(proj_in, 80, kernel_size=1, bias=True)
        self.tce = None
        if 'tce_context' in self.atoms:
            attn = MultiHeadedAttention(5, 80, 30)
            ff = PositionwiseFeedForward(80, 120, .1)
            layer = EncoderLayer(80, attn, ff, 30, .1)
            self.tce = TCE(layer, 2)
        self.modern_blocks = nn.ModuleDict()
        if 'convnext1d' in self.atoms:
            self.modern_blocks['convnext1d'] = ConvNeXt1DBlock(80)
        if 'geglu_mlp' in self.atoms:
            self.modern_blocks['geglu_mlp'] = GEGLUBlock(80)
        if 'eca_feature_gate' in self.atoms:
            self.modern_blocks['eca_feature_gate'] = ECAFeatureGate(80)
        if 'dilated_depthwise' in self.atoms:
            self.modern_blocks['dilated_depthwise'] = DilatedDepthwiseBlock(80)
        if 'unireplk1d' in self.atoms:
            self.modern_blocks['unireplk1d'] = UniRepLK1DBlock(80)
        if 'mambaout_gated' in self.atoms:
            self.modern_blocks['mambaout_gated'] = MambaOutGatedBlock(80)
        if 'cbramod_lite' in self.atoms:
            self.modern_blocks['cbramod_lite'] = CBraModLiteBlock(80)
        self.fc = nn.Linear(80 * 30, 5)

    def forward(self, x, channel_prior=None):
        if self.channel_gate is not None:
            x = self.channel_gate(x)
        z1 = F.adaptive_avg_pool1d(self.features1(x), 30)
        if self.features2 is not None:
            z2 = F.adaptive_avg_pool1d(self.features2(x), 30)
            z = torch.cat((z1, z2), dim=1)
        else:
            z = z1
        if self.afr is not None:
            z = self.afr(z)
        z = self.proj(z).transpose(1, 2)
        for block in self.modern_blocks.values():
            z = block(z)
        if self.tce is not None:
            z = self.tce(z)
        return self.fc(z.contiguous().view(z.shape[0], -1))


class ConvNeXt1DBlock(nn.Module):
    """ConvNeXt-style depthwise temporal convolution and pointwise MLP."""
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.pw1 = nn.Linear(dim, expansion * dim)
        self.pw2 = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)

    def forward(self, x):
        z = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        z = self.norm(z)
        z = self.pw2(F.gelu(self.pw1(z)))
        return x + self.gamma * z


class GEGLUBlock(nn.Module):
    """Gated MLP/GEGLU token mixer with a residual path."""
    def __init__(self, dim, hidden=160):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.out_proj = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)

    def forward(self, x):
        z = self.in_proj(self.norm(x))
        value, gate = z.chunk(2, dim=-1)
        return x + self.gamma * self.out_proj(value * F.gelu(gate))


class ECAFeatureGate(nn.Module):
    """Efficient channel-attention gate over the 80-dimensional token features."""
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        pooled = x.mean(dim=1, keepdim=True)
        weights = torch.sigmoid(self.conv(pooled))
        return x + self.scale * x * weights


class DilatedDepthwiseBlock(nn.Module):
    """Dilated depthwise temporal convolution for longer local context."""
    def __init__(self, dim, dilation=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, groups=dim)
        self.pwconv = nn.Conv1d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)

    def forward(self, x):
        z = self.norm(x)
        z = self.pwconv(F.gelu(self.dwconv(z.transpose(1, 2)))).transpose(1, 2)
        return x + self.gamma * z


class UniRepLK1DBlock(nn.Module):
    """Lightweight UniRepLKNet-style large-kernel re-parameterization probe."""
    def __init__(self, dim, large_kernel=31, small_kernel=5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.large = nn.Conv1d(dim, dim, large_kernel, padding=large_kernel // 2, groups=dim, bias=False)
        self.small = nn.Conv1d(dim, dim, small_kernel, padding=small_kernel // 2, groups=dim, bias=False)
        self.pw1 = nn.Linear(dim, dim * 2)
        self.pw2 = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)

    def forward(self, x):
        z = self.norm(x)
        local = self.large(z.transpose(1, 2)) + self.small(z.transpose(1, 2))
        local = local.transpose(1, 2)
        value, gate = self.pw1(local).chunk(2, dim=-1)
        return x + self.gamma * self.pw2(value * F.gelu(gate))


class MambaOutGatedBlock(nn.Module):
    """MambaOut-inspired gated convolution block without external Mamba dependencies."""
    def __init__(self, dim, kernel_size=7, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.in_proj = nn.Linear(dim, dim * expansion * 2)
        self.out_proj = nn.Linear(dim * expansion, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)

    def forward(self, x):
        z = self.norm(x)
        z = self.dw(z.transpose(1, 2)).transpose(1, 2)
        value, gate = self.in_proj(z).chunk(2, dim=-1)
        return x + self.gamma * self.out_proj(value * torch.sigmoid(gate))


class CBraModLiteBlock(nn.Module):
    """Criss-cross temporal/channel attention probe for multichannel EEG tokens."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.time_norm = nn.LayerNorm(dim)
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.channel_gate = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=dim, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-3)

    def forward(self, x):
        z = self.time_norm(x)
        attended, _ = self.time_attn(z, z, z, need_weights=False)
        channel_weights = self.channel_gate(x.transpose(1, 2)).transpose(1, 2)
        return x + self.gamma * attended * channel_weights


def compatible_load(model, state):
    current = model.state_dict()
    matched = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    current.update(matched)
    model.load_state_dict(current)
    return len(matched), len(current)


def train(model, loader, val_loader, criterion, device, args, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    best = (-np.inf, -np.inf); best_state = copy.deepcopy(model.state_dict()); best_metrics = None; history = []
    for epoch in range(1, epochs + 1):
        if epoch == args.lr_drop_epoch + 1:
            for group in optimizer.param_groups: group['lr'] = args.lr_after_drop
        model.train(); losses=[]; t0=time.time()
        for x, y, prior in loader:
            x=x.float().to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
            out=model(x, prior); loss=criterion(out,y); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        val=evaluate(model,val_loader,criterion,device)
        n1_f1=float(val['classification_report'].get('N1',{}).get('f1-score',0.0))
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_accuracy':val['accuracy'],'val_macro_f1':val['macro_f1'],'val_kappa':val['kappa'],'val_n1_f1':n1_f1,'lr':optimizer.param_groups[0]['lr'],'elapsed_sec':round(time.time()-t0,2)}
        history.append(row)
        key=(float(val['macro_f1']),float(val['accuracy']))
        if key > best:
            best=key; best_state=copy.deepcopy(model.state_dict())
            best_metrics={'best_val_macro_f1':float(val['macro_f1']),'best_val_accuracy':float(val['accuracy']),'best_val_kappa':float(val['kappa']),'best_val_n1_f1':n1_f1}
    model.load_state_dict(best_state)
    if best_metrics is None: raise RuntimeError('Validation loader produced no metrics')
    return best_metrics, history


def make_model(device, atoms, channel_dropout):
    return DecomposedAttnSleep(input_channels=4, atoms=atoms, channel_dropout=channel_dropout).to(device)


def configure_determinism(enabled):
    if not enabled:
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(args.attnsleep_dir.resolve()))
    from utils.util import calc_class_weight, load_folds_data
    configure_determinism(args.deterministic)
    device=resolve_device(args.device); folds=load_folds_data(str(args.np_data_dir.resolve()),args.num_folds)
    selected=[int(x.strip()) for x in args.fold_ids.split(',') if x.strip()]; remaining=[x.strip() for x in args.atoms.split(',') if x.strip()]
    if any(x not in ATOMS for x in remaining): raise ValueError(f'Unknown atom in {remaining}; allowed {ATOMS}')
    per_fold={}; active=()
    # Train the minimal model independently on each fold.
    for fold in selected:
        set_random_seed(args.seed + fold)
        train_files,test_files=folds[fold]; model_train,val_files,val_sids,_=split_train_validation_files_with_info(train_files,args.val_subjects_per_fold,'tail',parse_channel_indices('all'),(0,1))
        tr,va,counts,channels,_=make_loaders(model_train,val_files,args.batch_size,None)
        weights=torch.tensor(calc_class_weight(counts),dtype=torch.float32,device=device)
        criterion=build_criterion(argparse.Namespace(loss_type='cross-entropy',label_smoothing=0.,focal_gamma=2.,n1_loss_multiplier=1.,supcon_weight=0.),weights)
        set_random_seed(args.seed + fold)
        model=make_model(device,active,args.channel_dropout)
        set_random_seed(args.seed + fold)
        result,hist=train(model,tr,va,criterion,device,args,args.base_epochs)
        per_fold[fold]={'model_state':copy.deepcopy(model.state_dict()),'result':result,'history':hist,'train_files':train_files,'test_files':test_files,'val_files':val_files,'val_subjects':val_sids,'channels':channels}
    rounds=[]
    for round_id in range(1,args.max_rounds+1):
        if not remaining: break
        # The current structure's control is identical for every candidate in a
        # round. Train it once per fold and reuse it to reduce compute while
        # preserving the same initialization and stochastic training stream.
        round_context={}
        for fold in selected:
            info=per_fold[fold]; model_train=info['train_files']; val_files=info['val_files']
            tr,va,counts,_,_=make_loaders(model_train,val_files,args.batch_size,None)
            weights=torch.tensor(calc_class_weight(counts),dtype=torch.float32,device=device)
            criterion=build_criterion(argparse.Namespace(loss_type='cross-entropy',label_smoothing=0.,focal_gamma=2.,n1_loss_multiplier=1.,supcon_weight=0.),weights)
            init_seed=args.seed+fold+round_id*1000
            train_seed=args.seed+fold+round_id*100
            set_random_seed(init_seed)
            control=make_model(device,active,args.channel_dropout); compatible_load(control,info['model_state'])
            set_random_seed(train_seed)
            control_res,control_hist=train(control,tr,va,criterion,device,args,args.growth_epochs)
            round_context[fold]={'train_loader':tr,'val_loader':va,'criterion':criterion,'control':control_res,'control_history':control_hist,'control_parameters':sum(p.numel() for p in control.parameters()),'init_seed':init_seed,'train_seed':train_seed}
        candidate_results={}
        for atom in list(remaining):
            rows=[]
            for fold in selected:
                info=per_fold[fold]; context=round_context[fold]
                set_random_seed(context['init_seed'])
                candidate=make_model(device,active+(atom,),args.channel_dropout); compatible_load(candidate,info['model_state'])
                set_random_seed(context['train_seed'])
                cand_res,cand_hist=train(candidate,context['train_loader'],context['val_loader'],context['criterion'],device,args,args.growth_epochs)
                control_res=context['control']
                rows.append({'fold_id':fold,'control':control_res,'candidate':cand_res,'delta_macro_f1':cand_res['best_val_macro_f1']-control_res['best_val_macro_f1'],'delta_accuracy':cand_res['best_val_accuracy']-control_res['best_val_accuracy'],'delta_kappa':cand_res['best_val_kappa']-control_res['best_val_kappa'],'delta_n1_f1':cand_res['best_val_n1_f1']-control_res['best_val_n1_f1'],'history':cand_hist,'state':copy.deepcopy(candidate.state_dict())})
            deltas=[r['delta_macro_f1'] for r in rows]
            n1_deltas=[r['delta_n1_f1'] for r in rows]
            kappa_deltas=[r['delta_kappa'] for r in rows]
            control_parameters=int(round_context[selected[0]]['control_parameters'])
            candidate_parameters=sum(p.numel() for p in candidate.parameters())
            parameter_growth_ratio=(candidate_parameters-control_parameters)/control_parameters
            macro_threshold = 0.0 if args.relaxed_directional_gate else args.min_delta
            macro_gate=bool(np.mean(deltas)>macro_threshold if args.relaxed_directional_gate else (np.mean(deltas)>=macro_threshold and sum(x>0 for x in deltas)>=2))
            if args.relaxed_directional_gate:
                macro_gate = bool(macro_gate and sum(x>0 for x in deltas)>=2)
            secondary_gate=bool((np.mean(n1_deltas)>=0.0 if args.relaxed_directional_gate else np.mean(n1_deltas)>=args.min_n1_delta) and (np.mean(kappa_deltas)>=0.0 if args.relaxed_directional_gate else np.mean(kappa_deltas)>=args.min_kappa_delta) and parameter_growth_ratio <= (0.10 if args.relaxed_directional_gate else args.max_parameter_growth_ratio))
            candidate_results[atom]={'delta_macro_f1_mean':float(np.mean(deltas)),'delta_macro_f1_std':float(np.std(deltas)),'delta_n1_f1_mean':float(np.mean(n1_deltas)),'delta_n1_f1_std':float(np.std(n1_deltas)),'delta_kappa_mean':float(np.mean(kappa_deltas)),'delta_kappa_std':float(np.std(kappa_deltas)),'improved_folds':int(sum(x>0 for x in deltas)),'control_parameters':int(control_parameters),'candidate_parameters':int(candidate_parameters),'parameter_growth_ratio':float(parameter_growth_ratio),'macro_gate_pass':macro_gate,'secondary_gate_pass':secondary_gate,'accept':bool(macro_gate and (secondary_gate if args.multi_metric_gate else True)),'rows':rows}
        accepted_candidates=[name for name,result in candidate_results.items() if result['accept']]
        best_atom=max(accepted_candidates or list(candidate_results), key=lambda x:candidate_results[x]['delta_macro_f1_mean'])
        accepted=candidate_results[best_atom]['accept']; forced=bool(args.force_all_growth and not accepted)
        rounds.append({'round':round_id,'active_before':list(active),'candidates':{k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in candidate_results.items()},'best_atom':best_atom,'accepted':accepted,'forced_accept':forced})
        if forced: accepted = True
        if not accepted: break
        active=active+(best_atom,); remaining.remove(best_atom)
        for row in candidate_results[best_atom]['rows']:
            per_fold[row['fold_id']]['model_state']=row['state']
    # Locked final test evaluation after all validation decisions.
    final=[]
    if not args.skip_final_test:
        for fold in selected:
            info=per_fold[fold]; _,te,counts,_,_=make_loaders(info['train_files'],info['test_files'],args.batch_size,None)
            weights=torch.tensor(calc_class_weight(counts),dtype=torch.float32,device=device)
            criterion=build_criterion(argparse.Namespace(loss_type='cross-entropy',label_smoothing=0.,focal_gamma=2.,n1_loss_multiplier=1.,supcon_weight=0.),weights)
            model=make_model(device,active,args.channel_dropout); compatible_load(model,info['model_state']); test=evaluate(model,te,criterion,device)
            final.append({'fold_id':fold,'test_accuracy':test['accuracy'],'test_macro_f1':test['macro_f1'],'test_kappa':test['kappa']})
    final_means={k:float(np.mean([r[k] for r in final])) for k in ('test_accuracy','test_macro_f1','test_kappa')} if final else None
    acceptance_rule=('mean validation Macro-F1 delta > 0 and >=2/3 folds improved' if args.relaxed_directional_gate else f'mean validation Macro-F1 delta >= {args.min_delta:g} and >=2/3 folds improved')
    if args.multi_metric_gate:
        if args.relaxed_directional_gate:
            acceptance_rule += '; mean N1 F1 delta >= 0; mean Kappa delta >= 0; parameter growth ratio <= 0.1'
        else:
            acceptance_rule += f'; mean N1 F1 delta >= {args.min_n1_delta:g}; mean Kappa delta >= {args.min_kappa_delta:g}; parameter growth ratio <= {args.max_parameter_growth_ratio:g}'
    acceptance_rule += ' unless --force-all-growth is set'
    summary={'experiment':'minimal-decomposed-attnsleep-greedy-growth','minimal_atoms':['features1_single_scale_temporal_cnn','adaptive_temporal_pool_to_30','1x1_projection_to_80','five_class_linear_head'],'candidate_atoms':ATOMS,'selected_atoms':list(active),'rounds':rounds,'final_test':final if final else None,'final_test_means':final_means,'protocol':{'base_definition':'single features1 temporal branch + adaptive pooling + 1x1 projection + five-class head','folds':selected,'base_epochs':args.base_epochs,'growth_epochs':args.growth_epochs,'seed':args.seed,'deterministic':bool(args.deterministic),'control_reused_across_candidates_within_round':True,'multi_metric_gate':bool(args.multi_metric_gate),'relaxed_directional_gate':bool(args.relaxed_directional_gate),'min_macro_f1_delta':args.min_delta,'min_n1_f1_delta':args.min_n1_delta if args.multi_metric_gate and not args.relaxed_directional_gate else None,'min_kappa_delta':args.min_kappa_delta if args.multi_metric_gate and not args.relaxed_directional_gate else None,'max_parameter_growth_ratio':args.max_parameter_growth_ratio if args.multi_metric_gate and not args.relaxed_directional_gate else (0.10 if args.relaxed_directional_gate else None),'acceptance_rule':acceptance_rule,'forced_growth':bool(args.force_all_growth),'validation_only':bool(args.skip_final_test),'test_labels_used_for_selection':False}}
    (args.output_dir/'growth_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()

import unittest
import numpy as np
import torch

from run_mainline_preprocessing_sensitivity import antialias_epochs,envelope_epochs,modes_from_native,decision
from run_end_to_end_growth_study import ROOT


class ProcessingTests(unittest.TestCase):
    def test_100hz_negative_control_exact_identity(self):
        rng=np.random.default_rng(5); channels=[rng.normal(0,1e-5,(4,3000)).astype(np.float32) for _ in range(4)]
        modes=modes_from_native(channels,[100]*4)
        for a,b in zip(modes['original'],modes['antialias']): np.testing.assert_array_equal(a,b)
        for a,b in zip(modes['emg_envelope'],modes['both']): np.testing.assert_array_equal(a,b)

    def test_high_freq_alias_rejected(self):
        fs=200; native=np.sin(2*np.pi*80*np.arange(6000*3)/fs).astype(np.float32).reshape(3,6000)
        aa=antialias_epochs(native,fs)
        self.assertEqual(aa.shape,(3,3000))
        self.assertLess(np.sqrt(np.mean(aa**2)),.03)

    def test_envelope_nonnegative_lowrate_and_finite(self):
        for fs in (100,125,128,200):
            t=np.arange(fs*30*3)/fs
            native=((3+np.sin(2*np.pi*.1*t))*np.sin(2*np.pi*25*t)*1e-6).reshape(3,fs*30).astype(np.float32)
            e=envelope_epochs(native,fs)
            self.assertEqual(e.shape,(3,3000)); self.assertTrue(np.isfinite(e).all()); self.assertGreaterEqual(e.min(),0.)
            self.assertTrue(np.all(e.reshape(3,30,100)==e.reshape(3,30,100)[:,:,:1]))
            self.assertGreater(e.max(),1e-6)
        with self.assertRaises(ValueError): envelope_epochs(np.ones((3,30),dtype=np.float32),1)

    def test_gate_shhs_followup_not_counted_as_independent(self):
        keys=('macro_f1','accuracy','n1_recall','n1_f1','n3_f1')
        values={c:{'deltas':{'both':{k:0. for k in keys}}} for c in ('st','isruc','shhs1','shhs2')}
        for c in ('shhs1','shhs2'): values[c]['deltas']['both']['macro_f1']=.01
        self.assertFalse(decision(values,'both')['passes_resource_gate'])
        values['isruc']['deltas']['both']['macro_f1']=.01
        self.assertTrue(decision(values,'both')['passes_resource_gate'])
        values['st']['deltas']['both']['n3_f1']=-.02
        self.assertFalse(decision(values,'both')['passes_resource_gate'])


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class CudaTests(unittest.TestCase):
    def test_vectorized_identity_ensemble_vs_serial(self):
        from model.model import AttnSleep
        from torch.func import stack_module_state
        from scripts.evaluate_attnsleep_external_ensemble import predict_ensemble_logits
        folder=ROOT/'outputs/attnsleep_multichannel_baseline_train/full20_quality_gate_4ch_emg_30epoch'
        models=[]
        for i in (0,1):
            model=AttnSleep(input_channels=4,channel_fusion='quality-gated',channel_dropout=.15).cuda().eval()
            ck=torch.load(folder/f'fold_{i:02d}/checkpoint_best.pt',map_location='cuda',weights_only=False)
            model.load_state_dict(ck['model_state']); models.append(model)
        x=np.random.default_rng(5).normal(0,2e-5,(4,4,3000)).astype(np.float32)
        params,buffers=stack_module_state(models)
        actual=predict_ensemble_logits(models[0],params,buffers,x,64,[{'kind':'identity'}],torch.device('cuda'))
        with torch.inference_mode(): expected=torch.stack([m(torch.from_numpy(x).cuda()) for m in models]).mean(0).cpu().numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
        np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))


if __name__=='__main__': unittest.main()

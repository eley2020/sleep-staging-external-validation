"""Repair regression tests: mock read guards/lifecycle and real EDF+fallback."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.prepare_sleepedf20_multichannel import read_channel_epochs,prepare_one

ROOT=Path(__file__).resolve().parents[1]
CHANNELS=['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental']


class FakeReader:
    def __init__(self,values): self.values=values; self.closed=False; self.signals_in_file=4
    def getSignalLabels(self): return CHANNELS
    def getSampleFrequency(self,i): return 100
    def getPhysicalDimension(self,i): return 'uV'
    def getNSamples(self): return [len(self.values)]*4
    def readSignal(self,i): return self.values.copy()
    def getStartdatetime(self): return 'test'
    def getFileDuration(self): return len(self.values)/100
    def close(self): self.closed=True


class RepairTests(unittest.TestCase):
    def test_nonfinite_native_fails(self):
        reader=FakeReader(np.full(3000,np.nan,dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            read_channel_epochs(reader,CHANNELS[0],1,3000,'physical-volts',0,0,0,4)

    def test_silent_zero_native_fails(self):
        reader=FakeReader(np.zeros(3000,dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'all-zero native'):
            read_channel_epochs(reader,CHANNELS[0],1,3000,'physical-volts',0,0,0,4)

    def test_psg_closed_before_annotation(self):
        reader=FakeReader(np.ones(3000,dtype=np.float32))
        def annotation(*args):
            self.assertTrue(reader.closed)
            return np.array([2],dtype=np.int32)
        with patch('scripts.prepare_sleepedf20_multichannel.pyedflib.EdfReader',return_value=reader),patch('scripts.prepare_sleepedf20_multichannel.read_labels',side_effect=annotation):
            x,y,_=prepare_one(Path('fake.edf'),Path('fake_ann.edf'),CHANNELS,3000,30,'physical-volts',0,0,0,4)
        self.assertEqual(x.shape,(1,4,3000));self.assertTrue(np.any(x));self.assertEqual(y.tolist(),[2])

    def test_real_fallback_record_nonzero_labels_unchanged(self):
        raw=ROOT/'data/sleep-edfx-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-telemetry'
        x,y,_=prepare_one(raw/'ST7021J0-PSG.edf',raw/'ST7021JM-Hypnogram.edf',CHANNELS,3000,30,'physical-volts',0,0,0,4)
        with np.load(ROOT/'data/sleepedf_st_multichannel_npz/eeg2_eog_emg_physical_volts/ST7021J0.npz') as old:
            self.assertEqual(x.shape,old['x'].shape);np.testing.assert_array_equal(y,old['y'])
        self.assertTrue(np.isfinite(x).all());self.assertFalse(np.any(np.all(x==0,axis=(1,2))))

    def test_real_unaffected_record_exactly_unchanged(self):
        raw=ROOT/'data/sleep-edfx-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-telemetry'
        x,y,_=prepare_one(raw/'ST7011J0-PSG.edf',raw/'ST7011JP-Hypnogram.edf',CHANNELS,3000,30,'physical-volts',0,0,0,4)
        with np.load(ROOT/'data/sleepedf_st_multichannel_npz/eeg2_eog_emg_physical_volts/ST7011J0.npz') as old:
            np.testing.assert_array_equal(x,old['x']);np.testing.assert_array_equal(y,old['y'])


if __name__=='__main__': unittest.main()

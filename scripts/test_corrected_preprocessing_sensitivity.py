"""Regression checks for new protocol safeguards; no source/target fitting."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_corrected_preprocessing_sensitivity as study

class Tests(unittest.TestCase):
    def test_reject_nonfinite(self):
        x=np.ones((3,4,3000),dtype=np.float32);x[0,0,0]=np.nan
        result=({m:x for m in study.MODES},np.zeros(3,dtype=int),np.arange(3),{})
        with patch.object(study.old,'load_inputs',return_value=result):
            with self.assertRaises(RuntimeError):study.load_inputs({'cohort':'isruc'})

    def test_reject_zero_epoch(self):
        x=np.ones((3,4,3000),dtype=np.float32);x[1]=0
        result=({m:x for m in study.MODES},np.zeros(3,dtype=int),np.arange(3),{})
        with patch.object(study.old,'load_inputs',return_value=result):
            with self.assertRaises(RuntimeError):study.load_inputs({'cohort':'isruc'})

    def test_reject_zero_channel(self):
        x=np.ones((3,4,3000),dtype=np.float32);x[:,2]=0
        result=({m:x for m in study.MODES},np.zeros(3,dtype=int),np.arange(3),{})
        with patch.object(study.old,'load_inputs',return_value=result):
            with self.assertRaises(RuntimeError):study.load_inputs({'cohort':'isruc'})

    def test_locked_scope_and_corrected_paths(self):
        p=json.loads((study.OUT/'protocol.json').read_text())
        self.assertEqual({c:sum(r['cohort']==c for r in p['records']) for c in study.COHORTS},{'st':44,'isruc':100,'shhs1':32,'shhs2':32})
        for row in p['records']:
            if row['cohort']=='st':
                self.assertEqual(Path(row['npz']).parent,study.DATA)
                self.assertEqual(Path(row['historical']).parent,study.REPAIR/'predictions')
        study.verify(p)

if __name__=='__main__':unittest.main()

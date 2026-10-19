from pathlib import Path
import sys
import unittest
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from calibration_experiment import prepare,quantize_model,run


class Experiment(unittest.TestCase):
    def test_disjoint_episode_splits_and_no_reference_mutation(self):
        reference,parts,_=prepare(epochs=3)
        seen=set()
        for rows in parts.values():
            ids={r.episode for r in rows};self.assertFalse(seen&ids);seen|=ids
        original={k:v.clone() for k,v in reference.state_dict().items()}
        candidate,artifacts,report=quantize_model(reference,parts['calibration'],'output_mse')
        import torch
        for k,v in reference.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
        self.assertEqual(len(artifacts),3);self.assertLess(report['packed_reference_payload_bytes'],sum(v.numel()*v.element_size() for v in reference.state_dict().values()))

    def test_complete_run_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:report=run(directory,epochs=3)
        for v in report['policies'].values():
            self.assertEqual(v['reload_max_abs_logit_error'],0)
            self.assertEqual(v['test']['examples'],576)
            self.assertEqual(v['shifted_test']['examples'],576)
            self.assertGreater(v['serialized_file_bytes'],v['packed_reference_payload_bytes'])

if __name__=='__main__':unittest.main()

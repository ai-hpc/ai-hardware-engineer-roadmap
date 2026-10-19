import json
import math
from pathlib import Path
import unittest
from quant_contracts import kv_payload_bytes, fp4_payload_bytes, speculative_mass, expected_emitted, exhaustive_savings


class Contracts(unittest.TestCase):
    def test_hybrid_metadata_not_all_layers(self):
        evidence=json.loads(Path(__file__).with_name('model-evidence.json').read_text())
        target=evidence['models'][0]; c=target['configuration']
        self.assertEqual((c['hidden_size'],c['vocab_size']),(5120,248320))
        self.assertEqual(len(c['layer_types']),64)
        self.assertEqual(c['layer_types'].count('full_attention'),16)
        self.assertEqual(c['mtp_num_hidden_layers'],0)
        self.assertEqual(target['index']['mtp_named_tensors'],0)
        self.assertIn('lm_head.weight_scale',target['index']['output_head_keys'])
        self.assertEqual(target['index']['total_tensor_bytes'],17915815528)
        self.assertEqual(kv_payload_bytes(1,262144,16,4,256,2),16*2**30)

    def test_storage_includes_tails_scales_global(self):
        self.assertEqual(fp4_payload_bytes(1,16),13)
        self.assertEqual(fp4_payload_bytes(1,17),22)
        self.assertEqual(fp4_payload_bytes(32,4),292)
        self.assertEqual(fp4_payload_bytes(1024,1024),589828)

    def test_exact_speculative_correction(self):
        for p,q in [([.8,.2],[.2,.8]),([0.,1.],[1.,0.]),([.2,.3,.5],[.2,.3,.5]),([.5,.5,0.],[0.,.5,.5])]:
            r=speculative_mass(p,q)
            for a,b in zip(r['output'],p):self.assertAlmostEqual(a,b)
            self.assertAlmostEqual(r['acceptance'],1-.5*sum(abs(a-b) for a,b in zip(p,q)))

    def test_wrong_rejection_shortcut_is_biased(self):
        p,q=[.8,.2],[.2,.8];r=speculative_mass(p,q)
        wrong=[a+r['rejection']*b for a,b in zip(r['accepted_mass'],p)]
        self.assertAlmostEqual(wrong[0],.68)
        self.assertGreater(abs(wrong[0]-p[0]),.1)

    def test_acceptance_is_not_reference_quality(self):
        reference=[.9,.1];draft=[.1,.9]
        self.assertLess(speculative_mass(reference,draft)['acceptance'],speculative_mass(draft,draft)['acceptance'])
        self.assertAlmostEqual(.5*sum(abs(a-b) for a,b in zip(reference,draft)),.8)

    def test_prefix_survival_not_mean_acceptance_power(self):
        self.assertAlmostEqual(expected_emitted([.8,.6,.3]),2.7)
        self.assertAlmostEqual(expected_emitted([.8,.8**2,.8**3]),2.952)
        with self.assertRaises(ValueError):expected_emitted([.4,.6])

    def test_greedy_counterexample(self):
        best=exhaustive_savings([(6,4),(4,3),(4,3)],6)
        self.assertEqual(best,{'saved':8,'cost':6,'selected':(False,True,True)})

    def test_invalid_inputs(self):
        for p,q in [([.5],[1.]),([float('nan'),0.],[1.,0.]),([1.],[.5,.5])]:
            with self.assertRaises(ValueError):speculative_mass(p,q)
        with self.assertRaises(ValueError):fp4_payload_bytes(0,16)
        with self.assertRaises(ValueError):kv_payload_bytes(1,-1,16,4,256,2)

if __name__=='__main__':unittest.main()

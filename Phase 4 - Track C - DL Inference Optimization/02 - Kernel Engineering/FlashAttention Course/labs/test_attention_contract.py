import unittest
from attention_contract import causal_mask, traffic, kv_bytes, gather_pages


class Contracts(unittest.TestCase):
    def test_rectangular_alignment_is_semantic(self):
        self.assertEqual(causal_mask(1,5,'upper_left'),[[True,False,False,False,False]])
        self.assertEqual(causal_mask(1,5,'lower_right'),[[True]*5])
        self.assertEqual(causal_mask(5,2,'lower_right')[:3],[[False,False]]*3)

    def test_square_alignment_agrees(self):
        self.assertEqual(causal_mask(4,4,'upper_left'),causal_mask(4,4,'lower_right'))

    def test_fixed_tile_traffic_is_not_linear_in_sequence(self):
        a=traffic(4096,4096,64,64,64);b=traffic(8192,8192,64,64,64)
        ratio=b['query_outer_logical_bytes']/a['query_outer_logical_bytes']
        self.assertGreater(ratio,3.9); self.assertLess(ratio,4)
        self.assertEqual(b['materialized_bytes'],8*8192**2+8*8192*64)

    def test_tail_counts_actual_elements_and_column_tile_not_an_io_free_pass(self):
        report=traffic(5,7,3,2,2)
        self.assertEqual(report['query_tiles'],3)
        self.assertEqual(report['query_outer_logical_bytes'],(5*5+3*7*5)*2+5*4)

    def test_cache_counts_both_k_and_v(self):
        self.assertEqual(kv_bytes(16384,8,128),64*1024**2)
        self.assertEqual(kv_bytes(16384,8,128,layers=32),2*1024**3)

    def test_page_mapping_and_partial_last_page(self):
        pages=[[40,41,42,43],[10,11,12,13],[30,31,32,33]]
        self.assertEqual(gather_pages(pages,[2,1],6),[30,31,32,33,10,11])
        for table in ([2], [2,9], [2,True]):
            with self.assertRaises(ValueError):gather_pages(pages,table,6)

    def test_invalid_contracts_rejected(self):
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError):kv_bytes(value,8,128)
        with self.assertRaises(ValueError):causal_mask(2,5,'automatic')


if __name__=='__main__':unittest.main()

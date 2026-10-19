import copy
from pathlib import Path
import tempfile
import unittest
from local_rag import Index, Principal, dispatch, evaluate


class RetrievalTests(unittest.TestCase):
    def setUp(self):self.index=Index()
    def tearDown(self):self.index.close()

    def test_actual_fts_current_revision_and_private_exclusion(self):
        self.assertEqual(self.index.retrieve('battery return threshold',1)[0]['id'],'battery-v2')
        self.assertNotIn('battery-v1',self.index.documents)
        self.assertNotIn('maintenance',self.index.documents)
        self.assertNotIn('future',self.index.documents)
        self.assertEqual(self.index.retrieve('VIOLET'),[])
        self.assertTrue(all(row['id'] != 'maintenance' for row in self.index.retrieve('VIOLET-ONLY')))
        # An irrelevant public hit is a relevance failure, not a private-data disclosure.

    def test_revision_boundary_is_half_open(self):
        old=Index(as_of='2026-08-31');new=Index(as_of='2026-09-01')
        try:
            self.assertEqual(old.retrieve('battery',1)[0]['id'],'battery-v1')
            self.assertEqual(new.retrieve('battery',1)[0]['id'],'battery-v2')
        finally:old.close();new.close()

    def test_trusted_roles_scope_index_and_cache(self):
        privileged=Index(Principal('maintainer',frozenset({'public','operator'})))
        try:
            self.assertEqual(privileged.retrieve('VIOLET-ONLY')[0]['id'],'maintenance')
            self.assertNotEqual(privileged.cache_key('battery'),self.index.cache_key('battery'))
            self.assertNotEqual(self.index.cache_key('battery',1),self.index.cache_key('battery',3))
        finally:privileged.close()

    def test_citations_bind_identity_revision_span_and_hash(self):
        citation=self.index.excerpt('camera checkerboard')['citations'][0]
        self.assertTrue(self.index.verify_citation(citation))
        for key,value in [('id','maintenance'),('revision','wrong'),('start',-1),('end',99999),('quote','invented'),('text_sha256','bad')]:
            bad={**citation,key:value};self.assertFalse(self.index.verify_citation(bad))

    def test_operator_injection_is_data_and_tool_schema_has_no_identity(self):
        self.assertTrue(self.index.retrieve('ignore previous instructions'))
        bad_calls=[{'name':'write_file','arguments':{}},
                   {'name':'search_manual','arguments':{'query':'battery','k':1,'roles':['operator']}},
                   {'name':'search_manual','arguments':{'query':'battery','k':True}}]
        for call in bad_calls:
            with self.assertRaises(ValueError):dispatch(self.index,call)
        self.assertEqual(dispatch(self.index,{'name':'search_manual','arguments':{'query':'battery','k':1}})[0]['id'],'battery-v2')
        # MATCH operators and punctuation are reduced to literal words, never SQL commands.
        self.index.retrieve('" OR *; DROP TABLE chunks; --')
        self.assertTrue(self.index.retrieve('camera'))

    def test_empty_unknown_and_invalid_queries(self):
        self.assertTrue(self.index.excerpt('underwater pressure rating')['abstained'])
        self.assertTrue(self.index.excerpt('the and of')['abstained'])
        for q,k in [('',1),('x'*501,1),('camera',0),('camera',6)]:
            with self.assertRaises(ValueError):self.index.retrieve(q,k)

    def test_metrics_use_relevance_denominator_and_unanswerable_separately(self):
        cases=[{'id':'a','question':'camera','relevant':['camera','audio']},
               {'id':'b','question':'zzunseen','relevant':[]}]
        d=evaluate(self.index,cases,1)
        self.assertEqual(d['macro_recall'],.5)
        self.assertEqual(d['mrr'],1)
        self.assertEqual(d['abstention_accuracy'],1)
        self.assertIsNone(d['cases'][1]['recall'])


if __name__=='__main__':unittest.main()

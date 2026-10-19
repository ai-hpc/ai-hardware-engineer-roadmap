import unittest
from backlog import queue_trace


class BacklogTests(unittest.TestCase):
    def test_known_fifo_ages(self):
        result=queue_trace(3,10,25)
        self.assertEqual([x['age_ms'] for x in result['completed']],[25,40,55])
        self.assertEqual(result['dropped'],[])

    def test_latest_pending_does_not_preempt_current_job(self):
        result=queue_trace(3,10,25,True)
        self.assertEqual([x['sequence'] for x in result['completed']],[0,2])
        self.assertEqual([x['age_ms'] for x in result['completed']],[25,30])
        self.assertEqual(result['dropped'],[1])

    def test_equal_arrival_and_completion_has_declared_order(self):
        result=queue_trace(4,10,20,True)
        self.assertEqual([x['sequence'] for x in result['completed']],[0,2,3])
        self.assertEqual(result['dropped'],[1])

    def test_worker_faster_than_producer_has_no_backlog(self):
        for latest in [False,True]:
            result=queue_trace(10,20,5,latest)
            self.assertTrue(all(x['age_ms']==5 for x in result['completed']))
            self.assertEqual(result['dropped'],[])

    def test_accounting_no_duplicate_or_lost_identity(self):
        result=queue_trace(101,7,19,True)
        ids=[x['sequence'] for x in result['completed']]+result['dropped']
        self.assertEqual(sorted(ids),list(range(101)))

    def test_latest_only_cannot_fix_slow_inference(self):
        result=queue_trace(5,10,100,True)
        self.assertTrue(all(x['age_ms']>=100 for x in result['completed']))

    def test_invalid_configuration(self):
        for args in [(0,10,20), (3,0,20), (3,10,-1), (True,10,20), (3,10.5,20), (100001,1,1)]:
            with self.assertRaises(ValueError):queue_trace(*args)
        with self.assertRaises(ValueError):queue_trace(latest_only=1)


if __name__=='__main__':unittest.main()

"""Exercise the actual 2D guide modules; requires NumPy/SciPy, no detector/GPU."""
import importlib
from pathlib import Path
import re
import sys
import tempfile
import unittest
import numpy as np


class TestGuideTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix='course-tracker-test-')
        guide = Path(__file__).resolve().parents[2]/'Guide.md'
        blocks = re.findall(r'```python\n(.*?)```', guide.read_text(), re.S)
        for name in ['kalman_filter.py', 'track.py', 'tracker.py']:
            matches = [b for b in blocks if b.startswith('# '+name+'\n')]
            if len(matches) != 1:
                raise AssertionError('Expected one complete module: '+name)
            Path(cls.folder.name, name).write_text(matches[0])
        sys.path.insert(0, cls.folder.name)
        cls.module = importlib.import_module('tracker')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.folder.name)
        for name in ['tracker','track','kalman_filter']:
            sys.modules.pop(name, None)
        cls.folder.cleanup()

    def test_gating_precedes_assignment_and_allows_unmatched(self):
        c=np.array([[.1,.2],[.2,.21]])
        pairs,ut,ud=self.module.gated_assignment(c,c<=.2)
        self.assertEqual(set(pairs),{(0,1),(1,0)})
        self.assertEqual((ut,ud),([],[]))
        self.assertEqual(self.module.gated_assignment(c,np.zeros_like(c,dtype=bool)),([], [0,1], [0,1]))

    def test_deletion_preserves_surviving_id_and_class(self):
        tracker=self.module.Tracker(max_age=1,min_hits=1)
        rows=tracker.update([[0,0,10,10,.9,2],[100,100,110,110,.8,7]])
        self.assertEqual(len(rows),2)
        survivor_id=rows[1][-2]
        tracker.update([[100,100,110,110,.8,7]])
        rows=tracker.update([[100,100,110,110,.8,7]])
        self.assertEqual(len(tracker.tracks),1)
        self.assertEqual((rows[0][-2],rows[0][-1]),(survivor_id,7))
        self.assertEqual(tracker.update([]),[])
        self.assertEqual(tracker.update([]),[])
        self.assertEqual(tracker.tracks,[])

    def test_confirmation_requires_declared_hits(self):
        tracker=self.module.Tracker(min_hits=2)
        self.assertEqual(tracker.update([[0,0,10,10,.9,1]]),[])
        self.assertEqual(len(tracker.update([[0,0,10,10,.9,1]])),1)

    def test_invalid_boxes_rejected(self):
        for box in [[0,0,0,3],[3,0,1,3],[0,0,np.nan,3]]:
            with self.assertRaises(ValueError):self.module.box_to_obs(box)


if __name__=='__main__':unittest.main()

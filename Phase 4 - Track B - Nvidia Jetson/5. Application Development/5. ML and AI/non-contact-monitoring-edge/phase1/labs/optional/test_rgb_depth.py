import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
PHASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PHASE))
from calibration import load_paras_txt, project_world_to_image
from depth_utils import gray_to_depth, depth_in_roi
from detection import CompositeDetector, PersonDetector
from signal_reference import peak_in_band


def calibration(camera_id):
    return f"camera_id {camera_id}\nresolution 160 128\nK_matrix 80 80 80 64\nR_matrix 1 0 0 0 1 0 0 0 1\nworld_position 0 0 0\n"


def fixture(root, fps_depth=10., depth_frames=4):
    (root/'Camera Parameters').mkdir()
    (root/'Camera Parameters/paras.txt').write_text(calibration(0)+calibration(1))
    for kind, fps, count in [('rgb',10.,4),('depth',fps_depth,depth_frames)]:
        writer = cv2.VideoWriter(str(root/f'camera02-{kind}.avi'), cv2.VideoWriter_fourcc(*'FFV1'), fps, (160,128))
        if not writer.isOpened():
            raise RuntimeError('test environment lacks FFV1 video writer')
        for i in range(count):
            writer.write(np.full((128,160,3), 1 if kind == 'depth' else 0, dtype=np.uint8))
        writer.release()


class RGBDepthTests(unittest.TestCase):
    def test_depth_encoding_is_explicit_even_for_dark_frame(self):
        byte = gray_to_depth(np.ones((2,2),dtype=np.uint8))
        normalized = gray_to_depth(np.ones((2,2)),encoding='normalized')
        self.assertGreater(byte[0,0],148)
        self.assertAlmostEqual(normalized[0,0],40)
        self.assertAlmostEqual(gray_to_depth(np.zeros((1,1)))[0,0],150)
        self.assertAlmostEqual(gray_to_depth(np.full((1,1),255),units_to_meters=.01)[0,0],.4)
        for x in (np.array([[np.nan]]),np.array([[256]]),np.array([])):
            with self.assertRaises(ValueError):gray_to_depth(x)

    def test_roi_is_clipped_and_nonfinite_depth_excluded(self):
        d=np.array([[np.nan,1],[np.inf,3.]])
        self.assertEqual(depth_in_roi(d,(-4,-4,4,4)),2)
        self.assertIsNone(depth_in_roi(d,(-4,0,-1,1)))
        self.assertIsNone(depth_in_roi(d,(2,0,3,2)))
        self.assertIsNone(depth_in_roi(d,(1,1,0,0)))

    def test_projection_camera_centre_and_invalid_depth(self):
        K=np.array([[100,0,20],[0,100,30],[0,0,1.]])
        self.assertTrue(np.allclose(project_world_to_image(K,np.eye(3),[1,0,0],[2,0,2]),[70,30]))
        self.assertEqual(project_world_to_image(K,np.eye(3),[0,0,0],[[0,0,2]]).shape,(1,2))
        with self.assertRaises(ValueError):project_world_to_image(K,np.eye(3),[0,0,0],[1,0,0])

    def test_bad_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'paras.txt'
            for text in ('camera_id 0\n',calibration(0)+calibration(0),calibration(0).replace('1 0 0 0 1 0 0 0 1','1 0 0 0 1 0 0 0 -1')):
                path.write_text(text)
                with self.assertRaises(ValueError):load_paras_txt(path)

    def test_actual_cpu_detectors_blank_frame_and_small_input(self):
        self.assertEqual(CompositeDetector().detect(np.zeros((128,160,3),dtype=np.uint8)),[])
        self.assertEqual(PersonDetector().detect(np.zeros((16,16,3),dtype=np.uint8)),[])

    def test_actual_headless_cli_selects_camera02(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            p=subprocess.run([sys.executable,str(PHASE/'run_pipeline.py'),'--dataset-dir',d,'--camera-index','1',
                              '--suffix','avi','--no-display','--no-person'],check=True,capture_output=True,text=True)
            report=json.loads(p.stdout)
            self.assertEqual(report['frames'],4)
            self.assertEqual(report['camera_video_prefix'],'camera02')
            self.assertEqual(report['depth_units'],'dataset units')

    def test_pair_fps_and_length_mismatch_fail(self):
        from run_pipeline import run
        for options in ({'fps_depth':12.},{'depth_frames':3}):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);fixture(root,**options)
                with self.assertRaises(ValueError):run(root,camera_index=1,suffix='avi',no_person=True)


class SignalTests(unittest.TestCase):
    def test_known_tone_and_frequency_resolution(self):
        t=np.arange(400)/20
        result=peak_in_band(t,2+np.sin(2*np.pi*1.2*t)+.1*np.sin(2*np.pi*4*t))
        self.assertAlmostEqual(result['peak_hz'],1.2)
        self.assertAlmostEqual(result['bin_spacing_hz'],.05)

    def test_alias_demonstrates_unidentifiable_source(self):
        t=np.arange(200)/10
        self.assertAlmostEqual(peak_in_band(t,np.sin(2*np.pi*8*t))['peak_hz'],2)

    def test_bad_timestamps_band_and_constant_signal(self):
        t=np.arange(100)/20;y=np.sin(2*np.pi*t)
        t[50]+=.02
        with self.assertRaises(ValueError):peak_in_band(t,y)
        with self.assertRaises(ValueError):peak_in_band(np.arange(100)/6,y)
        with self.assertRaises(ValueError):peak_in_band(np.arange(100)/20,np.ones(100))


if __name__=='__main__':unittest.main()

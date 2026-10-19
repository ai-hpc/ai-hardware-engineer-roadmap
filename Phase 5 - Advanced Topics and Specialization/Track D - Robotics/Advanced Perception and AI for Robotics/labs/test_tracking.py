import copy
import math
import unittest
from tracking import Pose,PoseHistory,detection_world,CVTrack,assign,demo


class GeometryTests(unittest.TestCase):
    def setUp(self):self.history=PoseHistory([Pose(0,0,0,0),Pose(1,1,0,0)])

    def test_capture_time_compensates_ego_motion(self):
        result=demo()
        self.assertAlmostEqual(result['capture_time_projection']['world_xy'][0],5.)
        self.assertAlmostEqual(result['capture_time_projection']['world_xy'][1],1.)
        self.assertAlmostEqual(result['wrong_time_error_m'],.2)

    def test_optical_right_is_body_negative_left(self):
        result=detection_world(self.history,capture=0,now=0,u=420,depth=2)
        self.assertEqual(result['world_xy'],[2.,-.5])

    def test_yaw_rotation_and_covariance(self):
        history=PoseHistory([Pose(0,0,0,math.pi/2),Pose(1,0,0,math.pi/2)])
        result=detection_world(history,capture=.5,now=.5,u=320,depth=2,
                               sigma_u=1,sigma_depth=.02,sigma_position=.01,sigma_yaw=0)
        self.assertAlmostEqual(result['world_xy'][0],0.)
        self.assertAlmostEqual(result['world_xy'][1],2.)
        p=result['covariance_xy']
        self.assertAlmostEqual(p[0][0],.000125)
        self.assertAlmostEqual(p[1][1],.0005)
        self.assertAlmostEqual(p[0][1],p[1][0])

    def test_angle_wrap_uses_short_arc(self):
        h=PoseHistory([Pose(0,0,0,math.radians(179)),Pose(1,0,0,math.radians(-179))])
        self.assertAlmostEqual(math.cos(h.at(.5).yaw),-1)
        ambiguous=PoseHistory([Pose(0,0,0,0),Pose(1,0,0,math.pi)])
        with self.assertRaises(ValueError):ambiguous.at(.5)

    def test_no_extrapolation_or_future_or_stale_measurement(self):
        for stamp in (-.01,1.01):
            with self.assertRaises(ValueError):self.history.at(stamp)
        for capture,now in [(0,.251),(.3,.2)]:
            with self.assertRaises(ValueError):detection_world(self.history,capture=capture,now=now,u=320,depth=1)

    def test_bad_depth_intrinsics_noise_and_transform_order(self):
        for key,value in [('depth',0),('depth',float('nan')),('fx',0),('u',True),('sigma_u',-1)]:
            args={'capture':0,'now':0,'u':320,'depth':1};args[key]=value
            with self.assertRaises(ValueError):detection_world(self.history,**args)
        with self.assertRaises(ValueError):PoseHistory([Pose(0,0,0,0),Pose(0,1,0,0)])


class TrackingTests(unittest.TestCase):
    def test_continuous_acceleration_covariance_prediction(self):
        track=CVTrack(0,1,0,p00=.25,p11=.25,q=.2);track.predict(1)
        self.assertEqual(track.x,1)
        self.assertAlmostEqual(track.p00,.5666666666666667)
        self.assertAlmostEqual(track.p01,.35)
        self.assertAlmostEqual(track.p11,.45)

    def test_known_update_and_joseph_covariance(self):
        track=CVTrack(0,0,0,p00=1,p01=.5,p11=1,q=0)
        self.assertTrue(track.update(2,1))
        self.assertEqual((track.x,track.v),(1,.5))
        self.assertEqual((track.p00,track.p01,track.p11),(.5,.25,.875))

    def test_gate_rejection_leaves_track_unchanged(self):
        track=CVTrack(0,0,0,p00=.1,p11=.1)
        before=copy.deepcopy(track.__dict__)
        self.assertFalse(track.update(100,.1))
        self.assertEqual(track.__dict__,before)

    def test_negative_dt_and_duplicate_measurement_rejected(self):
        track=CVTrack(0,0,0)
        with self.assertRaises(ValueError):track.predict(-1)
        track.update(0,.1)
        with self.assertRaises(ValueError):track.update(0,.1)
        track.predict(.1);self.assertTrue(track.update(0,.1))

    def test_covariance_remains_symmetric_positive_over_trace(self):
        track=CVTrack(0,1,0,q=.1)
        for k in range(1,101):
            track.predict(k*.1);track.update(k*.1+.01*math.sin(k),.04)
            self.assertGreaterEqual(track.p00,0);self.assertGreaterEqual(track.p11,0)
            self.assertGreaterEqual(track.p00*track.p11-track.p01**2,-1e-12)

    def test_crossing_prediction_preserves_identity_in_fixture(self):
        self.assertEqual(demo()['crossing_assignment'],[(0,1),(1,0)])

    def test_association_unique_and_outliers_can_be_unmatched(self):
        tracks=[CVTrack(0,0,0,p00=.01),CVTrack(.1,0,0,p00=.01)]
        matched=assign(tracks,[(0,.01),(100,.01)])
        self.assertEqual(len(matched),1)
        self.assertEqual(len({j for i,j in matched}),1)
        self.assertEqual(assign(tracks,[]),[])
        self.assertEqual(assign([],[(0,.1)]),[])

    def test_invalid_covariance_and_variance(self):
        for kwargs in [{'p00':-1},{'p00':1,'p01':2,'p11':1},{'q':-1}]:
            with self.assertRaises(ValueError):CVTrack(0,0,0,**kwargs)
        track=CVTrack(0,0,0)
        with self.assertRaises(ValueError):track.update(0,-1)
        with self.assertRaises(ValueError):assign([track]*7,[])

    def test_prediction_overflow_is_transactional(self):
        track=CVTrack(0,1e308,0)
        before=copy.deepcopy(track.__dict__)
        with self.assertRaises(ValueError):track.predict(10)
        self.assertEqual(track.__dict__,before)
        with self.assertRaises(ValueError):CVTrack(0,0,0,p00=1e308,p01=1e308,p11=1e307)

    def test_update_overflow_is_transactional(self):
        track=CVTrack(0,0,0,p00=1e308,p01=1e308,p11=1e308)
        before=copy.deepcopy(track.__dict__)
        with self.assertRaises(ValueError):track.update(0,0)
        self.assertEqual(track.__dict__,before)


if __name__=='__main__':unittest.main()

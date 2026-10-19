"""Planar calibrated detection, capture-time transforms and a 1D CV Kalman tracker.

Synthetic inputs only; no detector, SLAM, ROS, 3D grasp or robot execution.
"""
from dataclasses import dataclass
import itertools
import json
import math


def finite(value,name):
    if type(value) not in (int,float):raise ValueError(f'{name} must be finite numeric')
    try:valid=math.isfinite(value)
    except OverflowError:valid=False
    if not valid:raise ValueError(f'{name} must be finite numeric')
    return float(value)


def nonnegative(value,name):
    value=finite(value,name)
    if value<0:raise ValueError(f'{name} must be nonnegative')
    return value


@dataclass(frozen=True)
class Pose:
    stamp: float
    x: float
    y: float
    yaw: float

    def __post_init__(self):
        for key in ('stamp','x','y','yaw'):finite(getattr(self,key),key)


class PoseHistory:
    def __init__(self,poses):
        self.poses=tuple(poses)
        if len(self.poses)<2 or any(not isinstance(p,Pose) for p in self.poses):
            raise ValueError('at least two Pose samples required')
        if any(a.stamp>=b.stamp for a,b in zip(self.poses,self.poses[1:])):
            raise ValueError('transform stamps must strictly increase')

    def at(self,stamp):
        stamp=finite(stamp,'stamp')
        if not self.poses[0].stamp<=stamp<=self.poses[-1].stamp:
            raise ValueError('transform extrapolation refused')
        for a,b in zip(self.poses,self.poses[1:]):
            if a.stamp<=stamp<=b.stamp:
                alpha=(stamp-a.stamp)/(b.stamp-a.stamp)
                delta=math.atan2(math.sin(b.yaw-a.yaw),math.cos(b.yaw-a.yaw))
                if abs(abs(delta)-math.pi)<1e-12:
                    raise ValueError('half-turn interpolation is ambiguous')
                return Pose(stamp,a.x+alpha*(b.x-a.x),a.y+alpha*(b.y-a.y),a.yaw+alpha*delta)
        raise AssertionError('bracket missing')


def detection_world(history,*,capture,now,u,depth,fx=400.,cx=320.,max_age=.25,
                    sigma_u=1.,sigma_depth=.02,sigma_position=.01,sigma_yaw=.005):
    capture,now=finite(capture,'capture'),finite(now,'now')
    max_age=nonnegative(max_age,'max_age')
    if not 0<=now-capture<=max_age:raise ValueError('future or stale detection')
    u,depth,fx,cx=(finite(v,k) for v,k in [(u,'u'),(depth,'depth'),(fx,'fx'),(cx,'cx')])
    if depth<=0 or fx<=0:raise ValueError('positive depth and focal length required')
    noise=[nonnegative(v,k) for v,k in [(sigma_u,'sigma_u'),(sigma_depth,'sigma_depth'),
                                       (sigma_position,'sigma_position'),(sigma_yaw,'sigma_yaw')]]
    sigma_u,sigma_depth,sigma_position,sigma_yaw=noise
    pose=history.at(capture)
    r=(u-cx)/fx;c,s=math.cos(pose.yaw),math.sin(pose.yaw)
    # Optical z=forward, x=right -> planar body x=forward, y=left.
    x=pose.x+depth*(c+s*r);y=pose.y+depth*(s-c*r)
    jacobians=[((depth*s/fx,-depth*c/fx),sigma_u),((c+s*r,s-c*r),sigma_depth),
               ((1.,0.),sigma_position),((0.,1.),sigma_position),
               ((depth*(-s+c*r),depth*(c+s*r)),sigma_yaw)]
    cov=[[sum(j[i]*j[k]*std*std for j,std in jacobians) for k in range(2)] for i in range(2)]
    if not all(math.isfinite(v) for v in [x,y,*cov[0],*cov[1]]):
        raise ValueError('projection overflow')
    return {'capture':capture,'world_xy':[x,y],'covariance_xy':cov,'age':now-capture}


def covariance(p00,p01,p11):
    p00,p01,p11=finite(p00,'P00'),finite(p01,'P01'),finite(p11,'P11')
    scale=max(abs(p00),abs(p01),abs(p11),1e-300)
    determinant=(p00/scale)*(p11/scale)-(p01/scale)**2
    if p00<0 or p11<0 or determinant < -1e-12:
        raise ValueError('covariance must be positive semidefinite')
    return p00,p01,p11


class CVTrack:
    """One world-coordinate axis: [position, velocity], continuous white acceleration."""
    def __init__(self,position,velocity,stamp,p00=1.,p01=0.,p11=1.,q=.1):
        self.x,self.v,self.stamp=finite(position,'position'),finite(velocity,'velocity'),finite(stamp,'stamp')
        self.p00,self.p01,self.p11=covariance(p00,p01,p11)
        self.q=nonnegative(q,'q');self.last_measurement=None

    def predict(self,stamp):
        stamp=finite(stamp,'stamp');dt=stamp-self.stamp
        if dt<0:raise ValueError('out-of-sequence prediction')
        try:
            x=finite(self.x+dt*self.v,'predicted position')
            predicted=covariance(self.p00+2*dt*self.p01+dt*dt*self.p11+self.q*dt**3/3,
                                 self.p01+dt*self.p11+self.q*dt*dt/2,self.p11+self.q*dt)
        except OverflowError as error:
            raise ValueError('prediction overflow') from error
        self.x=x;self.p00,self.p01,self.p11=predicted
        self.stamp=stamp

    def innovation(self,z,variance):
        z=finite(z,'measurement');variance=nonnegative(variance,'variance')
        total=finite(self.p00+variance,'innovation variance')
        if total<=0:raise ValueError('positive innovation variance required')
        residual=finite(z-self.x,'innovation residual')
        scaled=residual/math.sqrt(total)
        return residual,total,scaled*scaled

    def update(self,z,variance,gate=9.):
        gate=nonnegative(gate,'gate')
        if self.last_measurement==self.stamp:raise ValueError('duplicate measurement timestamp')
        residual,total,d2=self.innovation(z,variance)
        if d2>gate:return False
        k0,k1=self.p00/total,self.p01/total
        a,b,c=self.p00,self.p01,self.p11
        # Joseph covariance update; H=[1,0], R=variance.
        posterior=covariance((1-k0)**2*a+k0*k0*variance,
                             (1-k0)*(b-k1*a)+k0*k1*variance,
                             c-2*k1*b+k1*k1*a+k1*k1*variance)
        x=finite(self.x+k0*residual,'updated position')
        v=finite(self.v+k1*residual,'updated velocity')
        # Commit only after all numerical checks; a failed update is not a sample.
        self.p00,self.p01,self.p11=posterior
        self.x,self.v=x,v;self.last_measurement=self.stamp
        return True


def assign(tracks,measurements,gate=9.):
    """Exhaustive tiny-frame assignment: maximum gated matches, then minimum sum d².

    This is an inspectable small fixture, not a production Hungarian implementation.
    """
    gate=nonnegative(gate,'gate')
    if len(tracks)>6 or len(measurements)>6:raise ValueError('fixture supports at most six tracks/detections')
    costs=[[track.innovation(z,r)[2] for z,r in measurements] for track in tracks]
    best=None
    for choice in itertools.product(range(-1,len(measurements)),repeat=len(tracks)):
        selected=[j for j in choice if j>=0]
        if len(set(selected))!=len(selected):continue
        if any(j>=0 and costs[i][j]>gate for i,j in enumerate(choice)):continue
        score=(-len(selected),sum(costs[i][j] for i,j in enumerate(choice) if j>=0),choice)
        if best is None or score<best[0]:best=(score,choice)
    return [(i,j) for i,j in enumerate(best[1]) if j>=0]


def demo():
    history=PoseHistory([Pose(0.,0.,0.,0.),Pose(1.,1.,0.,0.)])
    # At capture .2, the camera has moved .2m; object is at world (5,1).
    observation=detection_world(history,capture=.2,now=.4,u=320-400/4.8,depth=4.8)
    wrong=detection_world(history,capture=.4,now=.4,u=320-400/4.8,depth=4.8)
    track=CVTrack(0.,1.,0.,p00=.25,p11=.25,q=.2)
    track.predict(1.)
    prior={'x':track.x,'v':track.v,'P':[track.p00,track.p01,track.p11]}
    accepted=track.update(1.2,.04)
    posterior={'x':track.x,'v':track.v,'P':[track.p00,track.p01,track.p11]}
    track.predict(2.);rejected=not track.update(20.,.04)
    crossing=[CVTrack(0.,1.,0.,p00=.04,p11=.01,q=.01),CVTrack(2.,-1.,0.,p00=.04,p11=.01,q=.01)]
    for item in crossing:item.predict(1.2)
    detections=[(.8,.01),(1.2,.01)]
    return {'scope':'synthetic planar geometry and 1D Gaussian motion/association; not a detector or robot',
            'capture_time_projection':observation,'wrong_receive_time_projection':wrong,
            'wrong_time_error_m':wrong['world_xy'][0]-observation['world_xy'][0],
            'kalman_prior':prior,'measurement_accepted':accepted,'kalman_posterior':posterior,
            'outlier_rejected':rejected,'crossing_assignment':assign(crossing,detections),
            'limitations':['linearized independent pixel/depth/pose noise','no shared-error or cross-track covariance',
                           '1D constant-velocity target model','exact crossing/occlusion may be ambiguous']}


if __name__=='__main__':print(json.dumps(demo(),indent=2,allow_nan=False))

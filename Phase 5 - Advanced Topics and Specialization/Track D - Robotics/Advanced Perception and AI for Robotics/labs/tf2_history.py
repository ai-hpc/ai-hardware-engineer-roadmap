#!/usr/bin/env python3
"""Actual Jazzy TF2 buffer semantics with generated transforms; no DDS or hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import platform


def experiment():
    from tf2_ros import Buffer,ExtrapolationException,LookupException
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    buffer=Buffer()
    for stamp,x in [(10,0.),(11,1.)]:
        transform=TransformStamped()
        transform.header.frame_id='world';transform.child_frame_id='camera_body'
        transform.header.stamp.sec=stamp;transform.transform.translation.x=x
        transform.transform.rotation.w=1.
        buffer.set_transform(transform,'course_fixture')
    at_capture=buffer.lookup_transform('world','camera_body',Time(seconds=10.2))
    latest=buffer.lookup_transform('world','camera_body',Time())
    inverse=buffer.lookup_transform('camera_body','world',Time(seconds=10.2))
    assert abs(at_capture.transform.translation.x-.2)<1e-12
    assert latest.transform.translation.x==1.
    assert abs(inverse.transform.translation.x+.2)<1e-12
    rejected=[]
    for label,stamp in [('past',9.),('future',12.)]:
        try:buffer.lookup_transform('world','camera_body',Time(seconds=stamp))
        except ExtrapolationException:rejected.append(label)
        else:raise AssertionError(f'{label} extrapolation unexpectedly accepted')
    try:buffer.lookup_transform('world','missing_camera',Time())
    except LookupException:rejected.append('missing_frame')
    else:raise AssertionError('missing frame unexpectedly accepted')
    return {'scope':'actual TF2 buffer library; transforms inserted directly; no middleware transport',
            'python':platform.python_version(),'ros_distro':'jazzy','checks_passed':6,
            'capture_time_translation_x':at_capture.transform.translation.x,
            'latest_translation_x':latest.transform.translation.x,
            'inverse_translation_x':inverse.transform.translation.x,'rejected':rejected,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitations':['no timestamped sensor driver or clock synchronization',
                           'no tf topic publication, network delay or robot motion']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();text=json.dumps(experiment(),indent=2)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')

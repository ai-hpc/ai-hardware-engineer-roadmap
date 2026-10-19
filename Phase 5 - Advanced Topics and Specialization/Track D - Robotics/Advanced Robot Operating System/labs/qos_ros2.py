#!/usr/bin/env python3
"""Actual localhost ROS 2 QoS experiment with string fixtures; no robot endpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import uuid


def experiment(domain):
    if type(domain) is not int or not 0 <= domain <= 101:
        raise ValueError('choose a local test domain in 0..101')
    for key in ('ROS_DISCOVERY_SERVER','ROS_STATIC_PEERS','ROS_LOCALHOST_ONLY',
                'FASTRTPS_DEFAULT_PROFILES_FILE','FASTDDS_DEFAULT_PROFILES_FILE'):
        os.environ.pop(key,None)
    os.environ.update(ROS_DOMAIN_ID=str(domain),ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
                      RMW_IMPLEMENTATION='rmw_fastrtps_cpp',SKIP_DEFAULT_XML='1')
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.event_handler import SubscriptionEventCallbacks
    from rclpy.qos import (QoSProfile,ReliabilityPolicy as R,DurabilityPolicy as D,
                           qos_check_compatible,QoSCompatibility,LivelinessPolicy as L)
    from std_msgs.msg import String

    namespace='/course_qos_'+uuid.uuid4().hex[:10]
    rclpy.init(args=[])
    producer=Node('producer',namespace=namespace,enable_rosout=False)
    consumer=Node('consumer',namespace=namespace,enable_rosout=False)
    executor=SingleThreadedExecutor();executor.add_node(producer);executor.add_node(consumer)
    handles=[];received={};events={};rows=[]

    def spin_for(seconds):
        end=time.monotonic()+seconds
        while time.monotonic()<end:executor.spin_once(timeout_sec=.01)

    def wait_for(predicate,description):
        end=time.monotonic()+5
        while not predicate():
            if time.monotonic()>end:raise TimeoutError(description)
            executor.spin_once(timeout_sec=.01)

    try:
        for offered in (R.BEST_EFFORT,R.RELIABLE):
            for requested in (R.BEST_EFFORT,R.RELIABLE):
                name=f'{offered.name.lower()}_to_{requested.name.lower()}'
                received[name]=[];events[name]=0
                pub_qos=QoSProfile(depth=10,reliability=offered,durability=D.VOLATILE,liveliness=L.AUTOMATIC)
                sub_qos=QoSProfile(depth=10,reliability=requested,durability=D.VOLATILE,liveliness=L.AUTOMATIC)
                compatible,reason=qos_check_compatible(pub_qos,sub_qos)
                def incompatible(event,key=name):events[key]=event.total_count
                pub=producer.create_publisher(String,name,pub_qos)
                sub=consumer.create_subscription(String,name,lambda msg,key=name:received[key].append(msg.data),
                        sub_qos,event_callbacks=SubscriptionEventCallbacks(incompatible_qos=incompatible))
                handles.append((pub,sub,name,compatible,reason))
        wait_for(lambda:all((events[name]>0 if compatible==QoSCompatibility.ERROR else pub.get_subscription_count()>0)
                           for pub,sub,name,compatible,reason in handles),'QoS discovery/event evidence')
        for sequence in range(10):
            for pub,sub,name,compatible,reason in handles:pub.publish(String(data=str(sequence)))
            spin_for(.02)
        spin_for(.2)
        for pub,sub,name,compatible,reason in handles:
            should_deliver=compatible!=QoSCompatibility.ERROR
            assert bool(received[name])==should_deliver,(name,received[name],reason)
            if not should_deliver:assert events[name]>=1
            rows.append({'case':name,'compatible':should_deliver,'received':len(received[name]),
                         'incompatible_events':events[name],'preflight_reason':reason})

        # A living transient-local writer retains history for a late durable reader.
        latched_qos=QoSProfile(depth=1,reliability=R.RELIABLE,durability=D.TRANSIENT_LOCAL,liveliness=L.AUTOMATIC)
        latch=producer.create_publisher(String,'configuration',latched_qos)
        latch.publish(String(data='before-subscription'))
        spin_for(.1)
        late={'volatile':[],'transient_local':[]};late_handles=[]
        for label,durability in [('volatile',D.VOLATILE),('transient_local',D.TRANSIENT_LOCAL)]:
            qos=QoSProfile(depth=1,reliability=R.RELIABLE,durability=durability,liveliness=L.AUTOMATIC)
            late_handles.append(consumer.create_subscription(String,'configuration',
                                lambda msg,key=label:late[key].append(msg.data),qos))
        wait_for(lambda:latch.get_subscription_count()==2 and bool(late['transient_local']),
                 'late durable subscriber history')
        spin_for(.2)
        assert late=={'volatile':[],'transient_local':['before-subscription']},late
        before={key:list(value) for key,value in late.items()}
        latch.publish(String(data='after-subscription'))
        wait_for(lambda:all('after-subscription' in values for values in late.values()),'live delivery to both readers')
        rows.extend({'case':'late_'+key,'historical':before[key],'after_new_publish':values}
                    for key,values in late.items())
        return {'scope':'actual ROS 2 Jazzy/Fast DDS localhost transport; two nodes in one process',
                'python':platform.python_version(),'domain_id':domain,'namespace':namespace,
                'checks_passed':6,'results':rows,
                'limitations':['no sensors, TF, robot or actuator','observed deliveries are not reliability/latency benchmarks',
                               'durable writer remains alive; no persistence after writer shutdown claimed'],
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    finally:
        executor.shutdown();producer.destroy_node();consumer.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain-id',type=int,default=79)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    text=json.dumps(experiment(args.domain_id),indent=2,allow_nan=False)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')

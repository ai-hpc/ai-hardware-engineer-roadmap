"""ROS 2 Jazzy CPU lab: real PointStamped transport, TF, and freshness checks.

Source /opt/ros/jazzy/setup.bash before running. This is separate from the portable
unit suite: it requires rclpy, geometry_msgs, tf2_ros and tf2_geometry_msgs.
Publishes only namespaced observation and TF topics; no actuator interfaces.
"""

import argparse
import json
import math
import os
from pathlib import Path
import platform
import secrets
import subprocess
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    domain = args.domain_id if args.domain_id is not None else 20 + secrets.randbelow(60)
    if not 0 <= domain <= 101:
        parser.error("use a local test domain in 0..101")

    # These settings affect this process only. Do not inherit remote discovery or
    # vendor XML profiles from a user's unrelated robotics workspace.
    for name in ("ROS_DISCOVERY_SERVER", "ROS_STATIC_PEERS", "ROS_LOCALHOST_ONLY",
                 "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE"):
        os.environ.pop(name, None)
    os.environ.update(ROS_DOMAIN_ID=str(domain),
                      ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
                      RMW_IMPLEMENTATION="rmw_fastrtps_cpp", SKIP_DEFAULT_XML="1")

    import rclpy
    from geometry_msgs.msg import PointStamped, TransformStamped
    from rclpy.duration import Duration
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.utilities import get_rmw_implementation_identifier
    from tf2_geometry_msgs import do_transform_point
    from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener

    token = "robotics_lesson_" + uuid.uuid4().hex[:10]
    namespace = "/" + token
    world, sensor = token + "_world", token + "_sensor"
    remaps = [("/tf", namespace + "/tf"), ("/tf_static", namespace + "/tf_static")]
    max_age = .5
    rows = []
    rclpy.init(args=[])
    # TF's conventional topics are absolute. Explicit remapping is needed in
    # addition to the node namespace to keep this lab's frame traffic separate.
    ros_remaps = [argument for old, new in remaps for argument in ("-r", old + ":=" + new)]
    producer = Node("producer", namespace=namespace, cli_args=["--ros-args", *ros_remaps],
                    use_global_arguments=False, enable_rosout=False)
    consumer = Node("consumer", namespace=namespace, cli_args=["--ros-args", *ros_remaps],
                    use_global_arguments=False, enable_rosout=False)
    executor = SingleThreadedExecutor()
    executor.add_node(producer)
    executor.add_node(consumer)
    buffer = Buffer(node=consumer)
    listener = TransformListener(buffer, consumer)
    broadcaster = StaticTransformBroadcaster(producer)
    publishers, subscriptions = {}, []

    def receive(label, message):
        now = consumer.get_clock().now()
        age = (now - Time.from_msg(message.header.stamp)).nanoseconds / 1e9
        row = {"case": label, "age_s": age, "input_frame": message.header.frame_id}
        if age < 0:
            row["outcome"] = "reject_future"
        elif age > max_age:
            row["outcome"] = "reject_stale"
        elif message.header.frame_id != sensor:
            row["outcome"] = "reject_frame"
        else:
            try:
                transform = buffer.lookup_transform(world, sensor, Time.from_msg(message.header.stamp))
                result = do_transform_point(message, transform)
                row.update(outcome="accepted", output_frame=result.header.frame_id,
                           point_world_m=[result.point.x, result.point.y, result.point.z])
            except TransformException as error:
                row.update(outcome="reject_missing_transform", reason=str(error))
        rows.append(row)

    cases = [("fresh", 0., sensor), ("stale", -2., sensor),
             ("future", 2., sensor), ("wrong_frame", 0., token + "_unknown")]
    for label, _, _ in cases:
        topic = "observations/" + label
        publishers[label] = producer.create_publisher(PointStamped, topic, 10)
        subscriptions.append(consumer.create_subscription(
            PointStamped, topic, lambda message, case=label: receive(case, message), 10))

    def spin_until(predicate, timeout=5.):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS discovery/message delivery exceeded the lab timeout")
            executor.spin_once(timeout_sec=.05)

    try:
        mount = TransformStamped()
        mount.header.stamp = producer.get_clock().now().to_msg()
        mount.header.frame_id = world
        mount.child_frame_id = sensor
        mount.transform.translation.x = 1.
        mount.transform.translation.y = 2.2
        mount.transform.rotation.z = math.sin(math.pi / 4)
        mount.transform.rotation.w = math.cos(math.pi / 4)
        broadcaster.sendTransform(mount)
        spin_until(lambda: all(pub.get_subscription_count() == 1 for pub in publishers.values())
                   and buffer.can_transform(world, sensor, Time()))
        for label, offset, frame in cases:
            message = PointStamped()
            message.header.stamp = (producer.get_clock().now() + Duration(seconds=offset)).to_msg()
            message.header.frame_id = frame
            message.point.x = 1.
            publishers[label].publish(message)
            spin_until(lambda: any(row["case"] == label for row in rows))

        expected = {"fresh": "accepted", "stale": "reject_stale",
                    "future": "reject_future", "wrong_frame": "reject_frame"}
        for row in rows:
            if row["outcome"] != expected[row["case"]]:
                raise AssertionError(f"unexpected result: {row}")
        point = rows[0]["point_world_m"]
        if any(abs(actual - target) > 1e-10 for actual, target in zip(point, [1., 3.2, 0.])):
            raise AssertionError(f"wrong transformed point: {point}")

        try:
            package_query = subprocess.run(
                ["dpkg-query", "-W", "ros-jazzy-rclpy", "ros-jazzy-tf2-ros",
                 "ros-jazzy-rmw-fastrtps-cpp", "ros-jazzy-geometry-msgs"],
                capture_output=True, text=True, check=False)
        except FileNotFoundError:
            package_query = None
        report = {"execution_status": "measured local ROS 2 transport and TF; synthetic input",
                  "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "python": platform.python_version(), "ros_distro": os.environ.get("ROS_DISTRO"),
                  "rmw": get_rmw_implementation_identifier(), "domain_id": domain,
                  "discovery_range": "LOCALHOST", "namespace": namespace,
                  "max_age_s": max_age, "clock": "same-host system clock",
                  "package_versions": package_query.stdout.splitlines() if package_query else [],
                  "checks_passed": len(rows), "results": rows,
                  "limitations": ["two nodes in one process on one host",
                                  "static transform; no moving frame or sim clock",
                                  "four samples are a correctness check, not a latency benchmark",
                                  "no robot, actuator, simulator, or physical safety validation"]}
        encoded = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    finally:
        listener.unregister()
        executor.shutdown()
        producer.destroy_node()
        consumer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real localhost ROS 2 transport with the atomic capstone protocol; no robot APIs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import time
import uuid
from capstone_protocol import CommandGate, Rejected, canonical, command_for, strict_loads


def isolate_ros(domain):
    if not 0 <= domain <= 101:
        raise ValueError("domain must be in 0..101 for this local test")
    for key in ("ROS_DISCOVERY_SERVER", "ROS_STATIC_PEERS", "ROS_LOCALHOST_ONLY",
                "FASTRTPS_DEFAULT_PROFILES_FILE", "FASTDDS_DEFAULT_PROFILES_FILE"):
        os.environ.pop(key, None)
    os.environ.update(ROS_DOMAIN_ID=str(domain), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
                      RMW_IMPLEMENTATION="rmw_fastrtps_cpp", SKIP_DEFAULT_XML="1")


def experiment(domain):
    isolate_ros(domain)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import String

    namespace = "/course_atomic_" + uuid.uuid4().hex[:10]
    rclpy.init(args=[])
    simulator = Node("simulation_stub", namespace=namespace, enable_rosout=False)
    policy = Node("scripted_policy_stub", namespace=namespace, enable_rosout=False)
    executor = SingleThreadedExecutor()
    executor.add_node(simulator); executor.add_node(policy)
    observations = simulator.create_publisher(String, "observation", 1)
    commands = policy.create_publisher(String, "normalized_action", 1)
    gate = CommandGate([-1]*7, [1]*7)
    gate.reset(0, [0]*7)
    state = {"case": "fresh", "last_command": None}
    rows = []

    def policy_callback(message):
        obs = strict_loads(message.data)
        if state["case"] == "mixed_sensor":
            obs["cube"][0] += .1
        command = command_for(obs, [.5]*7)
        if state["case"] == "invalid_numeric":
            command["action"][0] = True
        state["last_command"] = command
        commands.publish(String(data=canonical(command)))

    def command_callback(message):
        now = time.monotonic_ns()
        command = strict_loads(message.data)
        try:
            result = gate.accept(command, now)
            rows.append({"case": state["case"], "accepted": True, "age_ns": result.age_ns,
                         "target": list(result.target)})
        except Rejected as error:
            rows.append({"case": state["case"], "accepted": False, "reason": str(error),
                         "target": list(gate.target)})

    subscriptions = [policy.create_subscription(String, "observation", policy_callback, 1),
                     simulator.create_subscription(String, "normalized_action", command_callback, 1)]

    def spin_until(condition):
        deadline = time.monotonic() + 5
        while not condition():
            if time.monotonic() > deadline:
                raise TimeoutError("local ROS discovery or delivery timeout")
            executor.spin_once(timeout_sec=.02)

    try:
        spin_until(lambda: observations.get_subscription_count() == 1 and commands.get_subscription_count() == 1)
        for case in ("fresh", "duplicate", "expired", "previous_episode", "mixed_sensor", "invalid_numeric"):
            state["case"] = case
            before = len(rows)
            if case == "duplicate":
                commands.publish(String(data=canonical(state["last_command"])))
            else:
                stamp = time.monotonic_ns() - (2_000_000_000 if case == "expired" else 0)
                obs = gate.issue([0]*9, [0]*9, [.5, 0, .025], [.6, 0, .025], stamp, 500_000_000)
                if case == "previous_episode":
                    gate.reset(1, [.2]*7)
                observations.publish(String(data=canonical(obs)))
            spin_until(lambda: len(rows) > before)
        assert [row["accepted"] for row in rows] == [True, False, False, False, False, False]
        assert rows[0]["target"] == [.025]*7
        assert all(row["target"] == [.2]*7 for row in rows[3:])
        return {"scope": "actual ROS 2 localhost transport; scripted action and simulated state fixtures",
                "python": platform.python_version(), "ros_distro": os.environ.get("ROS_DISTRO"),
                "domain_id": domain, "namespace": namespace, "clock": "controller monotonic clock echoed by policy",
                "ttl_ms": 500, "results": rows, "checks_passed": len(rows),
                "limitations": ["two nodes in one process", "no Isaac, model inference, actuator or robot",
                                "six cases demonstrate correctness, not latency percentiles or deadlines"],
                "protocol_sha256": hashlib.sha256(Path(__file__).with_name("capstone_protocol.py").read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    finally:
        executor.shutdown()
        simulator.destroy_node(); policy.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=20+secrets.randbelow(60))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(experiment(args.domain_id), indent=2, allow_nan=False) + "\n"
    if args.output: args.output.write_text(encoded)
    print(encoded, end="")

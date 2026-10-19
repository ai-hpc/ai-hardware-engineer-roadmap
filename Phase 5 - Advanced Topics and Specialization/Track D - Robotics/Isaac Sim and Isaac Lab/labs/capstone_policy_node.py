#!/usr/bin/env python3
"""Optional CPU policy adapter for the simulator-only atomic bridge.

Requires a trusted exported TorchScript policy, CPU PyTorch and ROS 2 Jazzy.
Transport/protocol is exercised by ros2_capstone_replay.py; trained inference is not.
"""
import argparse
import json
from pathlib import Path
import time
from capstone_protocol import (JOINT_NAMES, Rejected, canonical, command_for,
                               policy_observation, strict_loads, vector)
from ros2_capstone_replay import isolate_ros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--domain-id", type=int, required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    isolate_ros(args.domain_id)
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from std_msgs.msg import String
    import torch

    contract = json.loads(args.contract.read_text())
    if tuple(contract["joint_names"]) != JOINT_NAMES:
        raise ValueError("checkpoint joint order differs from the course contract")
    q0 = vector(contract["q0"], 9, "q0")
    model = torch.jit.load(str(args.policy), map_location="cpu").eval()
    rclpy.init(args=[])
    node = Node("push_policy", namespace=args.namespace, enable_rosout=False)
    publisher = node.create_publisher(String, "normalized_action", 1)
    last_key = (-1, -1)

    def receive(message):
        nonlocal last_key
        try:
            obs = strict_loads(message.data)
            values = policy_observation(obs, q0)
            key = (obs["episode"], obs["sequence"])
            if key <= last_key:
                raise Rejected("old or duplicate observation")
            start = time.perf_counter_ns()
            with torch.inference_mode():
                output = model(torch.tensor([values], dtype=torch.float32))
            inference_ns = time.perf_counter_ns()-start
            if output.shape != (1, 7) or not torch.isfinite(output).all():
                raise Rejected("policy must return seven finite actions")
            action = output[0].clamp(-1, 1).tolist()
            command = command_for(obs, action)
            # No target integration or last-action update here: only the simulator knows acceptance.
            publisher.publish(String(data=canonical(command)))
            last_key = key
            print(json.dumps({"episode": key[0], "sequence": key[1],
                              "inference_ns": inference_ns,
                              "inference_through_publish_ns": time.perf_counter_ns()-start}), flush=True)
        except (Rejected, KeyError, TypeError) as error:
            print(json.dumps({"rejected_observation": str(error)}), flush=True)

    subscription = node.create_subscription(String, "observation", receive, 1)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__": main()

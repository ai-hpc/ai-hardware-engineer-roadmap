#!/usr/bin/env python3
"""Exercise the actual policy adapter in another process using a tiny known-weight model.

Requires ROS 2 Jazzy and CPU PyTorch. It creates no pretrained/robot artifact and
does not install packages or launch Isaac Sim. All temporary files are removed.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from capstone_protocol import JOINT_NAMES, CommandGate, canonical, strict_loads
from ros2_capstone_replay import isolate_ros


def experiment(domain):
    isolate_ros(domain)
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    import torch

    namespace = "/course_policy_smoke_" + uuid.uuid4().hex[:10]
    with tempfile.TemporaryDirectory(prefix="course-policy-smoke-") as folder:
        work = Path(folder)
        policy_path, contract_path = work/"known-policy.pt", work/"contract.json"
        linear = torch.nn.Linear(31, 7)
        with torch.no_grad():
            linear.weight.zero_(); linear.bias.fill_(.25)
            linear.weight[0, 0] = 2.; linear.bias[0] = .1
        torch.jit.script(linear.eval()).save(str(policy_path))
        contract_path.write_text(json.dumps({"joint_names": JOINT_NAMES, "q0": [0]*9,
                                             "limits": [[-1,1]]*7, "action_scale": .05,
                                             "physics_dt": 1/120, "decimation": 2}))
        process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("capstone_policy_node.py")),
                                    "--policy", str(policy_path), "--contract", str(contract_path),
                                    "--domain-id", str(domain), "--namespace", namespace],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rclpy.init(args=[])
        node = Node("known_state_controller", namespace=namespace, enable_rosout=False)
        publisher = node.create_publisher(String, "observation", 1)
        replies = []
        subscription = node.create_subscription(String, "normalized_action",
                                                lambda msg: replies.append(strict_loads(msg.data)), 1)
        gate = CommandGate([-1]*7, [1]*7)
        gate.reset(0, [0]*7)

        def spin_until(condition):
            deadline = time.monotonic()+15
            while not condition():
                if process.poll() is not None:
                    raise RuntimeError("policy process exited during smoke test")
                if time.monotonic() > deadline:
                    raise TimeoutError("policy discovery/inference timed out")
                rclpy.spin_once(node, timeout_sec=.02)

        results = []
        expected_target = [0.0]*7
        try:
            spin_until(lambda: publisher.get_subscription_count() == 1 and subscription.get_publisher_count() == 1)
            for episode, q_first, expected_first in [(0,.2,.5), (0,.4,.9), (1,.1,.3)]:
                if episode != gate.episode:
                    gate.reset(episode, [0]*7)
                    expected_target = [0.0]*7
                q = [q_first]+[0]*8
                obs = gate.issue(q,[0]*9,[.5,0,.025],[.6,0,.025],time.monotonic_ns(),5_000_000_000)
                index = len(replies)
                publisher.publish(String(data=canonical(obs)))
                spin_until(lambda: len(replies) > index)
                command = replies[-1]
                expected = [expected_first]+[.25]*6
                assert max(abs(a-b) for a,b in zip(command["action"],expected)) < 1e-6
                result = gate.accept(command, time.monotonic_ns())
                expected_target = [max(-1., min(1., target+.05*action))
                                   for target,action in zip(expected_target,expected)]
                assert max(abs(a-b) for a,b in zip(result.target,expected_target)) < 1e-6
                results.append({"episode":episode,"sequence":obs["sequence"],"action":command["action"],
                                "target":list(result.target),"roundtrip_ns":result.age_ns})
        finally:
            node.destroy_node(); rclpy.shutdown()
            process.send_signal(signal.SIGINT)
            try:
                stdout,stderr=process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); stdout,stderr=process.communicate()
        assert process.returncode == 0, stderr
        return {"scope":"actual separate-process TorchScript policy adapter over localhost ROS; known-weight linear fixture",
                "torch":torch.__version__,"domain_id":domain,"namespace":namespace,"checks_passed":3,
                "results":results,"policy_log":stdout.splitlines(),"policy_stderr":stderr.splitlines(),
                "limitations":["no pretrained model or learned task quality","no Isaac Sim or robot",
                                "five-second diagnostic TTL; not a control deadline benchmark"]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id",type=int,default=78)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    encoded=json.dumps(experiment(args.domain_id),indent=2,allow_nan=False)+"\n"
    if args.output:args.output.write_text(encoded)
    print(encoded,end="")

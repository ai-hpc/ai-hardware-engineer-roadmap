"""Exercise two real OpenThread CLI processes over their simulated radio.

Requires the build in CPU-Simulation.md and pexpect 4.9.0. No board, root,
border router, or Internet route is used by the simulation itself.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import time

PINNED_COMMIT = "227561fe2446fa57e8a1aa192c81e31396db9ac9"
PUBLIC_TEST_KEY = "00112233445566778899aabbccddeeff"


def run(binary: Path, port_offset: int | None = None) -> dict:
    import pexpect

    binary = binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"Not an executable OpenThread CLI: {binary}")
    if port_offset is None:
        port_offset = 100 + secrets.randbelow(900)
    if not 1 <= port_offset <= 1000:
        raise ValueError("port offset must be in 1..1000 for this 33-node build")
    environment = dict(os.environ, PORT_OFFSET=str(port_offset), PORT_BASE="9000")
    report = {
        "passed": False,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "port_offset": port_offset,
        "transport": "OpenThread UDP radio simulation on Linux loopback",
        "checks": [],
        "log": [],
    }
    nodes = []

    def command(node, text, timeout=12):
        node.sendline(text)
        result = node.expect([r"\r\nDone\r\n", r"\r\nError \d+: [^\r\n]*\r\n"],
                             timeout=timeout)
        output = node.before.strip()
        status = node.after.strip()
        node.expect_exact("> ", timeout=timeout)
        report["log"].append({"node": nodes.index(node) + 1, "command": text,
                              "output": output, "status": status})
        if result:
            raise RuntimeError(report["log"][-1])
        return output

    def wait_role(node, accepted):
        deadline = time.monotonic() + 35
        while True:
            role = command(node, "state").splitlines()[-1]
            if role in accepted:
                return role
            if time.monotonic() > deadline:
                raise RuntimeError(f"Expected {accepted}, last role {role}")
            time.sleep(0.3)

    def ping(node, address, received, count=3):
        output = command(node, f"ping {address} 8 {count} 0.2 64 1", timeout=15)
        expected = f"{count} packets transmitted, {received} packets received."
        if expected not in output:
            raise RuntimeError(f"Expected {expected!r}; actual: {output}")

    def udp(sender, receiver, address, payload):
        command(sender, f"udp send {address} 12345 {payload}")
        # Wait on the receiving process, not on the sender's echoed command.
        receiver.expect(r"\d+ bytes from [0-9a-f:]+ \d+ " + re.escape(payload) + r"\r\n",
                        timeout=10)
        report["log"].append({"node": nodes.index(receiver) + 1,
                              "event": "UDP received", "payload": payload,
                              "output": receiver.before + receiver.after})

    # The documented build uses settings path ".". Each run therefore gets
    # new flash files; a previous network cannot make the attachment test pass.
    with tempfile.TemporaryDirectory(prefix="course-thread-") as work:
        try:
            for node_id in (1, 2):
                node = pexpect.spawn(str(binary), ["-L", "lo", str(node_id)],
                                     cwd=work, env=environment, encoding="utf-8",
                                     echo=False, timeout=12)
                nodes.append(node)
                node.expect_exact("> ")
            a, b = nodes
            report["version"] = command(a, "version").splitlines()[-1]
            command(a, "dataset init new")
            command(a, "dataset networkname CourseLab")
            command(a, "dataset channel 15")
            command(a, "dataset panid 0x1234")
            command(a, f"dataset networkkey {PUBLIC_TEST_KEY}")
            command(a, "dataset commit active")
            dataset = command(a, "dataset active -x").splitlines()[-1]
            bytes.fromhex(dataset)  # Reject malformed CLI extraction.
            command(a, "ifconfig up")
            command(a, "thread start")
            wait_role(a, {"leader"})
            command(b, "dataset set active " + dataset)
            command(b, "ifconfig up")
            command(b, "thread start")
            report["attached_role"] = wait_role(b, {"child", "router"})
            report["checks"].append("leader and second-node attachment")
            address = command(b, "ipaddr mleid").splitlines()[-1]
            ipaddress.IPv6Address(address)
            ping(a, address, 3)
            report["checks"].append("three ICMPv6 replies")
            command(a, "udp open")
            command(b, "udp open")
            command(b, "udp bind :: 12345")
            udp(a, b, address, "course-seq-1")
            report["checks"].append("UDP payload observed on receiver")

            command(b, "thread stop")
            command(b, "ifconfig down")
            wait_role(b, {"disabled"})
            ping(a, address, 0, count=1)
            report["checks"].append("offline node returns zero ICMPv6 replies")

            command(b, "ifconfig up")
            command(b, "thread start")
            wait_role(b, {"child", "router"})
            address = command(b, "ipaddr mleid").splitlines()[-1]
            ipaddress.IPv6Address(address)
            ping(a, address, 3)
            udp(a, b, address, "course-seq-2")
            report["checks"].append("reattachment restores new ping and UDP traffic")
            flash_files = sorted(Path(work).glob("*.flash"))
            if len(flash_files) != 2:
                raise RuntimeError("Expected two private flash files; rebuild with "
                                   "OT_POSIX_SETTINGS_PATH set to quoted dot")
            report["checks"].append("both nodes used isolated persistent settings")
            report["passed"] = True
        finally:
            for node in nodes:
                node.close(force=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--port-offset", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.cli, args.port_offset)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print("PASS: " + "; ".join(report["checks"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Explicit optional Docker build/run test. Only removes its own named containers."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid


HERE = Path(__file__).resolve().parent


def docker(*args, timeout=30):
    result = subprocess.run(["docker", *args], check=True, capture_output=True,
                            text=True, timeout=timeout)
    return result.stdout.strip()


def inspect(name):
    return json.loads(docker("inspect", name))[0]


def request(port, method, path, body=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request(method, path, json.dumps(body) if body is not None else None,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(image_tag):
    docker("version", "--format", "{{.Server.Version}}")
    docker("build", "--tag", image_tag, str(HERE), timeout=180)
    observations = []
    with tempfile.TemporaryDirectory(prefix="course-container-model-") as temp:
        bad_model = Path(temp) / "bad-model.json"
        bad_model.write_text('{"schema":1,"horizon_s":1,"margin_m":9}\n')
        for bad in (False, True):
            name = "course-lifecycle-" + uuid.uuid4().hex[:12]
            args = ["run", "--detach", "--name", name, "--init", "--read-only",
                    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--memory", "64m", "--memory-swap", "64m", "--cpus", "0.5",
                    "--pids-limit", "32", "--publish", "127.0.0.1::8080"]
            if bad:
                args += ["--mount", f"type=bind,source={bad_model},target=/app/model.json,readonly"]
            args.append(image_tag)
            try:
                docker(*args)
                deadline = time.monotonic() + 15
                wanted = "unhealthy" if bad else "healthy"
                while True:
                    state = inspect(name)
                    health = state["State"].get("Health", {}).get("Status")
                    if health == wanted:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"container did not become {wanted}: {state['State']}")
                    time.sleep(.1)
                port = int(state["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostPort"])
                live = request(port, "GET", "/live")
                ready = request(port, "GET", "/ready")
                output = request(port, "POST", "/infer",
                                 {"distance_m": 1, "closing_speed_mps": 1, "sensor_age_ms": 20})
                stale = request(port, "POST", "/infer",
                                {"distance_m": 1, "closing_speed_mps": 1, "sensor_age_ms": 101})
                require(live[0] == 200, "liveness request failed")
                require(ready[0] == (503 if bad else 200), "readiness did not reflect self-test")
                require(output[0] == (503 if bad else 200), "inference status mismatch")
                require(stale[0] == (503 if bad else 422), "stale-input status mismatch")
                if not bad:
                    require(output[1]["stop"] is True, "known-input inference failed")
                require(state["State"]["Running"], "health result must be independent of process running")
                require(state["Config"]["User"] == "10001:10001", "unexpected runtime user")
                require(state["HostConfig"]["ReadonlyRootfs"], "rootfs is not read-only")
                require(state["HostConfig"]["CapDrop"] == ["ALL"], "capabilities were not dropped")
                require(docker("exec", name, "id", "-u") == "10001", "effective UID differs")
                write_attempt = subprocess.run(["docker", "exec", name, "python", "-c",
                                                "open('/app/should-not-exist', 'w').write('x')"],
                                               capture_output=True, timeout=10)
                require(write_attempt.returncode != 0, "unexpected writable app directory")
                docker("stop", "--timeout", "5", name)
                stopped = inspect(name)["State"]
                require(stopped["ExitCode"] == 0 and not stopped["OOMKilled"], "unclean shutdown")
                observations.append({"model": "wrong parameters" if bad else "validated parameters",
                                     "docker_health": health, "running_before_stop": True,
                                     "live_status": live[0], "ready_status": ready[0],
                                     "inference_status": output[0], "stale_input_status": stale[0],
                                     "uid": 10001, "read_only_write_rejected": True,
                                     "clean_sigterm_exit": stopped["ExitCode"],
                                     "model_sha256": ready[1]["model_sha256"]})
            finally:
                subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=15)
    image = json.loads(docker("image", "inspect", image_tag))[0]
    return {"evidence_kind": "actual Docker CPU containers, no Jetson or GPU",
            "server_version": docker("version", "--format", "{{.Server.Version}}"),
            "image_id": image["Id"], "image_architecture": image["Architecture"],
            "base_reference": (HERE / "Dockerfile").read_text().splitlines()[0].removeprefix("FROM "),
            "observations": observations,
            "files_sha256": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                             for name in ("Dockerfile", "service.py", "model.json", "container_smoke.py")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # A unique local image name avoids overwriting an existing user's tag.
    tag = "course-edge-lifecycle:" + uuid.uuid4().hex[:12]
    result = run(tag)
    result["local_image_tag"] = tag
    result["checked_on"] = time.strftime("%Y-%m-%d", time.gmtime())
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)

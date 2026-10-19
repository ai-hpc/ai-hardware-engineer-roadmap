# Run, reject, and stop an edge service

**Time:** 45–60 minutes. **Requirements:** Python 3.10+ for the local lab; a working Docker installation for the optional container run. No GPU, Jetson, package installation, or network is needed for the Python tests. Building the container may download the pinned base image.

The service implements a **deterministic teaching stop rule**, not a learned model or a robot controller. It lets you practice artifact identity, input contracts, health checks, constrained containers, and shutdown with small code you can inspect.

## 1. Download the complete exercise

Place [service.py](service.py), [model.json](model.json), and [test_service.py](test_service.py) in one directory. For Docker, also get [Dockerfile](Dockerfile) and [container_smoke.py](container_smoke.py).

```bash
python3 -m unittest discover -s . -p 'test_*.py' -v
python3 service.py
```

The second command stays running on `127.0.0.1:8080`; stop it with Ctrl+C after the requests below. The ten tests exercise real local HTTP requests, invalid inputs, incorrect parameters, draining, and a separate process's SIGTERM shutdown.

The rule is:

```text
stop = distance_m <= max(closing_speed_mps, 0) * horizon_s + margin_m
```

The supplied parameters are a 1-second horizon and a 0.4-metre margin. A distance of 1 m and closing speed of 1 m/s produce `stop=true`, because `1 <= 1 × 1 + 0.4`. These constants and the rule are arbitrary teaching inputs; they are not a stopping-distance model or a safe control policy.

## 2. Observe three different contracts

In a second terminal:

```bash
python3 - <<'PY'
import json
import urllib.request
base = 'http://127.0.0.1:8080'
for path in ('/live', '/ready'):
    with urllib.request.urlopen(base + path, timeout=3) as response:
        print(path, response.status, response.read().decode())
body = json.dumps({'distance_m': 1, 'closing_speed_mps': 1, 'sensor_age_ms': 20}).encode()
request = urllib.request.Request(base + '/infer', data=body,
                                 headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=3) as response:
    print('/infer', response.status, response.read().decode())
PY
```

| Endpoint | What it establishes | What it does not establish |
| --- | --- | --- |
| `/live` | This HTTP loop answered | Correct parameters or useful robot behavior |
| `/ready` | Parameters parsed and the supplied known-input cases passed, and shutdown has not begun | Full model quality, live sensor freshness, or GPU availability |
| `/infer` | A validated request produced the deterministic rule's result | A physical action occurred or was safe |

The parameter SHA-256 identifies exactly which bytes were loaded. It is not a publisher signature. The supplied self-test checks several known answers, not every possible input or every parameter setting. A real learned-model service needs its own numerical, shape, quality, and runtime checks.

## 3. Send requests that must fail

The JSON body must have exactly three named numeric fields. Booleans, NaN/infinity, strings, missing fields, duplicate keys, and extra fields fail. The body is limited to 1,024 bytes; unknown content types and unsupported transfer encodings are rejected.

Set `sensor_age_ms` to 101. The service returns HTTP 422 because its teaching acceptance window is 0–100 ms. This value is **caller-declared**. The server has no camera or trusted acquisition clock and cannot establish that the caller told the truth. A real pipeline must carry a trusted timestamp and enforce age at both admission and completion, as developed in the [real-time inference chapter](../../Orin-Nano-Real-Time-Inference/Guide.md).

Copy `model.json` to a different filename and set the margin to 9. Run with `--model YOUR_COPY.json --port 8081`. The process still answers `/live`, but `/ready` and `/infer` return 503 because the known-input self-test fails. Keeping the liveness response distinct makes this failure observable.

This small single-threaded HTTP server is for a local exercise. It has no TLS, authentication, rate limiting, worker pool, reverse proxy, or production overload policy. The socket timeout limits inactivity, not total request duration: a client that keeps sending small pieces can occupy the one server for longer. This example has no absolute request deadline.

## 4. Run the real container checks

The Dockerfile pins the multi-platform Python base by its content digest, copies only the service and parameters, and uses UID/GID 10001. Its default command is in exec form so signal handling is explicit.

```bash
python3 container_smoke.py --output /tmp/course-container-result.json
```

The harness builds a uniquely named local image and runs two temporary containers. Host port publication is restricted to loopback. It applies a read-only root filesystem, drops capabilities, enables `no-new-privileges`, and sets memory/CPU/process limits. It creates no registry publication and removes only its own containers when finished. The resulting local image tag is recorded in the report; remove that specific tag when you no longer need it.

The [retained execution report](container-result.json) records an actual **amd64 Docker 29.3.0** run on 2026-10-18:

| Case | Docker health | `/live` | `/ready` | `/infer` | Process still running before stop |
| --- | --- | --- | --- | --- | --- |
| Supplied parameters | healthy | 200 | 200 | 200 | yes |
| Wrong parameters | unhealthy | 200 | 503 | 503 | yes |

Both used UID 10001, rejected a write to `/app`, and stopped with exit code 0 on SIGTERM. The valid service rejected the stale input with 422. The unhealthy container remained running: a health status alone did not restart or roll it back.

This run validates the CPU container on the recorded host. The base image supports multiple architectures, but this review did not execute its arm64 variant or exercise NVIDIA device access.

## 5. Explain shutdown and deployment acceptance

When SIGTERM arrives, the service marks itself draining and exits after the current handler returns. The process test checks shutdown while idle; another test checks that the draining state rejects new work. The inactivity timeout does not establish an overall shutdown deadline for a client that keeps sending. An application with a required shutdown bound must enforce an absolute handler deadline or explicitly interrupt the active connection. A real asynchronous GPU service needs additional steps: stop admission, account for in-flight buffers and operations, synchronize or cancel under an explicit policy, record outcomes, then release resources.

A deployment controller must decide what to do with an unhealthy process. Restarting the same bad model forever is not recovery. Apply bounded attempts, preserve the previous compatible deployment, and record the rejection reason. Continue with the [interrupted-update lab](../../Orin-Nano-OTA-Deep-Dive/labs/README.md).

**Completion:** run the ten local tests, explain the known-answer calculation, demonstrate one invalid request and one live-but-unready model, and inspect the saved container report. If you execute Docker, attach your own report and distinguish the host CPU result from a future Jetson GPU test. [Back to deployment guide](../Guide.md).

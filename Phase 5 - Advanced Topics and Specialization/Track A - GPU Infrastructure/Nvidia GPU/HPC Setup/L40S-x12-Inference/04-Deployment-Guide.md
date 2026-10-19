# 04 — Deploy one L40S service, then add a second replica

**Outcome:** make placement, readiness, streaming and measurement observable before scaling to twelve cards. Reviewed 2026-10-18; the examples below require allocated GPU/server infrastructure and were not deployed during this course update.

## Prepare a reproducible service identity

Record the GPU UUID/PCI address, driver, container digest or Python package lock, immutable model/tokenizer snapshot, runtime flags and data/license access. A rule such as “driver ≥535, CUDA ≥12.1, PyTorch ≥2.2” is not a compatibility matrix for a current runtime. Match the exact wheel/container's documented requirements. Use [the current runtime/model setup](../8x-H200-Training-Inference/03-Inference-Setup.md).

Check hardware using `nvidia-smi` queries and topology. L40S has no NVLink. Runtime-reported memory need not equal a hard-coded 48,102 MiB, and idle PCIe link speed may downshift. Compare supported maximum and active-load state before declaring a fault. Temperature/power limits come from the actual server. ECC configuration is a maintenance decision, not part of each service restart.

## Two explicit replicas

Prepare a compatible local model at `/models/course-snapshot`, and a pinned environment at `/opt/course-vllm`. Start one instance, validate it, then start the second on its own allocated GPU. These paths are example deployment destinations that must exist first.

```bash
CUDA_VISIBLE_DEVICES=0 /opt/course-vllm/bin/vllm serve /models/course-snapshot \
    --served-model-name course-model --dtype bfloat16 \
    --max-model-len 4096 --max-num-seqs 8 \
    --host 127.0.0.1 --port 8000
```

```bash
CUDA_VISIBLE_DEVICES=1 /opt/course-vllm/bin/vllm serve /models/course-snapshot \
    --served-model-name course-model --dtype bfloat16 \
    --max-model-len 4096 --max-num-seqs 8 \
    --host 127.0.0.1 --port 8001
```

Manage these with the site's supervisor rather than leaving orphan shell jobs. Under systemd, use an absolute executable from the pinned environment, an explicit working directory, model mount permissions, resource placement and restart limits. Under containers, pin the image digest and mount the prepared snapshot; do not use `latest` as a reproducibility identifier.

The single-GPU model must fit with cache/workspace. A 70B BF16 model does not fit on a two-L40S group; see [capacity and layout](03-Multi-GPU-Strategy.md).

## Streaming reverse proxy example

This is a complete local demonstration `nginx.conf` for the two replicas. It uses a local high port. Production endpoint authentication, TLS, exposure and request limits belong to the deployment's access policy; validate that configuration before exposing it.

```nginx
worker_processes 1;
events { worker_connections 1024; }
http {
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    upstream course_replicas {
        least_conn;
        server 127.0.0.1:8000;
        server 127.0.0.1:8001;
        keepalive 16;
    }
    server {
        listen 127.0.0.1:8080;
        location /v1/ {
            limit_req zone=api_limit burst=20 nodelay;
            proxy_pass http://course_replicas;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_read_timeout 60s;
            proxy_send_timeout 60s;
            proxy_next_upstream off;
        }
    }
}
```

Validate with the installed `nginx -t -c /absolute/path/nginx.conf` before use. Buffering can hide streaming progress. Disabling transparent upstream retry avoids silently replaying a partially completed generation; the application must define retry semantics. A read timeout measures gaps between reads, not necessarily total request lifetime. These example time limits are not a fleet SLO. [NGINX proxy directives](https://nginx.org/en/docs/http/ngx_http_proxy_module.html), [load balancing](https://nginx.org/en/docs/http/load_balancing.html).

## Health, readiness and startup are different

A process can be alive while still loading weights, compiling kernels or recovering. A ready replica must serve a small representative request correctly. Let startup have a bounded but realistic budget before applying normal liveness checks. A too-short liveness probe can restart a healthy large-model load forever. On failure, stop admitting requests, drain/cancel according to policy and remove the replica before restart.

For Kubernetes, use a startup probe plus readiness/liveness appropriate to the runtime, explicit GPU resource limits, model volume and supported placement rules. Twelve replicas are a desired count, not proof that twelve L40S devices are available. A generic `nvidia.com/gpu: 1` request needs suitable node selection if hardware SKU matters; a shared model volume must support the intended access mode. [Kubernetes probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/).

## Monitor the actual emitted metrics

Save a sample of `/metrics` from the pinned runtime. Metric names changed between engine releases; do not assume an old `gpu_cache_usage_perc` or token-counter name still exists. Current vLLM metrics document the available queue/cache/request histograms and their labels. [vLLM production metrics](https://docs.vllm.ai/en/latest/design/metrics/).

For a **classic Prometheus histogram** exposing `vllm:time_to_first_token_seconds_bucket`, an example per-instance p95 query is:

```promql
histogram_quantile(0.95,
  sum by (le, instance) (
    rate(vllm:time_to_first_token_seconds_bucket[5m])
  )
)
```

The bucket rate and `le` label are essential. Do not pass the base histogram name to the classic-histogram form or average per-replica p95 values to obtain a fleet p95. Low traffic yields unstable tail estimates; also report counts. For native histograms, use that histogram type's documented query form. [Prometheus histograms](https://prometheus.io/docs/practices/histograms/).

Use DCGM exporter for supported GPU counters and record its version. `MEM_COPY_UTIL` is an activity indicator, not a direct byte/s bandwidth measurement. Correlate utilization, memory, power, queue length, error rate and request goodput; low utilization can mean no demand rather than a faulty GPU. [DCGM exporter](https://github.com/NVIDIA/dcgm-exporter).

## Test the service boundary

Run a bounded workload, then intentionally remove **one test replica** through its supervisor. Verify routing stops admitting to it, the other replica remains useful, and in-flight failures are recorded rather than counted as successes. Restore it and verify readiness before admitting traffic. This tests a service deployment; it is not a GPU hardware failure experiment.

For twelve cards, repeat the validated placement pattern with a topology/capacity map and measured shared CPU/NIC limits. For multiple model families, use separate pools and model-aware routing: request/response schemas, tokenizer/template, quality and deadlines differ. A function with undefined `get_connection_count` and `forward` is an architecture sketch, not a production router.

**Exit evidence:** configuration and environment identity, one direct request per replica, streamed request through the proxy, actual metric names/query output, a bounded overload test, and a replica drain/restart report. Mark all deployment outcomes unexecuted until those artifacts exist.

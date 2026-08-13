# 04 — Production Deployment Guide for L40S x12

## 1. Pre-Deployment Checklist

### Hardware Validation

```bash
# 1. Verify all 12 GPUs are detected
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
# Expected: 12 lines, "NVIDIA L40S", "48102 MiB"

# 2. Check PCIe link speeds
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv
# Expected: gen=4, width=16 for all GPUs

# 3. Verify NVLink is NOT expected (L40S has none)
nvidia-smi nvlink --status -i 0
# Expected: "NVLink not supported" — this is correct for L40S

# 4. Check PCIe topology (identify GPU pairs on same switch)
nvidia-smi topo -m

# 5. Thermal baseline (before load)
nvidia-smi --query-gpu=index,temperature.gpu,power.draw --format=csv
# GPU temp should be 30-45°C idle

# 6. ECC status
nvidia-smi --query-gpu=index,ecc.mode.current --format=csv
# Enable ECC for production: sudo nvidia-smi -e 1 (requires reboot)
```

### Software Stack

```bash
# Required versions for production L40S deployment
nvidia-smi   # Driver ≥ 535.x
nvcc -V      # CUDA ≥ 12.1
python -c "import torch; print(torch.__version__)"  # PyTorch ≥ 2.2
python -c "import vllm; print(vllm.__version__)"    # vLLM ≥ 0.4.0

# Install flash-attn (Ada Lovelace compatible)
pip install flash-attn --no-build-isolation

# Verify flash-attn works
python -c "
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 1, 32, 128, device='cuda', dtype=torch.float16)
k = torch.randn(1, 1, 32, 128, device='cuda', dtype=torch.float16)
v = torch.randn(1, 1, 32, 128, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v)
print('FlashAttention OK')
"
```

## 2. Single-GPU Deployment (7B / 13B Models)

For models ≤ 48 GB, deploy one model per L40S. With 12 GPUs, this gives 12 independent replicas.

### systemd Service for Each GPU

```ini
# /etc/systemd/system/vllm-gpu0.service
[Unit]
Description=vLLM Inference Server GPU 0
After=network.target

[Service]
Type=simple
User=mlops
Environment=CUDA_VISIBLE_DEVICES=0
Environment=TRANSFORMERS_CACHE=/models/cache
ExecStart=/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Deploy all 12 GPUs
for i in $(seq 0 11); do
    PORT=$((8000 + i))
    sed "s/GPU 0/GPU $i/; s/DEVICES=0/DEVICES=$i/; s/port 8000/port $PORT/" \
        vllm-gpu0.service > /etc/systemd/system/vllm-gpu${i}.service
done

systemctl daemon-reload
systemctl enable --now vllm-gpu{0..11}
```

### Docker Compose Alternative

```yaml
# docker-compose.yml
version: "3.8"
services:
  vllm-0:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    ports: ["8000:8000"]
    volumes:
      - /models:/models
    command: >
      --model meta-llama/Llama-3-8b-instruct
      --dtype bfloat16
      --max-model-len 8192
      --port 8000
    environment:
      - HF_TOKEN=${HF_TOKEN}
    restart: always

  vllm-1:
    # ... same, device_id: ['1'], port: 8001

  # ... repeat for GPU 2-11
```

## 3. Load Balancer Configuration (NGINX)

```nginx
# /etc/nginx/conf.d/vllm-cluster.conf
upstream vllm_8b {
    least_conn;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    server 127.0.0.1:8004;
    server 127.0.0.1:8005;
    server 127.0.0.1:8006;
    server 127.0.0.1:8007;
    server 127.0.0.1:8008;
    server 127.0.0.1:8009;
    server 127.0.0.1:8010;
    server 127.0.0.1:8011;
    keepalive 64;
}

server {
    listen 80;
    server_name inference.yourdomain.com;

    location /v1/ {
        proxy_pass http://vllm_8b;
        proxy_http_version 1.1;
        proxy_set_header Connection "";  # keepalive
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;

        # Rate limiting per client
        limit_req zone=api_limit burst=100 nodelay;
    }

    location /health {
        proxy_pass http://vllm_8b;
    }
}

# Rate limiting zone
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
```

## 4. Monitoring Stack

### Prometheus + Grafana Setup

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000', 'localhost:8001', ...]  # all 12 instances
    metrics_path: /metrics

  - job_name: 'nvidia_gpu'
    static_configs:
      - targets: ['localhost:9400']  # dcgm-exporter
```

```bash
# Run DCGM exporter for GPU metrics
docker run -d --gpus all \
    --rm -p 9400:9400 \
    nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04 \
    -f /etc/dcgm-exporter/dcp-metrics-included.csv
```

### Key Metrics to Monitor

```python
# vLLM exposes these Prometheus metrics at /metrics
VLLM_METRICS = {
    "vllm:num_requests_running":    "Active requests",
    "vllm:num_requests_waiting":    "Queued requests",
    "vllm:gpu_cache_usage_perc":    "KV cache usage %",
    "vllm:time_to_first_token_seconds":  "TTFT histogram",
    "vllm:time_per_output_token_seconds": "TPOT histogram",
    "vllm:request_success_total":   "Successful requests",
    "vllm:request_prompt_tokens_total": "Input tokens served",
    "vllm:request_generation_tokens_total": "Output tokens served",
}

# GPU metrics (via DCGM)
GPU_METRICS = {
    "DCGM_FI_DEV_GPU_UTIL":      "GPU utilization %",
    "DCGM_FI_DEV_MEM_COPY_UTIL": "Memory bandwidth utilization %",
    "DCGM_FI_DEV_FB_USED":       "GPU memory used (MB)",
    "DCGM_FI_DEV_POWER_USAGE":   "Power draw (W)",
    "DCGM_FI_DEV_GPU_TEMP":      "Temperature (°C)",
}
```

### Alert Rules

```yaml
# alert_rules.yml
groups:
  - name: l40s_inference
    rules:
      - alert: HighTTFT
        expr: histogram_quantile(0.95, vllm:time_to_first_token_seconds) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P95 TTFT > 500ms on {{ $labels.instance }}"

      - alert: GPUHighTemp
        expr: DCGM_FI_DEV_GPU_TEMP > 82
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.gpu }} approaching thermal limit"

      - alert: HighQueueDepth
        expr: vllm:num_requests_waiting > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Request queue backing up on {{ $labels.instance }}"

      - alert: LowGPUUtilization
        expr: DCGM_FI_DEV_GPU_UTIL < 30
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "GPU underutilized — consider reducing replicas"
```

## 5. Multi-Model Deployment (Mixed Workloads)

Use 12 GPUs to serve multiple models simultaneously:

```
GPU 0-5   → 6 × Llama-3-8B   (BF16, 1 GPU each) — high-volume chat
GPU 6-7   → 1 × Llama-3-70B  (BF16, TP=2)       — complex reasoning
GPU 8-9   → 1 × Llama-3-70B  (BF16, TP=2)       — complex reasoning
GPU 10    → 1 × CodeLlama-34B (INT8, 1 GPU)      — code generation
GPU 11    → 1 × embedding model                  — RAG embeddings
```

```python
# Intelligent router based on task type
ROUTING_CONFIG = {
    "chat":         {"endpoints": [f"http://localhost:{8000+i}" for i in range(6)]},
    "reasoning":    {"endpoints": ["http://localhost:8006", "http://localhost:8007"]},
    "code":         {"endpoints": ["http://localhost:8010"]},
    "embeddings":   {"endpoints": ["http://localhost:8011"]},
}

async def route_request(task_type: str, request: dict):
    endpoints = ROUTING_CONFIG[task_type]["endpoints"]
    # least-connections load balancing
    endpoint = min(endpoints, key=lambda ep: get_connection_count(ep))
    return await forward(endpoint, request)
```

## 6. Kubernetes Deployment

```yaml
# GPU-per-pod strategy: 12 pods, 1 GPU each
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b
spec:
  replicas: 12
  selector:
    matchLabels:
      app: vllm-8b
  template:
    metadata:
      labels:
        app: vllm-8b
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          resources:
            limits:
              nvidia.com/gpu: "1"    # 1 L40S per pod
              memory: "64Gi"
              cpu: "8"
          args:
            - "--model=meta-llama/Llama-3-8b-instruct"
            - "--dtype=bfloat16"
            - "--max-model-len=8192"
            - "--gpu-memory-utilization=0.90"
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 5
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: vllm-8b-service
spec:
  selector:
    app: vllm-8b
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

## 7. Health Checking and Auto-Recovery

```python
# health_monitor.py — auto-restart unhealthy instances
import asyncio, aiohttp, subprocess, logging

INSTANCES = [{"gpu": i, "port": 8000+i} for i in range(12)]
HEALTH_INTERVAL = 30  # seconds
RESTART_COOLDOWN = 60  # seconds

async def check_health(session: aiohttp.ClientSession, port: int) -> bool:
    try:
        async with session.get(f"http://localhost:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False

async def restart_instance(gpu_id: int, port: int):
    logging.warning(f"Restarting vllm on GPU {gpu_id} port {port}")
    subprocess.run(["systemctl", "restart", f"vllm-gpu{gpu_id}"])
    await asyncio.sleep(RESTART_COOLDOWN)

async def monitor():
    restart_times = {}
    async with aiohttp.ClientSession() as session:
        while True:
            for inst in INSTANCES:
                healthy = await check_health(session, inst["port"])
                if not healthy:
                    last_restart = restart_times.get(inst["gpu"], 0)
                    if asyncio.get_event_loop().time() - last_restart > RESTART_COOLDOWN:
                        await restart_instance(inst["gpu"], inst["port"])
                        restart_times[inst["gpu"]] = asyncio.get_event_loop().time()
            await asyncio.sleep(HEALTH_INTERVAL)

asyncio.run(monitor())
```

## References

- [vLLM Production Deployment](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)
- [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)
- [Kubernetes GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [NGINX Load Balancing](https://nginx.org/en/docs/http/load_balancing.html)
- [Prometheus Monitoring](https://prometheus.io/docs/)

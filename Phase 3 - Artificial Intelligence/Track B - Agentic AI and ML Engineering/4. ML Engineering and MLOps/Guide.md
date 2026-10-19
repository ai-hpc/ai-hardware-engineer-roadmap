# Module 4B — ML Engineering & MLOps

<div class="course-identity auto-course" style="--course-accent: #be123c; --course-accent-rgb: 190, 18, 60;" markdown="1">
<div class="course-identity__icon">M4ME</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · AI Workloads</p>
<p class="course-identity__title">Specialized course identity for Module 4B — ML Engineering & MLOps.</p>
<p class="course-identity__meta">Artifact: model or workload study · Measure: accuracy, latency, memory, throughput</p>
</div>
</div>


**Parent:** [Phase 3 — Artificial Intelligence](../../Guide.md) · Track B

> *Build and operate ML training/inference pipelines — model lifecycle, data pipelines, serving infrastructure.*

**Prerequisites:** Module 2 (Frameworks — PyTorch fluency) and one trained model. Agentic AI is a companion when the workload uses tools; it is not a prerequisite for MLOps.

**Role targets:** Machine Learning Engineer · MLOps Engineer · AI/ML Engineer · ML Platform Engineer

---

> **★ Featured deep-dive course:** [**Practical Machine Learning (CS329P)**](Practical%20Machine%20Learning%20%28CS329P%29/README.md) — the full ML lifecycle a practitioner actually meets, adapted from Stanford CS329P (Mu Li & Alex Smola) and refreshed for 2026: data acquisition/labeling, validation that doesn't lie, distribution-shift detection, tuning, and a hardware-facing [model-compression lecture](Practical%20Machine%20Learning%20%28CS329P%29/Lecture-10.md). 11 lectures. Start with the [Overview](Practical%20Machine%20Learning%20%28CS329P%29/README.md).

---

## Why This Matters for AI Hardware

ML engineers define the **training and serving workloads** that hardware must support:
- Distributed training (data/model/pipeline parallelism) → L3: NCCL, multi-GPU runtime
- Model serving (latency SLAs, throughput targets) → L1a: TensorRT, Triton, vLLM
- Experiment tracking and model versioning → infrastructure that uses GPU clusters (Phase 5A)
- Data pipelines → I/O bandwidth requirements that drive GPUDirect Storage (Phase 5B)

---

## First complete artifact on a CPU

Complete the [PyTorch robot-stop experiment](../../2.%20Deep%20Learning%20Frameworks/PyTorch/CPU-Robot-Stop-Lab/Guide.md) before distributed training. It produces a model, export, data/split hashes, errors under sensor shift, and an exact environment. Add those artifacts to an experiment tracker, then reconstruct one prediction from the saved feature order and normalization. No GPU cluster is needed to learn the lifecycle.

**Diagnostic:** does saving weights alone let you resume AdamW training exactly? **Answer:** no; also preserve optimizer/scheduler state, random-generator states, epoch/batch or sampler position, preprocessing, and data revisions. Inference restoration and exact training continuation are different contracts. Seeds alone do not guarantee equality across devices or library versions.

## 1. Training Pipelines

* **Data pipelines:** PyTorch DataLoader, NVIDIA DALI, streaming datasets
* **Distributed training:** DDP (DistributedDataParallel), FSDP (Fully Sharded), DeepSpeed
* **Mixed precision:** `torch.amp` with supported device/dtype choices; BF16 and separately validated FP8 training recipes
* **Checkpointing:** model saving, resume from checkpoint, elastic training
* **Hyperparameter tuning:** Optuna, Ray Tune, grid/random/Bayesian search

**Projects:**
1. Train a model with DDP across 2+ GPUs. Measure scaling efficiency.
2. Add mixed precision training. Compare FP32 vs BF16 training speed and accuracy.

---

## 2. Experiment Tracking & Model Registry

* **Experiment tracking:** MLflow, Weights & Biases (W&B), Neptune
* **Model registry:** immutable versions, tags, aliases, release gates, and rollback
* **Dataset versioning:** DVC, LakeFS
* **Reproducibility:** environment tracking, seed management, deterministic training

**Projects:**
1. Set up MLflow tracking for a training run. Log hyperparameters, metrics, and artifacts.
2. Register a model in MLflow. Validate a candidate, assign a `champion` alias, and rehearse restoring the prior version. MLflow model stages are deprecated; use the current [aliases/tags workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/) (checked 2026-10-18), while retaining a concrete version ID in each release manifest.

---

## 3. Model Serving & Inference Infrastructure

* **Serving frameworks:** Triton Inference Server, vLLM, TensorRT-LLM. Treat [TorchServe](https://github.com/pytorch/serve) as a historical case study: its repository declares limited maintenance
* **Batching:** static batching, dynamic batching, continuous batching (for LLMs)
* **Scaling:** horizontal (replicas), vertical (larger GPU), model parallelism for large models
* **Monitoring:** latency percentiles (p50/p99), throughput (QPS), error rates, GPU utilization
* **Policy gateways:** request filtering, moderation sidecars, auth, rate limits, safety traces
* **A/B testing:** canary deployments, shadow mode, traffic splitting

**Projects:**
1. Deploy a model on Triton with dynamic batching. Load test and measure p50/p99 latency.
2. Deploy an LLM on vLLM. Compare throughput with different batch sizes and quantization levels.
3. Insert a moderation or policy gateway in front of a serving stack and measure the latency tradeoff.

---

## 4. MLOps & CI/CD for Models

* **CI/CD:** GitHub Actions / GitLab CI for model training, testing, deployment
* **Container orchestration:** Docker, Kubernetes, NVIDIA GPU Operator
* **Model testing:** unit tests for preprocessing, integration tests for inference, data drift detection
* **Safety regression testing:** jailbreak suites, blocked-topic tests, prompt-attack replay, policy drift checks
* **Feature stores:** Feast, Tecton — manage reusable feature definitions; still test point-in-time joins, freshness, and training/serving parity
* **Orchestration:** Kubeflow Pipelines, Apache Airflow, Prefect

**Projects:**
1. Build a CI/CD pipeline: on push → train → evaluate quality, latency, and compatibility → stage a candidate → test rollback → promote through an explicit release gate. A better aggregate accuracy score alone is insufficient.
2. Set up GPU-enabled Kubernetes with NVIDIA GPU Operator. Deploy a model serving workload.

---

## Resources

| Resource | What it covers |
|----------|---------------|
| [MLflow Documentation](https://mlflow.org/docs/latest/) | Experiment tracking, model registry |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | Production model serving |
| [vLLM](https://github.com/vllm-project/vllm) | LLM serving engine |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Distributed training |
| *Designing Machine Learning Systems* (Huyen) | ML systems design |

---

## Next

→ [**Module 5B — LLM Application Development**](../5.%20LLM%20Application%20Development/Guide.md)

**Study method:** use the [shared lesson cycle and completion rubric](../../../Learning-Guide.md) with this course's existing labs. Predict first, test one variation, retain a failure case, and revisit the self-check after two days and one week.


## Carry this course into the physical-AI system

**Start from:** A trained model with dataset and preprocessing revisions.

**Build and challenge:** Build a release candidate with quality, timing, memory, and compatibility gates. Rehearse rollback and prove that restoring weights also restores normalization and runtime configuration.

**Pass forward:** A reproducible release manifest and rollback result for the robot policy. Continue with [the connected course](../../../Phase%205%20-%20Advanced%20Topics%20and%20Specialization/Track%20D%20-%20Robotics/VLA%20Optimization%20and%20Action-Parity%20Harness/README.md) and keep the artifact in the [Physical AI Design Studio](../../../Physical-AI-Design-Studio.md).

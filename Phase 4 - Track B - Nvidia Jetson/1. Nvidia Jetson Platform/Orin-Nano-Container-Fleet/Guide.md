# Containerized Deployment, Fleet Management, and DevOps for Jetson Orin Nano

<div class="course-identity auto-course" style="--course-accent: #475569; --course-accent-rgb: 71, 85, 105;" markdown="1">
<div class="course-identity__icon">CDFM</div>
<div markdown="1">
<p class="course-identity__eyebrow">Deep Dive · Jetson Track</p>
<p class="course-identity__title">Specialized course identity for Containerized Deployment, Fleet Management, and DevOps for Jetson Orin Nano.</p>
<p class="course-identity__meta">Artifact: Jetson integration demo · Measure: latency, memory, power, logs</p>
</div>
</div>


A deep-dive guide for the Jetson Orin Nano 8GB (T234 SoC) covering containers,
orchestration, fleet management, CI/CD, monitoring, and security at the edge.

---

## 1. Introduction

### 1.1 Why Containers Matter for Edge AI

Edge AI devices like the Jetson Orin Nano are deployed in environments far removed
from the controlled conditions of a data center. They sit in factories, on drones,
inside vehicles, and at retail kiosks. Managing software on hundreds or thousands
of these devices without containers quickly becomes untenable.

Containers solve three fundamental problems at the edge:

1. **Reproducibility** -- A container image that works on a developer's Jetson works
   identically on every Jetson in the fleet. There is no "works on my device" problem.
2. **Isolation** -- Multiple workloads (inference, preprocessing, telemetry) run
   side-by-side without dependency conflicts.
3. **Atomic updates** -- A new container image is pulled and started, or it is not.
   There is no half-updated system state.

### 1.2 Containers vs. Bare-Metal on Jetson

| Concern | Bare-Metal | Containerized |
|---|---|---|
| Dependency management | System-wide apt packages | Per-container, isolated |
| Update mechanism | apt upgrade + pray | Pull new image, rollback if needed |
| Multi-application | Conflicts between Python/CUDA versions | Each service has its own stack |
| Reproducibility | Snowflake devices over time | Identical images across fleet |
| Rollback | Difficult, often requires re-flash | Stop old container, start previous tag |
| Resource control | Manual cgroup configuration | Built-in Docker/K8s resource limits |
| Security surface | All processes share the host | Namespace isolation, read-only rootfs |
| Overhead | None | ~1-3% CPU, negligible memory |

The overhead on the Orin Nano is minimal. Container syscall overhead is measured in
nanoseconds. The shared 8GB LPDDR5 memory is the real constraint, and containers
do not duplicate GPU memory -- they share the unified memory space just as bare-metal
processes do.

### 1.3 Orin Nano Hardware Context

The Jetson Orin Nano 8GB (T234 SoC) provides:

- 6 Arm Cortex-A78AE CPU cores
- 1024-core Ampere GPU (SM 8.7)
- 8GB unified LPDDR5 memory (shared between CPU and GPU)
- 1x NVDLA v2.0 engine
- Up to 40 TOPS INT8 AI performance (at 15W)

The 8GB shared memory is the dominant constraint for container planning. Every
container, every CUDA context, and every model shares this single pool.

### 1.4 JetPack and L4T Version Alignment

Throughout this guide, we target JetPack 6.x (L4T R36.x) which ships with:

- CUDA 12.2+
- cuDNN 8.9+
- TensorRT 8.6+
- Linux kernel 5.15

Container base images must match the L4T version running on the host. This is a
hard requirement. A container built for L4T R35.x will not work correctly on an
R36.x host because the NVIDIA driver interface between the container userspace
libraries and the host kernel driver must be ABI-compatible.

Check your host version:

```bash
# L4T version
head -1 /etc/nv_tegra_release
# Example output: # R36 (release), REVISION: 3.0, ...

# JetPack version
apt list --installed 2>/dev/null | grep nvidia-jetpack
```

---

## 2. Container Runtime on Jetson

### 2.1 The NVIDIA Container Runtime

The NVIDIA Container Runtime is the bridge that gives containers access to the GPU.
On Jetson (Tegra) devices, it works differently from discrete GPU systems. There is
no separate GPU card with its own memory -- the GPU shares the SoC's unified memory.

The runtime is implemented as an OCI runtime wrapper around runc. When Docker (or
containerd) starts a container with `--runtime nvidia`, the NVIDIA runtime:

1. Intercepts the OCI spec before passing it to runc
2. Mounts the necessary CUDA libraries from the host into the container
3. Exposes /dev/nvhost-* and /dev/nvmap devices
4. Sets environment variables for CUDA device visibility

### 2.2 Installing Docker with NVIDIA Support

JetPack 6.x typically ships with Docker pre-installed. If it is not present:

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io

# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use nvidia runtime by default
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default

# Restart Docker
sudo systemctl restart docker

# Verify
sudo docker run --rm nvcr.io/nvidia/l4t-base:r36.3.0 nvidia-smi
```

### 2.3 Docker Daemon Configuration

The key configuration file is `/etc/docker/daemon.json`. A production-ready
configuration for the Orin Nano:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "storage-driver": "overlay2",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "default-shm-size": "256M",
    "storage-opts": [
        "overlay2.size=20G"
    ]
}
```

Key decisions:

- **default-runtime: nvidia** -- Every container gets GPU access by default. This
  avoids the constant `--runtime nvidia` flag and prevents "GPU not found" bugs in
  production.
- **log rotation** -- Critical on devices with limited storage. Without rotation,
  a chatty container will fill the disk.
- **shm-size** -- PyTorch DataLoader workers use shared memory. The default 64MB
  is insufficient; 256MB is a reasonable baseline.
- **overlay2.size** -- Prevents a single container from consuming all disk.

### 2.4 containerd Configuration

If you are using K3s or prefer containerd directly:

```bash
# /etc/containerd/config.toml

version = 2

[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  privileged_without_host_devices = false
  runtime_engine = ""
  runtime_root = ""
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
```

Restart containerd after editing:

```bash
sudo systemctl restart containerd
```

### 2.5 cgroup Configuration

The Orin Nano kernel supports both cgroup v1 and v2. For Kubernetes (K3s), cgroup
v2 (unified hierarchy) is preferred:

```bash
# Check current cgroup version
stat -fc %T /sys/fs/cgroup/
# "cgroup2fs" = v2, "tmpfs" = v1

# To switch to cgroup v2, add to /boot/extlinux/extlinux.conf APPEND line:
#   systemd.unified_cgroup_hierarchy=1 cgroup_no_v1=all
```

After editing `/boot/extlinux/extlinux.conf`:

```
LABEL primary
    MENU LABEL primary kernel
    LINUX /boot/Image
    INITRD /boot/initrd
    APPEND ${cbootargs} root=PARTUUID=xxxx rootwait rootfstype=ext4 systemd.unified_cgroup_hierarchy=1
```

Reboot for the change to take effect.

---

## 3. NVIDIA L4T Base Containers

### 3.1 L4T Image Hierarchy

NVIDIA provides a layered set of base images hosted on `nvcr.io`. Each layer adds
libraries on top of the previous one:

```
+------------------------------------------+
|            l4t-ml                         |
|   (PyTorch, TensorFlow, cuDNN, TRT)      |
+------------------------------------------+
|            l4t-tensorrt                   |
|   (TensorRT + cuDNN + CUDA)              |
+------------------------------------------+
|            l4t-cuda                       |
|   (CUDA toolkit + runtime)               |
+------------------------------------------+
|            l4t-base                       |
|   (Ubuntu + NVIDIA driver userspace)     |
+------------------------------------------+
|            Ubuntu ARM64                   |
+------------------------------------------+
```

### 3.2 Image Details and Sizes

| Image | Contents | Compressed Size | Use Case |
|---|---|---|---|
| `l4t-base:r36.3.0` | Ubuntu 22.04 + driver libs | ~400 MB | Minimal GPU workloads |
| `l4t-cuda:12.2.2-runtime` | + CUDA runtime | ~800 MB | Custom CUDA applications |
| `l4t-cuda:12.2.2-devel` | + CUDA toolkit (nvcc) | ~2.5 GB | Building CUDA code in container |
| `l4t-tensorrt:r36.3.0` | + TensorRT + cuDNN | ~1.5 GB | TensorRT inference |
| `l4t-ml:r36.3.0` | + PyTorch + TF + scikit | ~5 GB | Full ML development |

### 3.3 Choosing the Right Base Image

Decision tree for selecting a base:

```
Need to compile CUDA code inside the container?
  YES --> l4t-cuda:*-devel
  NO  --> Do you need TensorRT?
            YES --> Do you also need PyTorch/TensorFlow?
                      YES --> l4t-ml (or build custom)
                      NO  --> l4t-tensorrt
            NO  --> Do you need CUDA runtime only?
                      YES --> l4t-cuda:*-runtime
                      NO  --> l4t-base
```

For production inference on the Orin Nano, `l4t-tensorrt` is the most common
starting point. The `l4t-ml` image is useful for development but too large for
constrained deployments.

### 3.4 Pulling and Verifying Images

```bash
# Pull the TensorRT base
sudo docker pull nvcr.io/nvidia/l4t-tensorrt:r36.3.0

# Verify GPU access
sudo docker run --rm nvcr.io/nvidia/l4t-tensorrt:r36.3.0 \
    python3 -c "import tensorrt; print(tensorrt.__version__)"

# Check CUDA
sudo docker run --rm nvcr.io/nvidia/l4t-base:r36.3.0 \
    nvcc --version 2>/dev/null || echo "nvcc not in runtime image"

sudo docker run --rm nvcr.io/nvidia/l4t-base:r36.3.0 \
    python3 -c "
import ctypes
libcudart = ctypes.CDLL('libcudart.so')
print('CUDA runtime loaded successfully')
"
```

### 3.5 Version Pinning

Always pin your base image to a specific L4T release. Never use `latest`:

```dockerfile
# GOOD: pinned to a specific L4T release
FROM nvcr.io/nvidia/l4t-tensorrt:r36.3.0

# BAD: will break when NVIDIA updates the tag
FROM nvcr.io/nvidia/l4t-tensorrt:latest
```

Verify the host/container L4T alignment:

```bash
# On host
cat /etc/nv_tegra_release

# In container -- should report compatible version
docker run --rm nvcr.io/nvidia/l4t-base:r36.3.0 cat /etc/nv_tegra_release
```

---

## 4. jetson-containers Project

### 4.1 Overview

The `dusty-nv/jetson-containers` repository is a community and NVIDIA-maintained
collection of Dockerfiles and build scripts for the Jetson platform. It provides
pre-built containers for dozens of AI/ML frameworks, all properly compiled for
aarch64 with JetPack GPU support.

Repository: https://github.com/dusty-nv/jetson-containers

### 4.2 Installation and Setup

```bash
# Clone the repository
git clone https://github.com/dusty-nv/jetson-containers.git
cd jetson-containers

# Install dependencies
pip3 install -r requirements.txt

# List available containers
./run.sh --list

# See details for a specific package
./run.sh --show pytorch
```

### 4.3 Available Pre-Built Containers

Key packages available for JetPack 6.x / Orin Nano:

| Package | Description | Typical Size |
|---|---|---|
| `pytorch` | PyTorch with CUDA support | ~6 GB |
| `tensorflow` | TensorFlow with CUDA | ~6 GB |
| `onnxruntime` | ONNX Runtime GPU | ~2 GB |
| `tritonserver` | Triton Inference Server | ~4 GB |
| `deepstream` | DeepStream SDK | ~5 GB |
| `ros:humble` | ROS 2 Humble | ~3 GB |
| `ollama` | Local LLM inference | ~3 GB |
| `text-generation-webui` | LLM web interface | ~8 GB |
| `stable-diffusion-webui` | Image generation | ~8 GB |
| `nanoowl` | Real-time OWL-ViT | ~4 GB |
| `nanodb` | Multimodal vector DB | ~4 GB |

### 4.4 Running Pre-Built Containers

```bash
# Run PyTorch container interactively
cd jetson-containers
./run.sh pytorch

# Run with a specific model directory mounted
./run.sh --volume /home/user/models:/models pytorch

# Run a specific version
./run.sh pytorch:2.1

# Combine multiple packages (layers are merged)
./run.sh pytorch tensorrt
```

### 4.5 Building Custom Containers from jetson-containers

The build system supports composing packages:

```bash
# Build a custom image combining PyTorch + TensorRT + ONNX Runtime
./build.sh --name my-inference pytorch tensorrt onnxruntime

# Build with a custom base
./build.sh --base nvcr.io/nvidia/l4t-tensorrt:r36.3.0 onnxruntime

# Skip packages already in your base
./build.sh --skip-packages cuda cudnn tensorrt --name my-app onnxruntime
```

### 4.6 Extending jetson-containers with Your Own Package

Create a directory under `packages/` with a `Dockerfile` and `config.py`:

```bash
mkdir -p packages/my-inference-app
```

`packages/my-inference-app/config.py`:

```python
from jetson_containers import L4T_VERSION

package = {
    'name': 'my-inference-app',
    'depends': ['pytorch', 'tensorrt'],
    'config': [
        {
            'name': 'my-inference-app',
            'requires': '>=36.0',  # L4T version requirement
        }
    ]
}
```

`packages/my-inference-app/Dockerfile`:

```dockerfile
# The build system injects the correct base automatically
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY app/ /opt/app/
RUN pip3 install -r /opt/app/requirements.txt

WORKDIR /opt/app
CMD ["python3", "inference_server.py"]
```

Build it:

```bash
./build.sh my-inference-app
```

---

## 5. Building Custom Containers

### 5.1 Multi-Stage Builds for Jetson

Multi-stage builds are essential on Jetson to keep final images small. The Orin Nano
typically has 32-64GB of NVMe storage; a bloated image directly reduces the number
of models and data you can store.

```dockerfile
# ============================================================
# Stage 1: Build environment
# ============================================================
FROM nvcr.io/nvidia/l4t-cuda:12.2.2-devel AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY src/ ./src/
COPY CMakeLists.txt .

RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCUDA_ARCHITECTURES=87 \
          .. && \
    make -j$(nproc)

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM nvcr.io/nvidia/l4t-tensorrt:r36.3.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgproc4.5d \
    libopencv-imgcodecs4.5d \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/build/my_inference_engine /usr/local/bin/
COPY models/ /opt/models/
COPY config/ /opt/config/

WORKDIR /opt
CMD ["my_inference_engine", "--config", "/opt/config/production.yaml"]
```

Key points:
- `CUDA_ARCHITECTURES=87` targets the Orin's SM 8.7 specifically, producing smaller
  and faster CUDA code than multi-architecture builds.
- The final stage uses `l4t-tensorrt` (runtime only), not the devel image.
- Build tools (cmake, gcc) are only in the builder stage and discarded.

### 5.2 Cross-Compilation: Building on x86 for arm64

Building directly on the Orin Nano is slow. A complex container can take hours to
build on the 6-core A78AE. Cross-compilation on an x86 workstation is 5-10x faster.

**Method 1: Docker Buildx with QEMU (simplest, slowest cross-compile)**

```bash
# On your x86 workstation
# Install QEMU user-static for arm64 emulation
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

# Create a buildx builder
docker buildx create --name jetson-builder --use
docker buildx inspect --bootstrap

# Build for arm64
docker buildx build \
    --platform linux/arm64 \
    --tag my-registry/my-inference:v1.0 \
    --push \
    .
```

QEMU emulation is 5-20x slower than native arm64, but still faster than building
on the Orin Nano due to the x86 host's superior single-thread performance and
available RAM.

**Method 2: Native cross-compilation (fastest)**

```bash
# Install aarch64 cross-compiler on x86
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Cross-compile outside Docker
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-toolchain.cmake \
      -DCUDA_ARCHITECTURES=87 \
      ..
make -j$(nproc)
```

The toolchain file (`cmake/aarch64-toolchain.cmake`):

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-cross/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

Then package the cross-compiled binary into a minimal runtime container.

### 5.3 Buildx Multi-Platform Workflow

For teams that deploy to both Jetson (arm64) and x86 servers:

```bash
# Build for both platforms simultaneously
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag my-registry/my-inference:v1.0 \
    --push \
    .
```

The Dockerfile should handle architecture-specific steps:

```dockerfile
FROM nvcr.io/nvidia/l4t-tensorrt:r36.3.0 AS base-arm64
FROM nvcr.io/nvidia/tensorrt:23.10-py3 AS base-amd64

ARG TARGETARCH
FROM base-${TARGETARCH} AS base

# Common steps below
COPY app/ /opt/app/
RUN pip3 install -r /opt/app/requirements.txt
```

### 5.4 Build Caching Strategies

On the Orin Nano's limited storage, Docker build cache can consume significant space.
Use targeted caching:

```bash
# Export cache to a specific location
docker buildx build \
    --cache-from type=local,src=/mnt/nvme/docker-cache \
    --cache-to type=local,dest=/mnt/nvme/docker-cache,mode=max \
    -t my-app:latest .

# Periodic cache cleanup
docker builder prune --keep-storage 5G
docker system prune -f
```

### 5.5 Minimizing Image Size

Practical techniques for keeping images small:

```dockerfile
FROM nvcr.io/nvidia/l4t-base:r36.3.0

# Combine RUN commands to reduce layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        libgomp1 \
    && pip3 install --no-cache-dir \
        flask==3.0.0 \
        numpy==1.26.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# Use .dockerignore to exclude unnecessary files
# Copy only what is needed
COPY --chown=1000:1000 app/ /opt/app/
COPY --chown=1000:1000 models/optimized/ /opt/models/
```

Create a `.dockerignore`:

```
.git
*.onnx
*.pth
__pycache__
*.pyc
.env
docker-compose*.yml
README.md
docs/
tests/
```

---

## 6. GPU Access in Containers

### 6.1 How NVIDIA Runtime Works on Tegra

On discrete GPU systems, the NVIDIA Container Toolkit uses the CDI (Container Device
Interface) or legacy mode to inject GPU devices. On Tegra (Jetson), the mechanism is
different because the GPU is integrated into the SoC.

The NVIDIA runtime on Jetson:

1. Mounts host CUDA libraries into the container at `/usr/lib/aarch64-linux-gnu/tegra/`
2. Exposes device nodes: `/dev/nvhost-*`, `/dev/nvmap`, `/dev/nvgpu/`
3. Binds the unified memory driver interface
4. Sets `LD_LIBRARY_PATH` to include the Tegra library path

### 6.2 Running Containers with GPU Access

```bash
# With default runtime set to nvidia (recommended)
docker run --rm my-cuda-app

# Explicit runtime specification
docker run --rm --runtime nvidia my-cuda-app

# For full device access (sometimes needed for multimedia)
docker run --rm --runtime nvidia --privileged my-cuda-app

# Targeted device access (preferred over --privileged)
docker run --rm --runtime nvidia \
    --device /dev/nvhost-ctrl \
    --device /dev/nvhost-ctrl-gpu \
    --device /dev/nvhost-prof-gpu \
    --device /dev/nvhost-gpu \
    --device /dev/nvmap \
    my-cuda-app
```

### 6.3 Device Nodes Reference

Devices typically needed for GPU compute on Orin Nano:

| Device | Purpose |
|---|---|
| `/dev/nvhost-ctrl` | NVIDIA host control |
| `/dev/nvhost-ctrl-gpu` | GPU control channel |
| `/dev/nvhost-gpu` | GPU execution channel |
| `/dev/nvhost-prof-gpu` | GPU profiling |
| `/dev/nvmap` | Unified memory mapping |
| `/dev/nvhost-as-gpu` | GPU address space |
| `/dev/nvhost-dbg-gpu` | GPU debugging |
| `/dev/nvhost-sched-gpu` | GPU scheduler |
| `/dev/nvhost-tsg-gpu` | GPU timeslice group |
| `/dev/nvhost-vic` | Video Image Compositor |
| `/dev/nvhost-nvdla0` | DLA engine 0 |
| `/dev/nvhost-nvdec` | Video decoder |
| `/dev/nvhost-nvenc` | Video encoder |

### 6.4 Verifying GPU Access Inside a Container

```bash
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.3.0 bash -c '
echo "=== Device nodes ==="
ls -la /dev/nv*

echo ""
echo "=== CUDA Libraries ==="
ldconfig -p | grep cuda

echo ""
echo "=== GPU Info ==="
python3 -c "
import subprocess
result = subprocess.run([\"cat\", \"/sys/devices/gpu.0/load\"], capture_output=True, text=True)
print(f\"GPU Load: {result.stdout.strip()}\")

result = subprocess.run([\"cat\", \"/sys/devices/gpu.0/railgate_enable\"], capture_output=True, text=True)
print(f\"Railgate: {result.stdout.strip()}\")
"

echo ""
echo "=== CUDA Quick Test ==="
python3 -c "
import ctypes
cuda = ctypes.CDLL(\"libcudart.so\")
device_count = ctypes.c_int()
cuda.cudaGetDeviceCount(ctypes.byref(device_count))
print(f\"CUDA Devices: {device_count.value}\")
"
'
```

### 6.5 CUDA Compute Capability

The Orin Nano GPU is SM 8.7 (compute capability 8.7). When building CUDA code
inside containers, target this specifically:

```bash
# In your Makefile or CMake
nvcc -arch=sm_87 my_kernel.cu -o my_kernel

# Or for portability across Jetson devices
nvcc -gencode arch=compute_72,code=sm_72 \  # Xavier NX
     -gencode arch=compute_87,code=sm_87 \  # Orin
     my_kernel.cu -o my_kernel
```

### 6.6 Environment Variables for GPU Control

```dockerfile
# In your Dockerfile or at runtime
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV OPENBLAS_CORETYPE=ARMV8
```

---

## 7. Camera Access in Containers

### 7.1 V4L2 Camera Access

USB cameras and some CSI cameras expose V4L2 device nodes. Passing these into
a container is straightforward:

```bash
# List available cameras on host
v4l2-ctl --list-devices

# Pass a USB camera into the container
docker run --rm --runtime nvidia \
    --device /dev/video0:/dev/video0 \
    my-camera-app

# Pass all video devices
docker run --rm --runtime nvidia \
    --device /dev/video0 \
    --device /dev/video1 \
    my-camera-app
```

Test camera access inside the container:

```bash
docker run --rm --runtime nvidia \
    --device /dev/video0 \
    nvcr.io/nvidia/l4t-base:r36.3.0 bash -c '
apt-get update && apt-get install -y v4l2-ctl
v4l2-ctl --device=/dev/video0 --all
v4l2-ctl --device=/dev/video0 --list-formats-ext
'
```

### 7.2 CSI Camera Access with Argus

CSI cameras (IMX219, IMX477, etc.) on Jetson use the NVIDIA Argus camera framework,
which communicates through `nvargus-daemon` running on the host. Container access
requires:

```bash
docker run --rm --runtime nvidia \
    --privileged \
    --volume /tmp/argus_socket:/tmp/argus_socket \
    --device /dev/video0 \
    my-csi-camera-app
```

For a non-privileged approach, map the specific devices:

```bash
docker run --rm --runtime nvidia \
    --volume /tmp/argus_socket:/tmp/argus_socket \
    --device /dev/video0 \
    --device /dev/nvhost-ctrl \
    --device /dev/nvhost-ctrl-gpu \
    --device /dev/nvhost-gpu \
    --device /dev/nvhost-vic \
    --device /dev/nvhost-isp \
    --device /dev/nvhost-vi \
    --device /dev/nvmap \
    my-csi-camera-app
```

Ensure `nvargus-daemon` is running on the host:

```bash
sudo systemctl start nvargus-daemon
sudo systemctl enable nvargus-daemon
```

### 7.3 GStreamer Pipeline in Containers

GStreamer with NVIDIA plugins is the standard way to capture from CSI cameras:

```python
# Python + GStreamer inside container
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

# CSI camera pipeline using nvarguscamerasrc
pipeline_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw,format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink"
)

pipeline = Gst.parse_launch(pipeline_str)
pipeline.set_state(Gst.State.PLAYING)
```

### 7.4 X11 Display Forwarding

For development and debugging, forward the display:

```bash
# Allow X11 connections
xhost +local:docker

# Run with display forwarding
docker run --rm --runtime nvidia \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/video0 \
    my-camera-app-with-gui
```

For Wayland (if using Weston compositor):

```bash
docker run --rm --runtime nvidia \
    --env WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
    --env XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    --volume $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
    my-wayland-app
```

### 7.5 Camera Dockerfile Example

```dockerfile
FROM nvcr.io/nvidia/l4t-tensorrt:r36.3.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    pycuda \
    flask

COPY app/ /opt/app/
WORKDIR /opt/app

CMD ["python3", "camera_inference.py"]
```

---

## 8. Container Networking

### 8.1 Network Modes Overview

Docker provides several networking modes. The choice impacts latency, isolation,
and service discovery.

| Mode | Isolation | Latency | Use Case |
|---|---|---|---|
| `bridge` (default) | Full | +50-100us | Multi-service, port mapping |
| `host` | None | Native | Low-latency inference endpoints |
| `macvlan` | Full | Native | Devices needing their own IP |
| `none` | Total | N/A | Offline processing |

### 8.2 Bridge Networking (Default)

Bridge mode creates an isolated network namespace. Services communicate via
published ports.

```bash
# Expose an inference API on port 8080
docker run -d --runtime nvidia \
    -p 8080:8080 \
    --name inference-server \
    my-inference:latest

# Multiple services on a custom bridge network
docker network create ai-network

docker run -d --runtime nvidia \
    --network ai-network \
    --name preprocessor \
    my-preprocessor:latest

docker run -d --runtime nvidia \
    --network ai-network \
    --name inference \
    -p 8080:8080 \
    my-inference:latest
```

Containers on the same bridge network can reach each other by container name:

```python
# Inside the inference container
import requests
response = requests.get("http://preprocessor:5000/preprocess", ...)
```

### 8.3 Host Networking for Low Latency

For inference endpoints where every microsecond matters:

```bash
docker run -d --runtime nvidia \
    --network host \
    --name inference-server \
    my-inference:latest
```

The container shares the host's network stack. No NAT, no port mapping overhead.
The container's services bind directly to the host's interfaces.

Trade-off: No network isolation. Port conflicts are possible if multiple containers
try to bind the same port.

### 8.4 Exposing Inference Endpoints

A typical pattern for a REST inference API:

```python
# inference_server.py
from flask import Flask, request, jsonify
import numpy as np
import tensorrt as trt
# ... TensorRT engine loading code ...

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "gpu": "available"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # ... run inference ...
    return jsonify({"predictions": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
```

### 8.5 mDNS for Device Discovery

In fleet scenarios, devices need to discover each other without a central registry.
mDNS (multicast DNS) via Avahi enables `.local` name resolution:

```bash
docker run -d --runtime nvidia \
    --network host \
    --name inference-server \
    -v /var/run/dbus:/var/run/dbus \
    -v /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
    my-inference:latest
```

Inside the container, register a service:

```bash
# Install avahi-utils in the container
apt-get install -y avahi-utils

# Register inference service
avahi-publish-service "JetsonInference" _http._tcp 8080 "model=yolov8n" &
```

Other devices on the network can then discover:

```bash
avahi-browse -rt _http._tcp
# Output: JetsonInference  _http._tcp  local  jetson-orin-nano-01.local:8080
```

### 8.6 Firewall Considerations

For production fleet deployments, lock down container networking:

```bash
# Allow only inference API port from the local network
sudo iptables -A DOCKER-USER -p tcp --dport 8080 -s 192.168.1.0/24 -j ACCEPT
sudo iptables -A DOCKER-USER -p tcp --dport 8080 -j DROP

# Persist rules
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save
```

---

## 9. Container Storage

### 9.1 Volume Mounts for Models and Data

AI models should never be baked into the container image. They change more frequently
than application code and can be very large.

```bash
# Mount model directory as read-only
docker run -d --runtime nvidia \
    -v /opt/models:/models:ro \
    -v /opt/config:/config:ro \
    -v /data/output:/output \
    my-inference:latest
```

Directory structure on the host:

```
/opt/
  models/
    yolov8n.engine          # TensorRT engine file
    resnet50.engine
    model_config.yaml
  config/
    inference.yaml
    camera.yaml
/data/
  output/                   # Inference results, logs
```

### 9.2 Overlay Filesystem Considerations

Docker's overlay2 storage driver uses layers. Each layer is read-only except the
top writable layer. On Jetson with limited NVMe:

```bash
# Check Docker disk usage
docker system df -v

# See layer sizes
docker history my-inference:latest

# Typical output:
# IMAGE          CREATED        SIZE      COMMENT
# abc123         2 hours ago    15MB      COPY app/
# def456         2 hours ago    245MB     pip install
# ghi789         3 hours ago    0B        WORKDIR
# jkl012         NVIDIA base    1.2GB     l4t-tensorrt base
```

Important: Writes inside a running container go to the overlay writable layer.
These writes are lost when the container is removed. They also consume disk space
that is not easily reclaimed until the container is deleted.

### 9.3 tmpfs for Scratch Data

For temporary processing data that does not need to persist:

```bash
# Mount a tmpfs for fast scratch space (uses RAM)
docker run -d --runtime nvidia \
    --tmpfs /tmp:rw,size=256m \
    --tmpfs /scratch:rw,size=512m \
    my-inference:latest
```

On the Orin Nano with 8GB shared memory, be conservative with tmpfs sizes.
Every megabyte of tmpfs is a megabyte less for GPU operations.

### 9.4 Persistent Storage Strategies

For edge devices that may lose power unexpectedly:

```bash
# Named volumes survive container recreation
docker volume create inference-data
docker run -d --runtime nvidia \
    -v inference-data:/data \
    my-inference:latest

# Bind mount to specific NVMe path
docker run -d --runtime nvidia \
    -v /mnt/nvme/inference-data:/data \
    my-inference:latest
```

For write-heavy workloads (logging, saving inference results), direct NVMe bind
mounts outperform Docker volumes because there is no overlay overhead.

### 9.5 Storage Cleanup Automation

Automated cleanup to prevent disk exhaustion:

```bash
# Cron job: /etc/cron.d/docker-cleanup
# Clean dangling images and stopped containers daily at 3 AM
0 3 * * * root docker system prune -f --filter "until=48h" >> /var/log/docker-cleanup.log 2>&1

# Clean unused images weekly
0 4 * * 0 root docker image prune -a --filter "until=168h" -f >> /var/log/docker-cleanup.log 2>&1
```

Monitor disk from within containers:

```bash
# Check available space in the container's writable layer
docker exec inference-server df -h /
```

---

## 10. Resource Management

### 10.1 CPU and Memory Limits

The Orin Nano has 6 CPU cores and 8GB of shared memory. Setting limits prevents
a runaway container from starving the system:

```bash
# Limit to 4 CPU cores and 4GB memory
docker run -d --runtime nvidia \
    --cpus 4.0 \
    --memory 4g \
    --memory-swap 4g \
    --name inference-server \
    my-inference:latest

# Pin to specific CPU cores (useful for real-time workloads)
docker run -d --runtime nvidia \
    --cpuset-cpus "2,3,4,5" \
    --memory 4g \
    my-inference:latest
```

Setting `--memory-swap` equal to `--memory` disables swap for that container,
preventing OOM-induced swap thrashing that would cripple inference latency.

### 10.2 Understanding Shared GPU Memory

On Jetson, there is no separate GPU memory to limit independently. CPU and GPU
share the same 8GB LPDDR5. Docker's `--memory` flag limits the container's RSS
(resident set size), which includes both CPU allocations and GPU allocations made
through unified memory.

A practical memory budget for the Orin Nano 8GB:

```
Total LPDDR5:           8192 MB
  System reserved:      ~700 MB  (kernel, drivers, systemd)
  Docker overhead:       ~50 MB  (per container, daemon)
  Inference container:  ~4000 MB (model + CUDA context + buffers)
  Preprocessing:        ~1000 MB
  API/networking:        ~200 MB
  Monitoring agent:      ~200 MB
  Headroom:            ~2042 MB
```

### 10.3 Monitoring Memory Usage

```bash
# Overall system memory including GPU
cat /proc/meminfo | head -5

# Per-container memory usage
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}"

# GPU-specific memory (Tegra unified memory)
cat /sys/kernel/debug/nvmap/iovmm/allocations 2>/dev/null | tail -20

# Jetson power/thermal/memory monitor
sudo tegrastats
# Output example:
# RAM 3456/7620MB (lfb 412x4MB) SWAP 0/0MB CPU [25%@1510,18%@1510,...] GR3D_FREQ 50%
```

### 10.4 cgroup v2 Resource Control

With cgroup v2 enabled (see Section 2.5):

```bash
# Check container's cgroup
docker inspect --format '{{.HostConfig.CgroupParent}}' inference-server

# View live resource usage via cgroup
cat /sys/fs/cgroup/system.slice/docker-<container-id>.scope/memory.current
cat /sys/fs/cgroup/system.slice/docker-<container-id>.scope/cpu.stat
```

### 10.5 OOM Configuration

Configure how the system handles out-of-memory conditions:

```bash
# Set OOM priority (lower = less likely to be killed)
docker run -d --runtime nvidia \
    --oom-score-adj -500 \
    --name critical-inference \
    my-inference:latest

# Disable OOM killer for critical containers (use with extreme caution)
docker run -d --runtime nvidia \
    --oom-kill-disable \
    --memory 4g \
    --name critical-inference \
    my-inference:latest
```

### 10.6 NVIDIA Power Mode Configuration

The Orin Nano supports multiple power modes that affect available compute:

```bash
# List available power modes
sudo nvpmodel -q --verbose

# Set 15W mode (maximum performance)
sudo nvpmodel -m 0

# Set 7W mode (power-constrained deployments)
sudo nvpmodel -m 1

# Lock clocks for consistent performance
sudo jetson_clocks

# In a container, read current power mode
docker run --rm --runtime nvidia \
    -v /etc/nvpmodel.conf:/etc/nvpmodel.conf:ro \
    nvcr.io/nvidia/l4t-base:r36.3.0 \
    cat /sys/devices/platform/gpu.0/devfreq/17000000.ga10b/cur_freq
```

---

## 11. Docker Compose for Multi-Service

### 11.1 Architecture Overview

A typical edge AI deployment consists of multiple cooperating services:

```
+----------+      +--------------+      +-----------+
|  Camera  | ---> | Preprocessor | ---> | Inference | ---> Results
|  Capture |      |   (resize,   |      | (TensorRT |
+----------+      |    normalize)|      |   engine) |
                  +--------------+      +-----------+
                                              |
                                              v
                                      +-------------+
                                      |   REST API  |
                                      | (Flask/Fast |
                                      |    API)     |
                                      +-------------+
                                              |
                                              v
                                      +-------------+
                                      |  Telemetry  |
                                      | (metrics +  |
                                      |   logging)  |
                                      +-------------+
```

### 11.2 Complete docker-compose.yml

```yaml
# docker-compose.yml for Jetson Orin Nano edge AI stack

version: "3.8"

services:
  # ---- Camera Capture Service ----
  camera:
    image: my-registry/camera-capture:v1.0
    runtime: nvidia
    restart: unless-stopped
    devices:
      - /dev/video0:/dev/video0
    volumes:
      - /tmp/argus_socket:/tmp/argus_socket
      - frame-buffer:/shared/frames
    environment:
      - CAMERA_ID=0
      - CAPTURE_WIDTH=1920
      - CAPTURE_HEIGHT=1080
      - CAPTURE_FPS=30
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
    healthcheck:
      test: ["CMD", "python3", "-c", "import os; assert os.path.exists('/shared/frames/latest.jpg')"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - ai-net

  # ---- Preprocessing Service ----
  preprocessor:
    image: my-registry/preprocessor:v1.0
    runtime: nvidia
    restart: unless-stopped
    depends_on:
      camera:
        condition: service_healthy
    volumes:
      - frame-buffer:/shared/frames:ro
      - tensor-buffer:/shared/tensors
    environment:
      - INPUT_SIZE=640
      - NORMALIZE=true
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - ai-net

  # ---- Inference Engine ----
  inference:
    image: my-registry/inference-engine:v1.0
    runtime: nvidia
    restart: unless-stopped
    depends_on:
      preprocessor:
        condition: service_healthy
    volumes:
      - /opt/models:/models:ro
      - tensor-buffer:/shared/tensors:ro
    environment:
      - MODEL_PATH=/models/yolov8n.engine
      - BATCH_SIZE=1
      - PRECISION=FP16
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 3G
    shm_size: "256m"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 30s
    networks:
      - ai-net

  # ---- REST API Gateway ----
  api:
    image: my-registry/api-gateway:v1.0
    restart: unless-stopped
    depends_on:
      inference:
        condition: service_healthy
    ports:
      - "8080:8080"
    environment:
      - INFERENCE_URL=http://inference:8080
      - LOG_LEVEL=info
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 3
    networks:
      - ai-net

  # ---- Telemetry Agent ----
  telemetry:
    image: my-registry/telemetry:v1.0
    restart: unless-stopped
    volumes:
      - /run/jtop.sock:/run/jtop.sock:ro
      - telemetry-data:/data
    environment:
      - PUSH_ENDPOINT=https://monitoring.example.com/api/v1/push
      - DEVICE_ID=${DEVICE_ID:-jetson-001}
      - PUSH_INTERVAL=60
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 128M
    networks:
      - ai-net

volumes:
  frame-buffer:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: size=128m
  tensor-buffer:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: size=64m
  telemetry-data:

networks:
  ai-net:
    driver: bridge
```

### 11.3 Running and Managing the Compose Stack

```bash
# Start the entire stack
docker compose up -d

# View service status
docker compose ps

# View logs from all services
docker compose logs -f

# View logs from a specific service
docker compose logs -f inference

# Scale (if applicable)
docker compose up -d --scale preprocessor=2

# Restart a single service
docker compose restart inference

# Stop everything
docker compose down

# Stop and remove volumes
docker compose down -v
```

### 11.4 Environment-Specific Overrides

Create a `docker-compose.override.yml` for development:

```yaml
# docker-compose.override.yml -- development overrides
version: "3.8"

services:
  inference:
    environment:
      - LOG_LEVEL=debug
      - PROFILE=true
    ports:
      - "8080:8080"   # Expose inference port directly for debugging
      - "6006:6006"   # TensorBoard

  camera:
    environment:
      - CAPTURE_FPS=15  # Lower FPS for development
    volumes:
      - ./test-images:/shared/frames  # Use test images instead of camera
```

Production override (`docker-compose.prod.yml`):

```yaml
# docker-compose.prod.yml
version: "3.8"

services:
  inference:
    restart: always
    logging:
      driver: json-file
      options:
        max-size: "5m"
        max-file: "3"

  api:
    restart: always

  camera:
    restart: always
```

Deploy with:

```bash
# Development
docker compose up -d

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## 12. Kubernetes on Jetson (K3s)

### 12.1 Why K3s for Jetson

K3s is a lightweight, certified Kubernetes distribution designed for edge and IoT.
It replaces etcd with SQLite, bundles containerd, and compiles to a single binary
under 100MB. This makes it feasible to run on the Orin Nano's constrained resources.

K3s memory footprint: ~300-500MB, compared to ~1-2GB for full Kubernetes.

### 12.2 Installing K3s on Orin Nano

```bash
# Install K3s as a single-node cluster
curl -sfL https://get.k3s.io | sh -s - \
    --write-kubeconfig-mode 644 \
    --kubelet-arg="feature-gates=DevicePlugins=true" \
    --kubelet-arg="cgroup-driver=systemd"

# Verify installation
kubectl get nodes
# NAME              STATUS   ROLES                  AGE   VERSION
# jetson-orin-001   Ready    control-plane,master   30s   v1.28.x+k3s1

# Check system pods
kubectl get pods -n kube-system
```

### 12.3 Configuring containerd for NVIDIA Runtime

K3s uses containerd. Configure it for NVIDIA runtime:

```bash
# Create the K3s containerd config template
sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd/

sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl << 'EOF'
version = 2

[plugins."io.containerd.internal.v1.opt"]
  path = "/var/lib/rancher/k3s/agent/containerd"

[plugins."io.containerd.grpc.v1.cri".containerd]
  default_runtime_name = "nvidia"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
  privileged_without_host_devices = false
  runtime_engine = ""
  runtime_root = ""
  runtime_type = "io.containerd.runc.v2"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
  runtime_type = "io.containerd.runc.v2"
EOF

# Restart K3s to pick up the new config
sudo systemctl restart k3s
```

### 12.4 NVIDIA Device Plugin for Kubernetes

The device plugin advertises GPU resources to the Kubernetes scheduler:

```bash
# Deploy NVIDIA device plugin
kubectl apply -f - << 'EOF'
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin
  template:
    metadata:
      labels:
        name: nvidia-device-plugin
    spec:
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: nvidia-device-plugin
          image: nvcr.io/nvidia/k8s-device-plugin:v0.14.3
          securityContext:
            privileged: true
          volumeMounts:
            - name: device-plugin
              mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
EOF

# Verify GPU is advertised
kubectl describe node jetson-orin-001 | grep -A5 "Allocatable"
# nvidia.com/gpu: 1
```

### 12.5 Deploying an Inference Workload

```yaml
# inference-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
  labels:
    app: inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
    spec:
      containers:
        - name: inference
          image: my-registry/inference-engine:v1.0
          ports:
            - containerPort: 8080
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "3Gi"
              cpu: "2"
            requests:
              memory: "2Gi"
              cpu: "1"
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
          env:
            - name: MODEL_PATH
              value: /models/yolov8n.engine
            - name: BATCH_SIZE
              value: "1"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 30
      volumes:
        - name: models
          hostPath:
            path: /opt/models
            type: Directory

---
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  selector:
    app: inference
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: NodePort
```

Apply and verify:

```bash
kubectl apply -f inference-deployment.yaml
kubectl get pods -l app=inference
kubectl logs -f deployment/inference-server
kubectl port-forward service/inference-service 8080:80

# Test
curl http://localhost:8080/health
```

### 12.6 Multi-Node K3s Cluster

For fleets of Jetson devices that need to cooperate:

```bash
# On the server node (first Jetson)
curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644

# Get the join token
sudo cat /var/lib/rancher/k3s/server/node-token
# Output: K10abc123::server:xyz789...

# On agent nodes (additional Jetsons)
curl -sfL https://get.k3s.io | K3S_URL=https://jetson-server:6443 \
    K3S_TOKEN="K10abc123::server:xyz789" sh -

# Verify on the server
kubectl get nodes
# NAME              STATUS   ROLES                  AGE   VERSION
# jetson-server     Ready    control-plane,master   10m   v1.28.x+k3s1
# jetson-agent-01   Ready    <none>                 30s   v1.28.x+k3s1
# jetson-agent-02   Ready    <none>                 15s   v1.28.x+k3s1
```

### 12.7 Helm for Packaging

Package your edge application as a Helm chart for repeatable deployments:

```bash
# Create a chart skeleton
helm create jetson-inference

# Structure:
# jetson-inference/
#   Chart.yaml
#   values.yaml
#   templates/
#     deployment.yaml
#     service.yaml
#     configmap.yaml
```

`values.yaml`:

```yaml
image:
  repository: my-registry/inference-engine
  tag: v1.0
  pullPolicy: IfNotPresent

model:
  path: /opt/models
  name: yolov8n.engine

resources:
  limits:
    nvidia.com/gpu: 1
    memory: 3Gi
    cpu: "2"
  requests:
    memory: 2Gi
    cpu: "1"

service:
  type: NodePort
  port: 80
  targetPort: 8080
  nodePort: 30080
```

Deploy:

```bash
helm install my-inference ./jetson-inference --values values.yaml
helm list
helm upgrade my-inference ./jetson-inference --set image.tag=v1.1
helm rollback my-inference 1
```

---

## 13. Fleet Management

### 13.1 The Fleet Management Problem

Managing a single Jetson is trivial. Managing 100 or 10,000 Jetsons in the field
requires answering:

- How do I push a new container image to all devices?
- How do I roll back if the new image has a bug?
- How do I monitor device health at scale?
- How do I handle devices that are offline when an update is pushed?
- How do I manage device-specific configuration?

### 13.2 Balena

Balena is purpose-built for fleet management of containerized edge devices.

**Setup:**

```bash
# Install Balena CLI
npm install -g balena-cli

# Login
balena login

# Create a fleet for Jetson Orin Nano
balena fleet create OrinNanoFleet --type jetson-orin-nano-devkit-nvme

# Download and flash a device image
balena os download jetson-orin-nano-devkit-nvme -o balena-orin.img
# Flash with Etcher or dd

# Push application code
balena push OrinNanoFleet
```

`docker-compose.yml` for Balena:

```yaml
version: "2.1"

services:
  inference:
    build: ./inference
    privileged: true
    restart: always
    labels:
      io.balena.features.gpu: "1"
      io.balena.features.kernel-modules: "1"
    volumes:
      - inference-data:/data
    environment:
      - MODEL_URL=https://models.example.com/yolov8n.engine
    ports:
      - "8080:8080"

volumes:
  inference-data:
```

Balena handles:
- OTA container updates with delta downloads (only changed layers)
- Automatic rollback on failed health checks
- Device dashboard with logs, terminal, environment variables
- Offline update queuing

### 13.3 AWS IoT Greengrass

AWS IoT Greengrass v2 can deploy Docker containers to Jetson devices:

```bash
# Install Greengrass core on Jetson
sudo -E java -Droot="/greengrass/v2" \
    -jar GreengrassInstaller/lib/Greengrass.jar \
    --aws-region us-west-2 \
    --thing-name JetsonOrinNano001 \
    --thing-group-name JetsonFleet \
    --component-default-user ggc_user:ggc_group \
    --provision true \
    --setup-system-service true
```

Deployment recipe for a container component:

```yaml
# recipe.yaml
---
RecipeFormatVersion: "2020-01-25"
ComponentName: com.example.JetsonInference
ComponentVersion: "1.0.0"
ComponentDescription: TensorRT inference on Jetson
ComponentPublisher: MyCompany
Manifests:
  - Platform:
      os: linux
      architecture: aarch64
    Lifecycle:
      install:
        Script: |
          docker pull my-registry/inference-engine:v1.0
      run:
        Script: |
          docker run --rm --runtime nvidia \
            -p 8080:8080 \
            -v /opt/models:/models:ro \
            --name inference-engine \
            my-registry/inference-engine:v1.0
      shutdown:
        Script: |
          docker stop inference-engine
    Artifacts:
      - URI: docker:my-registry/inference-engine:v1.0
```

### 13.4 Azure IoT Edge

Azure IoT Edge provides container orchestration on Jetson:

```bash
# Install IoT Edge runtime
wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y aziot-edge

# Configure with connection string
sudo iotedge config mp --connection-string "HostName=..."
sudo iotedge config apply
```

Deployment manifest:

```json
{
    "modulesContent": {
        "$edgeAgent": {
            "properties.desired": {
                "modules": {
                    "inference": {
                        "type": "docker",
                        "status": "running",
                        "restartPolicy": "always",
                        "settings": {
                            "image": "my-registry/inference-engine:v1.0",
                            "createOptions": {
                                "HostConfig": {
                                    "Runtime": "nvidia",
                                    "Binds": ["/opt/models:/models:ro"],
                                    "PortBindings": {
                                        "8080/tcp": [{"HostPort": "8080"}]
                                    },
                                    "DeviceRequests": [
                                        {
                                            "Count": -1,
                                            "Capabilities": [["gpu"]]
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
```

### 13.5 OTA Update Strategies

**Blue-Green deployment on a single device:**

```bash
#!/bin/bash
# ota-update.sh -- Blue-green deployment script

NEW_IMAGE="my-registry/inference-engine:v2.0"
OLD_CONTAINER="inference-blue"
NEW_CONTAINER="inference-green"
HEALTH_URL="http://localhost:8081/health"

# Pull new image
docker pull "$NEW_IMAGE" || { echo "Pull failed"; exit 1; }

# Start new version on a different port
docker run -d --runtime nvidia \
    --name "$NEW_CONTAINER" \
    -p 8081:8080 \
    -v /opt/models:/models:ro \
    "$NEW_IMAGE"

# Wait for health check
for i in $(seq 1 30); do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "New container healthy"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Health check failed, rolling back"
        docker stop "$NEW_CONTAINER"
        docker rm "$NEW_CONTAINER"
        exit 1
    fi
    sleep 2
done

# Swap ports: stop old, rebind new to production port
docker stop "$OLD_CONTAINER"
docker rm "$OLD_CONTAINER"
docker stop "$NEW_CONTAINER"
docker rm "$NEW_CONTAINER"

docker run -d --runtime nvidia \
    --name "$OLD_CONTAINER" \
    -p 8080:8080 \
    -v /opt/models:/models:ro \
    "$NEW_IMAGE"

echo "Update complete"
```

### 13.6 Rollback Strategies

```bash
# Tag-based rollback
docker stop inference-engine
docker rm inference-engine
docker run -d --runtime nvidia \
    --name inference-engine \
    -p 8080:8080 \
    my-registry/inference-engine:v1.0  # Previous known-good version

# Keep last N images for rollback
# In /etc/cron.d/docker-image-retention
0 4 * * * root docker images --format '{{.Repository}}:{{.Tag}}' | \
    grep inference-engine | sort -V | head -n -3 | \
    xargs -r docker rmi
```

---

## 14. CI/CD for Jetson

### 14.1 Architecture Overview

```
+------------------+     +------------------+     +------------------+
|   Developer      |     |   CI Server      |     |   Container      |
|   pushes code    | --> | (GitHub Actions / | --> |   Registry       |
|   to git         |     |  GitLab CI)      |     | (NVCR/DockerHub/ |
+------------------+     +------------------+     |  ECR/ACR)        |
                                                  +------------------+
                                                          |
                                                          v
                                                  +------------------+
                                                  | Fleet Manager    |
                                                  | (Balena/AWS IoT/ |
                                                  |  Azure IoT)      |
                                                  +------------------+
                                                          |
                                                          v
                                                  +------------------+
                                                  | Jetson Devices   |
                                                  | (pull and run)   |
                                                  +------------------+
```

### 14.2 GitHub Actions for arm64 Builds

```yaml
# .github/workflows/build-jetson.yml
name: Build Jetson Container

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/inference-engine

jobs:
  build-arm64:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
        with:
          platforms: arm64

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=sha,prefix=
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/arm64
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            L4T_VERSION=r36.3.0
            CUDA_ARCH=87

  test-on-jetson:
    needs: build-arm64
    runs-on: self-hosted  # Jetson device as self-hosted runner
    if: github.event_name != 'pull_request'

    steps:
      - name: Pull new image
        run: |
          docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

      - name: Run smoke test
        run: |
          docker run --rm --runtime nvidia \
            -v /opt/test-models:/models:ro \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            python3 -c "
          import tensorrt as trt
          logger = trt.Logger(trt.Logger.WARNING)
          runtime = trt.Runtime(logger)
          print('TensorRT initialized successfully')
          # Run a quick inference test
          "
        timeout-minutes: 5

      - name: Run integration test
        run: |
          # Start the inference server
          docker run -d --runtime nvidia \
            --name test-inference \
            -p 8080:8080 \
            -v /opt/test-models:/models:ro \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

          # Wait for health
          for i in $(seq 1 30); do
            curl -sf http://localhost:8080/health && break
            sleep 2
          done

          # Run test request
          curl -sf -X POST http://localhost:8080/predict \
            -H "Content-Type: application/json" \
            -d '{"image_path": "/models/test_image.jpg"}' | \
            python3 -c "import sys,json; d=json.load(sys.stdin); assert 'predictions' in d"

          # Cleanup
          docker stop test-inference
          docker rm test-inference
```

### 14.3 GitLab CI for Jetson

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  IMAGE_TAG: $CI_REGISTRY_IMAGE/inference-engine
  L4T_VERSION: r36.3.0

build-arm64:
  stage: build
  image: docker:24.0
  services:
    - docker:24.0-dind
  variables:
    DOCKER_BUILDKIT: 1
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    - docker buildx create --use
  script:
    - docker buildx build
        --platform linux/arm64
        --tag $IMAGE_TAG:$CI_COMMIT_SHORT_SHA
        --tag $IMAGE_TAG:latest
        --push
        --cache-from type=registry,ref=$IMAGE_TAG:buildcache
        --cache-to type=registry,ref=$IMAGE_TAG:buildcache,mode=max
        .

test-on-device:
  stage: test
  tags:
    - jetson-orin-nano  # GitLab runner on actual Jetson hardware
  script:
    - docker pull $IMAGE_TAG:$CI_COMMIT_SHORT_SHA
    - docker run --rm --runtime nvidia
        $IMAGE_TAG:$CI_COMMIT_SHORT_SHA
        python3 -m pytest /opt/app/tests/ -v
  allow_failure: false

deploy-fleet:
  stage: deploy
  only:
    - tags
  script:
    - balena push OrinNanoFleet --source .
    # Or for AWS IoT Greengrass:
    # - aws greengrassv2 create-deployment ...
```

### 14.4 Setting Up Jetson as a Self-Hosted CI Runner

For GitHub Actions:

```bash
# On the Jetson device
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-arm64.tar.gz -L \
    https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-arm64-2.311.0.tar.gz
tar xzf actions-runner-linux-arm64.tar.gz

# Configure
./config.sh --url https://github.com/your-org/your-repo \
    --token YOUR_TOKEN \
    --labels jetson-orin-nano,gpu,arm64

# Install as system service
sudo ./svc.sh install
sudo ./svc.sh start
```

For GitLab:

```bash
# Install GitLab Runner on Jetson
curl -L --output gitlab-runner \
    https://gitlab-runner-downloads.s3.amazonaws.com/latest/binaries/gitlab-runner-linux-arm64
chmod +x gitlab-runner
sudo mv gitlab-runner /usr/local/bin/

# Register
sudo gitlab-runner register \
    --url https://gitlab.com/ \
    --registration-token YOUR_TOKEN \
    --executor shell \
    --tag-list "jetson-orin-nano,gpu,arm64" \
    --description "Jetson Orin Nano Runner"

sudo gitlab-runner start
```

### 14.5 Automated Model Deployment Pipeline

```yaml
# .github/workflows/model-deploy.yml
name: Deploy Updated Model

on:
  workflow_dispatch:
    inputs:
      model_url:
        description: "URL to the new TensorRT engine file"
        required: true
      model_name:
        description: "Model filename"
        required: true
        default: "yolov8n.engine"

jobs:
  validate-model:
    runs-on: self-hosted
    steps:
      - name: Download model
        run: |
          wget -O /tmp/${{ inputs.model_name }} "${{ inputs.model_url }}"

      - name: Validate model on device
        run: |
          docker run --rm --runtime nvidia \
            -v /tmp/${{ inputs.model_name }}:/models/${{ inputs.model_name }}:ro \
            my-registry/inference-engine:latest \
            python3 -c "
          import tensorrt as trt
          logger = trt.Logger(trt.Logger.WARNING)
          with open('/models/${{ inputs.model_name }}', 'rb') as f:
              runtime = trt.Runtime(logger)
              engine = runtime.deserialize_cuda_engine(f.read())
              assert engine is not None, 'Failed to load engine'
              print(f'Engine loaded: {engine.num_bindings} bindings')
              for i in range(engine.num_bindings):
                  print(f'  {engine.get_binding_name(i)}: {engine.get_binding_shape(i)}')
          "

      - name: Deploy model to fleet storage
        run: |
          aws s3 cp /tmp/${{ inputs.model_name }} \
            s3://jetson-fleet-models/${{ inputs.model_name }}

      - name: Trigger fleet update
        run: |
          # Signal devices to pull new model
          aws iot-data publish \
            --topic "fleet/models/update" \
            --payload '{"model": "${{ inputs.model_name }}", "version": "${{ github.run_number }}"}'
```

---

## 15. Monitoring and Logging

### 15.1 Monitoring Architecture for Edge

```
Jetson Device                          Cloud/On-Prem
+----------------------------------+   +---------------------------+
| +----------+ +----------------+  |   | +----------+ +---------+ |
| | App      | | Prometheus     |  |   | | Prometheus| | Grafana | |
| | Container| | Node Exporter  |------>| | Central  | |         | |
| +----------+ +----------------+  |   | +----------+ +---------+ |
|                                  |   |                           |
| +----------+ +----------------+  |   | +----------+ +---------+ |
| | tegra-   | | Promtail /     |  |   | | Loki     | |         | |
| | stats    | | Fluentd        |------>| |          | |         | |
| +----------+ +----------------+  |   | +----------+ +---------+ |
+----------------------------------+   +---------------------------+
```

### 15.2 Prometheus Node Exporter on Jetson

```yaml
# docker-compose.monitoring.yml
version: "3.8"

services:
  node-exporter:
    image: prom/node-exporter:v1.7.0
    restart: unless-stopped
    network_mode: host
    pid: host
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--path.rootfs=/rootfs'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    deploy:
      resources:
        limits:
          cpus: "0.25"
          memory: 64M
```

### 15.3 Custom Jetson Metrics Exporter

Standard node-exporter misses Jetson-specific metrics. Create a custom exporter:

```python
#!/usr/bin/env python3
# jetson_exporter.py -- Prometheus exporter for Jetson metrics

import subprocess
import re
import time
from prometheus_client import start_http_server, Gauge

# Define metrics
gpu_load = Gauge('jetson_gpu_load_percent', 'GPU utilization percentage')
gpu_freq = Gauge('jetson_gpu_freq_mhz', 'GPU clock frequency in MHz')
cpu_temp = Gauge('jetson_cpu_temp_celsius', 'CPU temperature')
gpu_temp = Gauge('jetson_gpu_temp_celsius', 'GPU temperature')
power_total = Gauge('jetson_power_total_mw', 'Total board power in milliwatts')
ram_used = Gauge('jetson_ram_used_mb', 'RAM used in MB')
ram_total = Gauge('jetson_ram_total_mb', 'Total RAM in MB')
swap_used = Gauge('jetson_swap_used_mb', 'Swap used in MB')
dla_load = Gauge('jetson_dla_load_percent', 'DLA utilization', ['engine'])
emc_freq = Gauge('jetson_emc_freq_mhz', 'Memory controller frequency in MHz')
nvpmodel = Gauge('jetson_nvpmodel_mode', 'Current NVP model power mode')

def read_sysfs(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return None

def collect_metrics():
    # GPU load
    val = read_sysfs('/sys/devices/gpu.0/load')
    if val:
        gpu_load.set(int(val) / 10.0)

    # GPU frequency
    val = read_sysfs('/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq')
    if val:
        gpu_freq.set(int(val) / 1000000)

    # Temperatures
    for zone_path in ['/sys/class/thermal/thermal_zone0/temp',
                      '/sys/class/thermal/thermal_zone1/temp']:
        val = read_sysfs(zone_path)
        if val and 'cpu' in read_sysfs(zone_path.replace('temp', 'type')).lower():
            cpu_temp.set(int(val) / 1000.0)
        elif val and 'gpu' in read_sysfs(zone_path.replace('temp', 'type')).lower():
            gpu_temp.set(int(val) / 1000.0)

    # Power (INA3221 sensor)
    val = read_sysfs('/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input')
    if val:
        power_total.set(int(val))

    # Memory
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
    total = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1)) / 1024
    available = int(re.search(r'MemAvailable:\s+(\d+)', meminfo).group(1)) / 1024
    ram_total.set(total)
    ram_used.set(total - available)

if __name__ == '__main__':
    start_http_server(9101)
    print("Jetson exporter running on :9101")
    while True:
        collect_metrics()
        time.sleep(5)
```

Dockerfile for the exporter:

```dockerfile
FROM python:3.10-slim

RUN pip install --no-cache-dir prometheus-client==0.19.0

COPY jetson_exporter.py /opt/

CMD ["python3", "/opt/jetson_exporter.py"]
```

### 15.4 Centralized Logging with Promtail and Loki

```yaml
# Promtail on each Jetson device
# promtail-config.yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: https://loki.example.com/loki/api/v1/push
    external_labels:
      device_id: ${DEVICE_ID}
      fleet: orin-nano-fleet

scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        target_label: container
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: stream
    pipeline_stages:
      - json:
          expressions:
            log: log
            timestamp: time
      - timestamp:
          source: timestamp
          format: RFC3339Nano
```

Docker Compose entry:

```yaml
  promtail:
    image: grafana/promtail:2.9.0
    restart: unless-stopped
    volumes:
      - ./promtail-config.yaml:/etc/promtail/config.yaml:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yaml
    deploy:
      resources:
        limits:
          cpus: "0.25"
          memory: 64M
```

### 15.5 Fluentd Alternative

For environments standardized on Fluentd:

```xml
<!-- fluent.conf -->
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<filter docker.**>
  @type record_transformer
  <record>
    device_id "#{ENV['DEVICE_ID']}"
    hostname "#{Socket.gethostname}"
  </record>
</filter>

<match docker.**>
  @type elasticsearch
  host logs.example.com
  port 9200
  index_name jetson-logs
  <buffer>
    flush_interval 30s
    chunk_limit_size 2M
    total_limit_size 64M
  </buffer>
</match>
```

Configure Docker to log to Fluentd:

```json
{
    "log-driver": "fluentd",
    "log-opts": {
        "fluentd-address": "localhost:24224",
        "tag": "docker.{{.Name}}"
    }
}
```

### 15.6 Health Monitoring Script

A lightweight health monitor that does not depend on external services:

```bash
#!/bin/bash
# /usr/local/bin/jetson-health-check.sh
# Runs every minute via cron

LOG_FILE="/var/log/jetson-health.log"
ALERT_ENDPOINT="https://alerts.example.com/webhook"
DEVICE_ID=$(hostname)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Check container status
UNHEALTHY=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" 2>/dev/null)
if [ -n "$UNHEALTHY" ]; then
    echo "$TIMESTAMP ALERT: Unhealthy containers: $UNHEALTHY" >> "$LOG_FILE"
    curl -sf -X POST "$ALERT_ENDPOINT" \
        -H "Content-Type: application/json" \
        -d "{\"device\":\"$DEVICE_ID\",\"alert\":\"unhealthy_container\",\"containers\":\"$UNHEALTHY\"}" \
        > /dev/null 2>&1
fi

# Check disk space
DISK_USAGE=$(df / --output=pcent | tail -1 | tr -d '% ')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "$TIMESTAMP ALERT: Disk usage at ${DISK_USAGE}%" >> "$LOG_FILE"
    # Auto-cleanup
    docker system prune -f > /dev/null 2>&1
fi

# Check GPU temperature
GPU_TEMP_RAW=$(cat /sys/class/thermal/thermal_zone1/temp 2>/dev/null)
if [ -n "$GPU_TEMP_RAW" ]; then
    GPU_TEMP=$((GPU_TEMP_RAW / 1000))
    if [ "$GPU_TEMP" -gt 85 ]; then
        echo "$TIMESTAMP ALERT: GPU temperature ${GPU_TEMP}C" >> "$LOG_FILE"
    fi
fi

# Check memory
MEM_AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "$MEM_AVAIL" -lt 512 ]; then
    echo "$TIMESTAMP ALERT: Available memory ${MEM_AVAIL}MB" >> "$LOG_FILE"
fi

# Rotate log
if [ "$(wc -l < "$LOG_FILE" 2>/dev/null)" -gt 10000 ]; then
    tail -5000 "$LOG_FILE" > "${LOG_FILE}.tmp"
    mv "${LOG_FILE}.tmp" "$LOG_FILE"
fi
```

Cron entry:

```
* * * * * root /usr/local/bin/jetson-health-check.sh
```

---

## 16. Security in Containers

### 16.1 Threat Model for Edge Devices

Edge devices face threats that data center servers do not:

- **Physical access** -- An attacker can physically access the device
- **Untrusted networks** -- Devices may be on public or semi-public networks
- **Long deployment lifetimes** -- Devices may run for years without hands-on maintenance
- **Supply chain** -- Container images may be tampered with in transit

### 16.2 Rootless Containers

Running containers without root privileges reduces the impact of container escapes:

```bash
# Install rootless Docker prerequisites
sudo apt-get install -y uidmap dbus-user-session

# Enable rootless mode for a user
dockerd-rootless-setuptool.sh install

# Verify
docker context use rootless
docker run --rm hello-world
```

Caveat: On Jetson, rootless mode has limitations with GPU access. The NVIDIA runtime
requires access to `/dev/nv*` devices which are typically root-owned. Workaround:

```bash
# Create a udev rule granting device access to a specific group
sudo tee /etc/udev/rules.d/99-nvidia-jetson.rules << 'EOF'
SUBSYSTEM=="nvhost", GROUP="docker", MODE="0660"
SUBSYSTEM=="nvmap", GROUP="docker", MODE="0660"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 16.3 Read-Only Root Filesystem

Prevent containers from writing to the filesystem (except designated volumes):

```bash
docker run -d --runtime nvidia \
    --read-only \
    --tmpfs /tmp:rw,size=64m \
    --tmpfs /run:rw,size=16m \
    -v inference-data:/data \
    my-inference:latest
```

In Docker Compose:

```yaml
services:
  inference:
    image: my-inference:latest
    read_only: true
    tmpfs:
      - /tmp:size=64m
      - /run:size=16m
    volumes:
      - inference-data:/data
```

### 16.4 Dropping Capabilities

Remove unnecessary Linux capabilities:

```bash
docker run -d --runtime nvidia \
    --cap-drop ALL \
    --cap-add SYS_NICE \
    --security-opt no-new-privileges \
    my-inference:latest
```

The `--cap-drop ALL --cap-add SYS_NICE` pattern gives the container only the
ability to adjust scheduling priority (useful for real-time inference) while
removing all other capabilities like network administration, raw socket access,
kernel module loading, and so on.

### 16.5 Secrets Management

Never bake secrets into container images.

**Method 1: Docker secrets (Compose/Swarm)**

```yaml
# docker-compose.yml
services:
  inference:
    image: my-inference:latest
    secrets:
      - model_api_key
      - tls_cert

secrets:
  model_api_key:
    file: ./secrets/api_key.txt
  tls_cert:
    file: ./secrets/tls.pem
```

Inside the container, secrets appear at `/run/secrets/<name>`.

**Method 2: Environment variables from a .env file**

```bash
# .env (never commit to git)
MODEL_API_KEY=sk-abc123xyz
CLOUD_ENDPOINT=https://api.example.com

# docker-compose.yml
services:
  inference:
    env_file: .env
```

**Method 3: HashiCorp Vault agent**

```bash
docker run -d --runtime nvidia \
    -e VAULT_ADDR=https://vault.example.com \
    -e VAULT_ROLE=jetson-inference \
    -v /var/run/vault:/var/run/vault \
    my-inference-with-vault:latest
```

### 16.6 Image Signing and Verification

Sign images with Docker Content Trust (Notary):

```bash
# Enable content trust
export DOCKER_CONTENT_TRUST=1

# Push a signed image
docker push my-registry/inference-engine:v1.0
# You will be prompted for signing keys

# On Jetson devices, enforce signed images only
# In /etc/docker/daemon.json:
{
    "content-trust": {
        "mode": "enforced"
    }
}
```

Alternatively, use cosign (Sigstore) for keyless signing:

```bash
# Sign
cosign sign --yes my-registry/inference-engine:v1.0

# Verify on device before running
cosign verify my-registry/inference-engine:v1.0 \
    --certificate-identity user@example.com \
    --certificate-oidc-issuer https://accounts.google.com
```

### 16.7 Vulnerability Scanning

Scan images before deploying to the fleet:

```bash
# Using Trivy
trivy image --severity HIGH,CRITICAL my-registry/inference-engine:v1.0

# In CI pipeline (GitHub Actions)
- name: Scan image
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: my-registry/inference-engine:v1.0
    format: 'table'
    exit-code: '1'
    severity: 'CRITICAL,HIGH'
```

### 16.8 Network Security

```bash
# Restrict container network access
docker network create --internal isolated-net

# Container can only talk to other containers on this network,
# not to the outside world
docker run -d --runtime nvidia \
    --network isolated-net \
    my-inference:latest

# Use iptables to restrict which containers can reach the internet
sudo iptables -I DOCKER-USER -i docker0 -d 0.0.0.0/0 -j DROP
sudo iptables -I DOCKER-USER -i docker0 -d 192.168.1.0/24 -j ACCEPT
```

---

## 17. Common Issues and Debugging

### 17.1 Container OOM on 8GB Shared Memory

**Symptom:** Container is killed with exit code 137. `dmesg` shows OOM killer.

**Root cause:** The 8GB LPDDR5 is shared between CPU, GPU, system services, and all
containers. A TensorRT engine loading a large model can easily consume 2-3GB alone.

**Diagnosis:**

```bash
# Check OOM events
dmesg | grep -i oom

# Monitor memory in real-time
watch -n 1 'free -m && echo "---" && docker stats --no-stream --format "{{.Name}}: {{.MemUsage}}"'

# Check what the GPU is consuming
cat /sys/kernel/debug/nvmap/iovmm/allocations 2>/dev/null | \
    awk '{sum += $2} END {print "GPU allocations: " sum/1024/1024 " MB"}'
```

**Solutions:**

```bash
# 1. Set explicit memory limits to prevent one container from taking everything
docker run --memory 3g --memory-swap 3g ...

# 2. Use FP16 or INT8 models instead of FP32
# FP32 YOLOv8n: ~25MB engine
# FP16 YOLOv8n: ~13MB engine
# INT8 YOLOv8n: ~7MB engine
# (GPU memory during inference is 5-10x the engine file size)

# 3. Add swap as a safety net (NVMe-backed)
sudo fallocate -l 4G /mnt/nvme/swapfile
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile
echo '/mnt/nvme/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 4. Reduce CUDA context overhead -- share a single context
#    by using CUDA MPS (Multi-Process Service)
sudo nvidia-cuda-mps-control -d
```

### 17.2 GPU Not Accessible in Container

**Symptom:** `CUDA error: no CUDA-capable device is detected` inside container.

**Checklist:**

```bash
# 1. Is the NVIDIA runtime configured?
docker info | grep -i runtime
# Should show: Runtimes: nvidia runc

# 2. Is the default runtime set?
docker info | grep "Default Runtime"
# Should show: Default Runtime: nvidia

# 3. Are the device nodes present?
ls -la /dev/nvhost-* /dev/nvmap
# If missing, the NVIDIA driver may not be loaded

# 4. Are NVIDIA kernel modules loaded?
lsmod | grep nv
# Should show: nvgpu, nvmap, nvhost, etc.

# 5. Test with explicit runtime
docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r36.3.0 \
    ls /dev/nv*

# 6. Test with privileged mode (debugging only)
docker run --rm --privileged nvcr.io/nvidia/l4t-base:r36.3.0 \
    python3 -c "import ctypes; ctypes.CDLL('libcudart.so'); print('CUDA OK')"

# 7. Reinstall NVIDIA Container Toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

### 17.3 L4T Version Mismatch

**Symptom:** Containers start but GPU operations fail with cryptic errors. Segfaults
in CUDA calls. `illegal instruction` errors.

**Diagnosis:**

```bash
# Host L4T version
head -1 /etc/nv_tegra_release
# Example: # R36 (release), REVISION: 3.0

# Container L4T version
docker run --rm nvcr.io/nvidia/l4t-base:r36.3.0 head -1 /etc/nv_tegra_release

# These MUST match the major version (R36)
```

**Solution:** Always match container base image to host L4T:

```bash
# If host is R36.3.0, use r36.3.0 base images
FROM nvcr.io/nvidia/l4t-tensorrt:r36.3.0
```

### 17.4 Filesystem Bloat

**Symptom:** Disk is full. Cannot pull new images or write data.

**Diagnosis:**

```bash
# Docker-specific disk usage
docker system df -v

# Find large images
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}" | sort -t$'\t' -k2 -h

# Find large containers (writable layers)
docker ps -s --format "table {{.Names}}\t{{.Size}}"

# Find large volumes
docker volume ls -q | xargs -I {} docker volume inspect {} --format '{{.Name}}: {{.Mountpoint}}' | \
    while read line; do
        path=$(echo "$line" | cut -d: -f2 | tr -d ' ')
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "$line -> $size"
    done
```

**Cleanup:**

```bash
# Remove stopped containers, unused networks, dangling images, build cache
docker system prune -f

# Also remove unused images (not just dangling)
docker system prune -a -f

# Remove specific old images
docker rmi $(docker images --filter "before=my-inference:v2.0" -q) 2>/dev/null

# Clean apt cache inside future builds
# Always end RUN commands with:
RUN apt-get update && apt-get install -y ... && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache
```

### 17.5 Container Networking Issues

**Symptom:** Containers cannot reach each other or the internet.

```bash
# Check Docker network
docker network ls
docker network inspect bridge

# Test DNS resolution between containers
docker exec container-a ping container-b

# If using bridge networking, check iptables
sudo iptables -L -n -v | grep DOCKER

# Common fix: restart Docker networking
sudo systemctl restart docker

# For DNS issues inside containers
docker run --rm --dns 8.8.8.8 my-image nslookup google.com

# Check if ip_forward is enabled
cat /proc/sys/net/ipv4/ip_forward
# Should be 1. If 0:
sudo sysctl net.ipv4.ip_forward=1
```

### 17.6 Camera Not Working in Container

**Symptom:** `/dev/video0` not found or camera returns black frames.

```bash
# Check if camera is detected on host first
v4l2-ctl --list-devices

# For CSI cameras, check nvargus-daemon
sudo systemctl status nvargus-daemon
# If not running:
sudo systemctl start nvargus-daemon

# Test CSI camera on host
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
    nvvidconv ! xvimagesink

# Pass the Argus socket into the container
docker run --rm --runtime nvidia \
    --volume /tmp/argus_socket:/tmp/argus_socket \
    --device /dev/video0 \
    my-camera-app

# If using USB camera, check permissions
ls -la /dev/video0
# If permission denied, add user to video group or use --group-add video
docker run --rm --group-add video --device /dev/video0 my-camera-app
```

### 17.7 Slow Container Startup

**Symptom:** Container takes 30+ seconds to start.

**Common causes and fixes:**

```bash
# 1. Large image pull -- use pre-pulled images
docker pull my-inference:latest  # Do this during maintenance window
# In compose: pull_policy: never

# 2. TensorRT engine building at startup
# Pre-build .engine files instead of converting .onnx at runtime
# trtexec on the host or in a build container:
docker run --rm --runtime nvidia \
    -v /opt/models:/models \
    nvcr.io/nvidia/l4t-tensorrt:r36.3.0 \
    trtexec --onnx=/models/yolov8n.onnx \
            --saveEngine=/models/yolov8n.engine \
            --fp16 \
            --workspace=1024

# 3. CUDA context initialization (~2-5 seconds, unavoidable)
# Mitigate with CUDA lazy loading:
ENV CUDA_MODULE_LOADING=LAZY

# 4. Python import overhead
# Use compiled Python (.pyc) and minimize imports
python3 -m compileall /opt/app/
```

### 17.8 Permission Errors

**Symptom:** `Permission denied` when accessing devices, sockets, or mounted volumes.

```bash
# Check user inside container
docker exec my-container id
# uid=0(root) gid=0(root) -- if running as root, should not have permission issues

# If running as non-root user in container:
docker run --rm --user 1000:1000 \
    --group-add video \
    --group-add $(getent group docker | cut -d: -f3) \
    --device /dev/video0 \
    my-inference:latest

# Fix volume mount permissions
# Option 1: Match UID/GID
docker run --user $(id -u):$(id -g) -v /opt/models:/models:ro ...

# Option 2: Fix ownership in Dockerfile
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup appuser && \
    chown -R appuser:appgroup /opt/app
USER appuser
```

### 17.9 Debugging Running Containers

```bash
# Enter a running container
docker exec -it inference-server bash

# View container logs
docker logs --tail 100 -f inference-server

# View container resource usage
docker stats inference-server

# Inspect container configuration
docker inspect inference-server

# View container processes
docker top inference-server

# Copy files out of a container for analysis
docker cp inference-server:/opt/app/error.log ./error.log

# Run tegrastats inside a container (if available)
docker exec inference-server tegrastats --interval 1000

# Attach strace to a process inside the container
# First, get the PID on the host:
PID=$(docker inspect --format '{{.State.Pid}}' inference-server)
sudo strace -f -p $PID -e trace=open,read,write 2>&1 | head -50

# Check NVIDIA GPU state from inside the container
docker exec inference-server cat /sys/devices/gpu.0/load
```

### 17.10 Performance Debugging

```bash
# Profile CUDA operations
docker run --rm --runtime nvidia \
    -v /opt/models:/models:ro \
    --cap-add SYS_ADMIN \
    my-inference:latest \
    nsys profile --stats=true python3 inference.py

# Check if GPU clocks are throttled
sudo jetson_clocks --show
# If frequencies are low, power mode may be restricting them:
sudo nvpmodel -q

# Check for thermal throttling
cat /sys/class/thermal/thermal_zone*/temp
cat /sys/class/thermal/thermal_zone*/trip_point_*_temp

# Container-level CPU profiling
docker run --rm --runtime nvidia \
    --cap-add SYS_PTRACE \
    my-inference:latest \
    python3 -m cProfile -s cumulative inference.py

# Measure inference latency end-to-end
docker exec inference-server python3 -c "
import time
import requests
import statistics

latencies = []
for i in range(100):
    start = time.perf_counter()
    resp = requests.post('http://localhost:8080/predict',
                         json={'image_path': '/models/test.jpg'})
    end = time.perf_counter()
    latencies.append((end - start) * 1000)

print(f'Latency (ms): mean={statistics.mean(latencies):.1f}, '
      f'p50={statistics.median(latencies):.1f}, '
      f'p99={sorted(latencies)[98]:.1f}, '
      f'min={min(latencies):.1f}, max={max(latencies):.1f}')
"
```

### 17.11 Quick Reference: Essential Commands

```bash
# System info
sudo tegrastats                              # Real-time Jetson stats
sudo jetson_clocks --show                    # Current clock frequencies
sudo nvpmodel -q                             # Current power mode
head -1 /etc/nv_tegra_release               # L4T version

# Docker operations
docker system df -v                          # Disk usage breakdown
docker stats --no-stream                     # Container resource usage
docker system prune -f                       # Clean unused resources
docker inspect <container>                   # Full container config

# GPU debugging
cat /sys/devices/gpu.0/load                 # GPU utilization (x/1000)
cat /sys/kernel/debug/nvmap/iovmm/clients   # GPU memory clients
ls /dev/nvhost-* /dev/nvmap                 # Device nodes

# Networking
docker network ls                            # List networks
docker network inspect bridge                # Bridge network details
docker exec <container> cat /etc/resolv.conf # DNS configuration

# Logs
docker logs --tail 200 -f <container>        # Container logs
journalctl -u docker --since "1 hour ago"    # Docker daemon logs
dmesg | tail -50                             # Kernel messages
```

---

## Summary

This guide covered the complete lifecycle of containerized edge AI on the Jetson
Orin Nano 8GB. The key takeaways:

1. **Always set NVIDIA as the default runtime** to avoid GPU access issues across
   the fleet.

2. **Match L4T versions exactly** between host and container base images. This is
   the single most common source of mysterious GPU failures.

3. **Budget the 8GB carefully.** With shared CPU/GPU memory, plan your container
   memory limits, model precision, and service count around a concrete memory budget.

4. **Use multi-stage builds** to keep images small. A 5GB image on a 32GB NVMe is
   a significant fraction of your total storage.

5. **Pre-build TensorRT engines** for the target device. Converting ONNX to TensorRT
   at container startup wastes minutes on every restart and consumes memory for the
   builder.

6. **Use K3s, not full Kubernetes**, if you need orchestration on Jetson. The memory
   savings are substantial.

7. **Implement health checks and automated rollbacks** in your fleet management
   strategy. An unreachable edge device is far more expensive to fix than a failing
   cloud server.

8. **Log rotation is not optional.** Without it, a chatty container will fill the
   disk and bring down the entire device.

9. **Security is amplified at the edge.** Physical access, untrusted networks, and
   long deployment lifetimes mean you must enforce read-only filesystems, drop
   capabilities, sign images, and manage secrets properly from day one.

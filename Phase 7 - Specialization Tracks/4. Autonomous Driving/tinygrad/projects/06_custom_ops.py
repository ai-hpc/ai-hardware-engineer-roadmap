"""
Project 6: Custom Operations and Backend Extensions
=====================================================
Goal: Implement custom ops from primitives, verify with gradients,
and understand what the compiler generates for them.

Run with:
  python3 06_custom_ops.py
  DEBUG=1 python3 06_custom_ops.py       # kernel counts
  DEBUG=3 python3 06_custom_ops.py       # generated kernel code
"""

import numpy as np


# ---------------------------------------------------------------------------
# Numerical gradient utility
# ---------------------------------------------------------------------------
def numerical_grad(fn, x_np, eps=1e-4):
    """Central difference finite-difference gradient estimator."""
    grad = np.zeros_like(x_np)
    for i in range(x_np.size):
        xp, xm = x_np.copy(), x_np.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        grad.flat[i] = (fn(xp) - fn(xm)) / (2 * eps)
    return grad


def check_grad(name, tg_fn, np_fn, x_np, atol=1e-3):
    """Compare tinygrad autograd gradient to finite-differences."""
    from tinygrad import Tensor
    x = Tensor(x_np.astype(np.float32), requires_grad=True)
    out = tg_fn(x)
    out.sum().backward()
    analytical = x.grad.numpy()
    numerical = numerical_grad(lambda xn: np_fn(xn.astype(np.float32)).sum(), x_np)
    err = np.abs(analytical - numerical).max()
    status = "✓" if err < atol else "✗"
    print(f"  {status} {name:25s}  grad_error={err:.2e}")
    return err < atol


# ---------------------------------------------------------------------------
# Task 1: Common activations from primitives
# ---------------------------------------------------------------------------
def task1_activations():
    print("\n=== Task 1: Custom Activations from Primitives ===")
    from tinygrad import Tensor

    x_np = np.random.randn(4, 8).astype(np.float32)

    # --- GELU ---
    # Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    def gelu(x: Tensor) -> Tensor:
        c = (2.0 / np.pi) ** 0.5
        return x * 0.5 * (1 + (c * (x + 0.044715 * x.pow(3))).tanh())

    def gelu_np(x): return x * 0.5 * (1 + np.tanh((2.0/np.pi)**0.5 * (x + 0.044715 * x**3)))

    tg_gelu = gelu(Tensor(x_np)).numpy()
    np_gelu = gelu_np(x_np)
    print(f"GELU  forward error:  {np.abs(tg_gelu - np_gelu).max():.2e}")
    check_grad("GELU", gelu, gelu_np, x_np)

    # --- Swish / SiLU ---
    # swish(x) = x * sigmoid(x)
    def swish(x: Tensor) -> Tensor:
        return x * x.sigmoid()

    def swish_np(x): return x / (1 + np.exp(-x))

    tg_swish = swish(Tensor(x_np)).numpy()
    np_swish = swish_np(x_np)
    print(f"Swish forward error:  {np.abs(tg_swish - np_swish).max():.2e}")
    check_grad("Swish (SiLU)", swish, swish_np, x_np)

    # --- Mish ---
    # mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    def mish(x: Tensor) -> Tensor:
        return x * (x.exp().add(1).log()).tanh()

    def mish_np(x): return x * np.tanh(np.log1p(np.exp(np.clip(x, -88, 88))))

    tg_mish = mish(Tensor(x_np)).numpy()
    np_mish = mish_np(x_np)
    print(f"Mish  forward error:  {np.abs(tg_mish - np_mish).max():.2e}")
    check_grad("Mish", mish, mish_np, x_np)


# ---------------------------------------------------------------------------
# Task 2: Normalization layers from primitives
# ---------------------------------------------------------------------------
def task2_normalization():
    print("\n=== Task 2: Normalization from Primitives ===")
    from tinygrad import Tensor

    x_np = np.random.randn(8, 32).astype(np.float32)

    # --- RMS Norm ---
    # rms_norm(x) = x / sqrt(mean(x^2) + eps)
    def rms_norm(x: Tensor, eps=1e-6) -> Tensor:
        rms = (x * x).mean(axis=-1, keepdim=True).add(eps).sqrt()
        return x / rms

    def rms_norm_np(x, eps=1e-6):
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        return x / rms

    tg_rms = rms_norm(Tensor(x_np)).numpy()
    np_rms = rms_norm_np(x_np)
    print(f"RMSNorm forward error:  {np.abs(tg_rms - np_rms).max():.2e}")
    check_grad("RMSNorm", rms_norm, rms_norm_np, x_np, atol=5e-3)

    # --- Layer Norm ---
    # layer_norm(x) = (x - mean(x)) / sqrt(var(x) + eps)
    def layer_norm(x: Tensor, eps=1e-5) -> Tensor:
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) * (x - mean)).mean(axis=-1, keepdim=True)
        return (x - mean) / (var.add(eps).sqrt())

    def layer_norm_np(x, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean)**2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    tg_ln = layer_norm(Tensor(x_np)).numpy()
    np_ln = layer_norm_np(x_np)
    print(f"LayerNorm forward error: {np.abs(tg_ln - np_ln).max():.2e}")
    check_grad("LayerNorm", layer_norm, layer_norm_np, x_np, atol=5e-3)


# ---------------------------------------------------------------------------
# Task 3: Attention mechanism from primitives
# ---------------------------------------------------------------------------
def task3_attention():
    print("\n=== Task 3: Scaled Dot-Product Attention ===")
    from tinygrad import Tensor

    B, T, D = 2, 8, 16  # batch, sequence, dimension
    H = 4                # heads
    d_head = D // H

    Q = Tensor.randn(B, H, T, d_head)
    K = Tensor.randn(B, H, T, d_head)
    V = Tensor.randn(B, H, T, d_head)

    def scaled_dot_product_attention(q, k, v):
        # score = (Q @ K^T) / sqrt(d_head)
        scale = d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale    # (B, H, T, T)
        weights = scores.softmax(axis=-1)              # (B, H, T, T)
        return weights @ v                             # (B, H, T, d_head)

    out = scaled_dot_product_attention(Q, K, V)
    print(f"  Input  shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"  Output shape:  {out.shape}")

    # Causal (masked) attention — upper triangle = -inf
    def causal_attention(q, k, v):
        scale = d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale       # (B, H, T, T)
        # Causal mask: -inf for positions j > i
        mask = Tensor.ones(T, T).tril().where(
            Tensor.zeros(T, T), Tensor.full((T, T), float('-inf'))
        )
        scores = scores + mask
        weights = scores.softmax(axis=-1)
        return weights @ v

    out_causal = causal_attention(Q, K, V)
    print(f"  Causal attn:   {out_causal.shape}")

    sched = out_causal.schedule()
    print(f"  Kernels for causal attention: {len(sched)}")
    print(f"  Run with DEBUG=1 to confirm kernel count.")


# ---------------------------------------------------------------------------
# Task 4: Loss functions from primitives
# ---------------------------------------------------------------------------
def task4_loss_functions():
    print("\n=== Task 4: Loss Functions from Primitives ===")
    from tinygrad import Tensor

    # --- Huber Loss ---
    def huber_loss(pred: Tensor, target: Tensor, delta=1.0) -> Tensor:
        diff = pred - target
        abs_diff = diff.abs()
        quadratic = 0.5 * diff * diff
        linear = delta * (abs_diff - 0.5 * delta)
        return (abs_diff <= delta).where(quadratic, linear).mean()

    pred = Tensor.randn(32, 1)
    target = Tensor.randn(32, 1)
    loss = huber_loss(pred, target)
    print(f"  Huber Loss: {loss.item():.4f}")

    # Gradient check
    pred_np = np.random.randn(8, 1).astype(np.float32)
    target_np = np.random.randn(8, 1).astype(np.float32)

    def huber_np(p, t=target_np, delta=1.0):
        diff = p - t
        abs_diff = np.abs(diff)
        quad = 0.5 * diff**2
        lin = delta * (abs_diff - 0.5 * delta)
        return np.where(abs_diff <= delta, quad, lin).mean()

    def huber_tg(x):
        t = Tensor(target_np, requires_grad=False)
        loss = huber_loss(x, t)
        loss.backward()
        return x.grad

    check_grad("Huber Loss",
               lambda x: huber_loss(x, Tensor(target_np)),
               lambda p: huber_np(p),
               pred_np, atol=1e-3)

    # --- Focal Loss (for class imbalance) ---
    # focal(p, y) = -alpha * (1-p)^gamma * log(p)  for positive class
    def focal_loss(logits: Tensor, targets: Tensor, gamma=2.0, alpha=0.25) -> Tensor:
        probs = logits.sigmoid()
        ce = -targets * probs.log() - (1 - targets) * (1 - probs).log()
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t).pow(gamma)
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        return (alpha_t * focal_weight * ce).mean()

    logits = Tensor.randn(64, 1)
    labels = Tensor((np.random.rand(64, 1) > 0.5).astype(np.float32))
    fl = focal_loss(logits, labels)
    print(f"  Focal Loss:  {fl.item():.4f}")


# ---------------------------------------------------------------------------
# Task 5: Fused vs unfused kernel comparison
# ---------------------------------------------------------------------------
def task5_fusion_impact():
    print("\n=== Task 5: Fusion Impact on Performance ===")
    import time
    from tinygrad import Tensor

    # Implement GELU in two ways:
    def gelu(x: Tensor) -> Tensor:
        c = (2.0 / np.pi) ** 0.5
        return x * 0.5 * (1.0 + (c * (x + 0.044715 * x.pow(3))).tanh())

    N = 1024 * 1024
    x = Tensor.randn(N)

    # Fused: one realize at the end
    REPS = 10
    t0 = time.perf_counter()
    for _ in range(REPS):
        gelu(x).realize()
    fused_ms = (time.perf_counter() - t0) * 1000 / REPS

    # Unfused: realize after each intermediate op
    def gelu_unfused(x: Tensor) -> Tensor:
        c = (2.0 / np.pi) ** 0.5
        x3 = x.pow(3).realize()
        inner = x.realize() + (0.044715 * x3).realize()
        tanh_arg = (c * inner.realize()).realize()
        tanh_val = tanh_arg.tanh().realize()
        return (x * 0.5 * (1.0 + tanh_val)).realize()

    t0 = time.perf_counter()
    for _ in range(REPS):
        gelu_unfused(Tensor.randn(N))
    unfused_ms = (time.perf_counter() - t0) * 1000 / REPS

    print(f"  GELU on {N//1024}K floats:")
    print(f"  Fused:   {fused_ms:.2f}ms")
    print(f"  Unfused: {unfused_ms:.2f}ms")
    print(f"  Speedup: {unfused_ms/fused_ms:.1f}x")
    print(f"\n  Run with DEBUG=1 to confirm kernel counts:")
    print(f"  Fused should have 1 kernel, unfused should have ~5.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    task1_activations()
    task2_normalization()
    task3_attention()
    task4_loss_functions()
    task5_fusion_impact()

    print("\n" + "="*60)
    print("EXERCISES:")
    print("  1. Implement Mamba SSM selective scan from primitives")
    print("  2. Implement rotary positional embeddings (RoPE)")
    print("  3. Build grouped-query attention (GQA) for LLM inference")
    print("  4. Add a custom loss: contrastive loss for embeddings")

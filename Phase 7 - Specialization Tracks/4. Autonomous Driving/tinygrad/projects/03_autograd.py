"""
Project 3: Autograd from Scratch
=================================
Goal: Understand how tinygrad implements reverse-mode autodiff.
Verify gradients analytically and numerically.

Run with:
  python3 03_autograd.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Task 1: Verify basic gradients analytically
# ---------------------------------------------------------------------------
def task1_basic_gradients():
    print("\n=== Task 1: Analytical Gradient Verification ===")
    from tinygrad import Tensor

    # --- y = sum(x^2),  dy/dx = 2x ---
    x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    expected = np.array([4.0, 6.0, 8.0])
    actual = x.grad.numpy()
    err = np.abs(actual - expected).max()
    print(f"y = sum(x^2):   dy/dx = 2x")
    print(f"  expected: {expected}")
    print(f"  got:      {actual}")
    print(f"  error:    {err:.2e}  {'✓' if err < 1e-5 else '✗'}")

    # --- y = sum(a * b),  dy/da = b,  dy/db = a ---
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    y2 = (a * b).sum()
    y2.backward()
    print(f"\ny = sum(a*b):   dy/da = b,  dy/db = a")
    print(f"  dy/da expected: {b.numpy()}  got: {a.grad.numpy()}")
    print(f"  dy/db expected: {a.numpy()}  got: {b.grad.numpy()}")

    # --- y = sigmoid(x) = 1/(1+exp(-x)),  dy/dx = sigmoid*(1-sigmoid) ---
    x3 = Tensor([0.0, 1.0, -1.0, 2.0], requires_grad=True)
    y3 = x3.sigmoid().sum()
    y3.backward()
    sig = 1 / (1 + np.exp(-x3.detach().numpy()))
    expected3 = sig * (1 - sig)
    actual3 = x3.grad.numpy()
    print(f"\ny = sigmoid(x):  dy/dx = sigmoid*(1-sigmoid)")
    print(f"  expected: {np.round(expected3, 4)}")
    print(f"  got:      {np.round(actual3, 4)}")
    print(f"  error:    {np.abs(actual3-expected3).max():.2e}")


# ---------------------------------------------------------------------------
# Task 2: Numerical gradient checker (finite differences)
# ---------------------------------------------------------------------------
def numerical_gradient(fn, x_np, eps=1e-4):
    """Estimate gradient of scalar fn at x_np using central differences."""
    grad = np.zeros_like(x_np)
    for i in range(x_np.size):
        x_plus = x_np.copy()
        x_plus.flat[i] += eps
        x_minus = x_np.copy()
        x_minus.flat[i] -= eps
        grad.flat[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
    return grad


def task2_numerical_check():
    print("\n=== Task 2: Numerical Gradient Checking ===")
    from tinygrad import Tensor

    np.random.seed(7)
    x_np = np.random.randn(3, 4).astype(np.float32) * 0.5

    # Function to check: RMS norm
    def rms_norm_np(x_np):
        rms = np.sqrt(np.mean(x_np ** 2) + 1e-6)
        out = x_np / rms
        return out.sum()

    def rms_norm_tg(x_np):
        x = Tensor(x_np, requires_grad=True)
        rms = (x * x).mean().add(1e-6).sqrt()
        out = (x / rms).sum()
        out.backward()
        return x.grad.numpy()

    numerical = numerical_gradient(rms_norm_np, x_np)
    analytical = rms_norm_tg(x_np)

    max_err = np.abs(numerical - analytical).max()
    rel_err = max_err / (np.abs(numerical).mean() + 1e-8)
    print(f"RMS Norm gradient check:")
    print(f"  Max abs error: {max_err:.2e}")
    print(f"  Rel error:     {rel_err:.2e}")
    print(f"  {'✓ PASS' if rel_err < 1e-3 else '✗ FAIL'}")

    # Custom activation: Huber loss
    def huber_np(x_np, delta=1.0):
        abs_x = np.abs(x_np)
        out = np.where(abs_x <= delta, 0.5 * x_np**2, delta * (abs_x - 0.5*delta))
        return out.sum()

    def huber_tg(x_np, delta=1.0):
        x = Tensor(x_np, requires_grad=True)
        abs_x = x.abs()
        delta_t = Tensor([delta])
        out = abs_x.leaky_relu(delta).where(0.5 * x * x, delta_t * (abs_x - delta_t * 0.5))
        # Simpler direct implementation:
        x2 = Tensor(x_np, requires_grad=True)
        abs_x2 = x2.abs()
        huber = (abs_x2 < delta).where(0.5 * x2 * x2, delta * (abs_x2 - 0.5 * delta))
        huber.sum().backward()
        return x2.grad.numpy()

    numerical_h = numerical_gradient(huber_np, x_np)
    analytical_h = huber_tg(x_np)
    max_err_h = np.abs(numerical_h - analytical_h).max()
    print(f"\nHuber Loss gradient check:")
    print(f"  Max abs error: {max_err_h:.2e}")
    print(f"  {'✓ PASS' if max_err_h < 1e-3 else '✗ FAIL (check boundary handling)'}")


# ---------------------------------------------------------------------------
# Task 3: Train XOR — overfit a tiny dataset
# ---------------------------------------------------------------------------
def task3_xor():
    print("\n=== Task 3: Overfit XOR ===")
    from tinygrad import Tensor
    from tinygrad.nn.optim import Adam

    # XOR dataset
    X_np = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Y_np = np.array([[0],[1],[1],[0]], dtype=np.float32)

    X = Tensor(X_np)
    Y = Tensor(Y_np)

    # Small MLP: 2 → 8 → 8 → 1
    class MLP:
        def __init__(self):
            self.w1 = Tensor.randn(2, 8)  * 0.5
            self.b1 = Tensor.zeros(8)
            self.w2 = Tensor.randn(8, 8) * 0.5
            self.b2 = Tensor.zeros(8)
            self.w3 = Tensor.randn(8, 1) * 0.5
            self.b3 = Tensor.zeros(1)

        def __call__(self, x):
            x = (x @ self.w1 + self.b1).relu()
            x = (x @ self.w2 + self.b2).relu()
            x = (x @ self.w3 + self.b3).sigmoid()
            return x

        def parameters(self):
            return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    model = MLP()
    optim = Adam(model.parameters(), lr=0.01)

    for step in range(2000):
        optim.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        loss.backward()
        optim.step()

        if step % 400 == 0 or step == 1999:
            pred_np = pred.numpy()
            predictions = (pred_np > 0.5).astype(int)
            accuracy = (predictions == Y_np.astype(int)).mean()
            print(f"  step {step:4d}: loss={loss.item():.4f}  acc={accuracy:.2%}")

    pred_np = model(X).numpy()
    print(f"\n  Final predictions: {pred_np.flatten().round(3)}")
    print(f"  Expected (XOR):    {Y_np.flatten()}")
    print(f"  Correctly classified: {((pred_np > 0.5).astype(int) == Y_np.astype(int)).all()}")


# ---------------------------------------------------------------------------
# Task 4: Chain rule demo — manually trace gradients
# ---------------------------------------------------------------------------
def task4_chain_rule():
    print("\n=== Task 4: Chain Rule — Manual Trace ===")
    from tinygrad import Tensor

    # y = relu(w*x + b) where x=2, w=3, b=-4
    # Forward: z = 3*2 + (-4) = 2,  y = relu(2) = 2
    # dy/dw = dy/dz * dz/dw = 1 * x = 2
    # dy/dx = dy/dz * dz/dx = 1 * w = 3
    # dy/db = dy/dz * dz/db = 1 * 1 = 1
    x_val, w_val, b_val = 2.0, 3.0, -4.0

    x = Tensor([x_val], requires_grad=True)
    w = Tensor([w_val], requires_grad=True)
    b = Tensor([b_val], requires_grad=True)

    z = w * x + b
    y = z.relu()
    y.backward()

    print(f"  Forward: z = w*x + b = {w_val}*{x_val} + {b_val} = {z.item():.1f}")
    print(f"  Forward: y = relu(z) = relu({z.item():.1f}) = {y.item():.1f}")
    print(f"\n  dy/dw (expected {x_val}):   got {w.grad.item():.1f}  {'✓' if abs(w.grad.item() - x_val) < 1e-5 else '✗'}")
    print(f"  dy/dx (expected {w_val}):   got {x.grad.item():.1f}  {'✓' if abs(x.grad.item() - w_val) < 1e-5 else '✗'}")
    print(f"  dy/db (expected 1.0): got {b.grad.item():.1f}  {'✓' if abs(b.grad.item() - 1.0) < 1e-5 else '✗'}")

    # Now set x such that relu output is 0 (z < 0)
    x2 = Tensor([-3.0], requires_grad=True)
    w2 = Tensor([1.0], requires_grad=True)
    z2 = w2 * x2
    y2 = z2.relu()
    y2.backward()
    print(f"\n  When relu is saturated (z={z2.item():.1f} < 0):")
    print(f"  dy/dw = 0 (dead relu): got {w2.grad.item():.1f}  {'✓' if abs(w2.grad.item()) < 1e-5 else '✗'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    task1_basic_gradients()
    task2_numerical_check()
    task3_xor()
    task4_chain_rule()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("  1. .backward() computes gradients via reverse-mode autodiff")
    print("  2. Each op has a backward() function that propagates grad")
    print("  3. Numerical gradients (finite diff) can verify any gradient")
    print("  4. Dead ReLUs have zero gradient — a real training problem")

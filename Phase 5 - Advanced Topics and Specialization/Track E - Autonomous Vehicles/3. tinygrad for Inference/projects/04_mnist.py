"""
Project 4: Training MNIST
==========================
Goal: Train a real CNN on MNIST end-to-end. Understand the full training loop,
evaluation, model saving/loading, and profiling.

Run with:
  python3 04_mnist.py
  DEBUG=1 python3 04_mnist.py         # see kernel counts per step
  BEAM=2 python3 04_mnist.py          # enable BEAM kernel optimization
"""

import os
import time
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mnist():
    """Download and load MNIST. Returns numpy arrays."""
    try:
        # tinygrad's built-in MNIST loader
        from tinygrad.nn.datasets import mnist
        X_train, Y_train, X_test, Y_test = mnist()
        # Normalize to [0, 1]
        X_train = X_train.numpy().astype(np.float32) / 255.0
        X_test = X_test.numpy().astype(np.float32) / 255.0
        Y_train = Y_train.numpy().astype(np.int32)
        Y_test = Y_test.numpy().astype(np.int32)
    except Exception:
        # Fallback: download manually via urllib
        import urllib.request, gzip, struct

        def download_and_parse(url, is_image):
            path = "/tmp/" + url.split("/")[-1]
            if not os.path.exists(path):
                print(f"  Downloading {url} ...")
                urllib.request.urlretrieve(url, path)
            with gzip.open(path, 'rb') as f:
                magic, n = struct.unpack(">II", f.read(8))
                if is_image:
                    rows, cols = struct.unpack(">II", f.read(8))
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                    data = data.reshape(n, rows, cols).astype(np.float32) / 255.0
                else:
                    data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
            return data

        base = "http://yann.lecun.com/exdb/mnist/"
        X_train = download_and_parse(base + "train-images-idx3-ubyte.gz", True)
        Y_train = download_and_parse(base + "train-labels-idx1-ubyte.gz", False)
        X_test = download_and_parse(base + "t10k-images-idx3-ubyte.gz", True)
        Y_test = download_and_parse(base + "t10k-labels-idx1-ubyte.gz", False)

    print(f"MNIST loaded: train={X_train.shape}, test={X_test.shape}")
    return X_train, Y_train, X_test, Y_test


# ---------------------------------------------------------------------------
# Model definition: simple CNN
# ---------------------------------------------------------------------------
class ConvBlock:
    def __init__(self, in_ch, out_ch, kernel_size=3):
        from tinygrad.nn import Conv2d, BatchNorm
        self.conv = Conv2d(in_ch, out_ch, kernel_size, padding=1)
        self.bn = BatchNorm(out_ch)

    def __call__(self, x):
        return self.bn(self.conv(x)).relu()


class MnistCNN:
    def __init__(self):
        from tinygrad.nn import Linear
        # Input: (N, 1, 28, 28)
        self.c1 = ConvBlock(1, 32, 3)      # → (N, 32, 28, 28)
        self.c2 = ConvBlock(32, 64, 3)     # → (N, 64, 28, 28)
        # After 2x2 max-pool: → (N, 64, 14, 14)
        self.c3 = ConvBlock(64, 128, 3)    # → (N, 128, 14, 14)
        # After 2x2 max-pool: → (N, 128, 7, 7)
        self.fc1 = Linear(128 * 7 * 7, 256)
        self.fc2 = Linear(256, 10)

    def __call__(self, x):
        from tinygrad import Tensor
        x = self.c1(x)
        x = self.c2(x)
        x = x.max_pool2d(2)       # (N, 64, 14, 14)
        x = self.c3(x)
        x = x.max_pool2d(2)       # (N, 128, 7, 7)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

    def parameters(self):
        from tinygrad.nn.state import get_parameters
        return get_parameters(self)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, X_train, Y_train, X_test, Y_test, epochs=5, batch_size=128):
    from tinygrad import Tensor
    from tinygrad.nn.optim import Adam
    from tinygrad.nn.state import get_state_dict, safe_save

    optim = Adam(model.parameters(), lr=3e-4)
    n_train = len(X_train)

    for epoch in range(epochs):
        model_train_mode(model)

        # Shuffle
        perm = np.random.permutation(n_train)
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        total_loss = 0.0
        n_correct = 0
        n_batches = 0
        t_epoch = time.perf_counter()

        for i in range(0, n_train, batch_size):
            xb = Tensor(X_shuf[i:i+batch_size].reshape(-1, 1, 28, 28))
            yb = Y_shuf[i:i+batch_size]

            optim.zero_grad()
            logits = model(xb)
            # Cross-entropy loss
            loss = logits.sparse_categorical_crossentropy(Tensor(yb))
            loss.backward()
            optim.step()

            loss_val = loss.item()
            total_loss += loss_val
            preds = logits.numpy().argmax(axis=1)
            n_correct += (preds == yb).sum()
            n_batches += 1

        train_acc = n_correct / n_train
        epoch_time = time.perf_counter() - t_epoch
        test_acc = evaluate(model, X_test, Y_test)

        print(f"Epoch {epoch+1}/{epochs}  "
              f"loss={total_loss/n_batches:.4f}  "
              f"train_acc={train_acc:.2%}  "
              f"test_acc={test_acc:.2%}  "
              f"time={epoch_time:.1f}s")

    # Save model
    save_path = "/tmp/mnist_model.safetensors"
    safe_save(get_state_dict(model), save_path)
    print(f"\nModel saved to {save_path}")
    return save_path


def model_train_mode(model):
    """Enable BN training mode."""
    for name, module in model.__dict__.items():
        if hasattr(module, 'training'):
            module.training = True


def evaluate(model, X_test, Y_test, batch_size=256):
    """Evaluate accuracy on test set."""
    from tinygrad import Tensor

    # Disable BN training mode
    for name, module in model.__dict__.items():
        if hasattr(module, 'training'):
            module.training = False

    n_correct = 0
    for i in range(0, len(X_test), batch_size):
        xb = Tensor(X_test[i:i+batch_size].reshape(-1, 1, 28, 28))
        yb = Y_test[i:i+batch_size]
        preds = model(xb).numpy().argmax(axis=1)
        n_correct += (preds == yb).sum()
    return n_correct / len(Y_test)


# ---------------------------------------------------------------------------
# Model save/load verification
# ---------------------------------------------------------------------------
def verify_save_load(X_test, Y_test):
    from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load

    save_path = "/tmp/mnist_model.safetensors"
    if not os.path.exists(save_path):
        print("No saved model found. Train first.")
        return

    # Load fresh model and restore weights
    model2 = MnistCNN()
    state = safe_load(save_path)
    load_state_dict(model2, state)

    acc = evaluate(model2, X_test, Y_test)
    print(f"\nReloaded model test accuracy: {acc:.2%}")
    print("✓ Save/load verification passed" if acc > 0.9 else "✗ Accuracy too low after reload")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Project 4: MNIST Training ===\n")

    X_train, Y_train, X_test, Y_test = load_mnist()

    model = MnistCNN()
    n_params = sum(p.numpy().size for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    # Quick sanity check: forward pass on one batch
    from tinygrad import Tensor
    xb = Tensor(X_train[:4].reshape(-1, 1, 28, 28))
    logits = model(xb)
    print(f"Sanity check — forward pass output shape: {logits.shape}")

    # Train
    print(f"\nTraining for 5 epochs...")
    print("(Run with DEBUG=1 to see kernel counts per step)\n")
    save_path = train(model, X_train, Y_train, X_test, Y_test, epochs=5)

    # Verify save/load
    verify_save_load(X_test, Y_test)

    print("\n" + "="*60)
    print("EXERCISES:")
    print("  1. Run with BEAM=2 — does it speed up training?")
    print("  2. Run with DEBUG=1 — how many kernels per batch?")
    print("  3. Replace Adam with SGD — how does convergence differ?")
    print("  4. Remove BatchNorm — how does accuracy change?")
    print("  5. Increase batch size to 512 — effect on speed vs. accuracy?")

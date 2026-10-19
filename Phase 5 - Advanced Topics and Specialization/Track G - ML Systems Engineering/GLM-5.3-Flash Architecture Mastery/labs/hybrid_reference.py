"""Small mathematical references, not a GLM checkpoint implementation.

Matrices are lists of rows; Python floats provide the CPU FP64 reference.
KDA inputs have already undergone any model-specific normalization/scaling.
"""
from dataclasses import dataclass
import copy
import json
import math
from pathlib import Path


def _finite_vector(vector, name):
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError(f"{name} must be a nonempty vector")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
        raise ValueError(f"{name} entries must be finite numbers")
    return len(vector)


def _matrix_shape(matrix, name):
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise ValueError(f"{name} must be a nonempty rectangular matrix")
    widths = [_finite_vector(row, name) for row in matrix]
    if any(width != widths[0] for width in widths):
        raise ValueError(f"{name} must be rectangular; ragged rows are invalid")
    return len(matrix), widths[0]


def _write_inputs(key, value, decay, beta):
    dk, dv = _finite_vector(key, "key"), _finite_vector(value, "value")
    if _finite_vector(decay, "decay") != dk:
        raise ValueError("key and decay dimensions must match")
    if type(beta) not in (int, float) or not math.isfinite(beta):
        raise ValueError("write gate must be finite")
    if not 0 <= beta <= 1 or any(not 0 <= a <= 1 for a in decay):
        raise ValueError("this fixture uses decay and write gates in [0, 1]")
    return dk, dv


def transpose(matrix):
    _matrix_shape(matrix, "matrix")
    return [list(column) for column in zip(*matrix)]


def matmul(left, right):
    _, inner = _matrix_shape(left, "left matrix")
    rows, _ = _matrix_shape(right, "right matrix")
    if inner != rows:
        raise ValueError("incompatible nonempty matrix dimensions")
    result = [[sum(a * b for a, b in zip(row, column))
               for column in transpose(right)] for row in left]
    _matrix_shape(result, "matrix product")
    return result


def add(left, right):
    if _matrix_shape(left, "left matrix") != _matrix_shape(right, "right matrix"):
        raise ValueError("matrix addition requires identical dimensions")
    result = [[a + b for a, b in zip(x, y)] for x, y in zip(left, right)]
    _matrix_shape(result, "matrix sum")
    return result


def kda_step(state, query, key, value, decay, beta):
    """Decay, predict, correct, write, then read one head.

    decay contains factors exp(g), not log-decays g. beta is a scalar.
    """
    dk, dv = _write_inputs(key, value, decay, beta)
    if _finite_vector(query, "query") != dk:
        raise ValueError("query and key dimensions must match")
    if _matrix_shape(state, "state") != (dk, dv):
        raise ValueError("state must have shape [key dimension, value dimension]")
    forgotten = [[decay[i] * state[i][j] for j in range(dv)] for i in range(dk)]
    prediction = [sum(key[i] * forgotten[i][j] for i in range(dk)) for j in range(dv)]
    error = [value[j] - prediction[j] for j in range(dv)]
    updated = [[forgotten[i][j] + beta * key[i] * error[j]
                for j in range(dv)] for i in range(dk)]
    output = [sum(query[i] * updated[i][j] for i in range(dk)) for j in range(dv)]
    _matrix_shape(updated, "updated state")
    _finite_vector(output, "output")
    return output, updated


def affine_update(key, value, decay, beta):
    """Return A,B for S' = A S + B; columns of A receive the decay."""
    size, _ = _write_inputs(key, value, decay, beta)
    a = [[((1.0 if i == j else 0.0) - beta * key[i] * key[j]) * decay[j]
          for j in range(size)] for i in range(size)]
    b = [[beta * k * v for v in value] for k in key]
    return a, b


def compose(after, before):
    """Affine composition is associative, but reversing order changes history."""
    a2, b2 = after
    a1, b1 = before
    return matmul(a2, a1), add(matmul(a2, b1), b2)


def apply_affine(update, state):
    a, b = update
    return add(matmul(a, state), b)


def softmax(scores):
    _finite_vector(scores, "scores")
    peak = max(scores)
    values = [math.exp(x - peak) for x in scores]
    total = sum(values)
    return [x / total for x in values]


def _mla_inputs(latents, key_up, value_up, query, scale):
    _, latent_dim = _matrix_shape(latents, "latents")
    key_dim, key_latent_dim = _matrix_shape(key_up, "key projection")
    _, value_latent_dim = _matrix_shape(value_up, "value projection")
    if latent_dim != key_latent_dim or latent_dim != value_latent_dim:
        raise ValueError("projection input dimensions must equal the latent dimension")
    if _finite_vector(query, "query") != key_dim:
        raise ValueError("query dimension must equal the expanded key dimension")
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("attention scale must be positive and finite")


def mla_expanded(latents, key_up, value_up, query, scale=1.0):
    """Single-head attention after materializing every key and value."""
    _mla_inputs(latents, key_up, value_up, query, scale)
    keys = matmul(latents, transpose(key_up))
    values = matmul(latents, transpose(value_up))
    probabilities = softmax([scale * sum(q * k for q, k in zip(query, row)) for row in keys])
    return matmul([probabilities], values)[0]


def mla_absorbed(latents, key_up, value_up, query, scale=1.0):
    """Move key projection into query; value projection after weighted reduction."""
    _mla_inputs(latents, key_up, value_up, query, scale)
    latent_query = matmul([query], key_up)[0]
    probabilities = softmax([scale * sum(q * c for q, c in zip(latent_query, row))
                             for row in latents])
    reduced_latent = matmul([probabilities], latents)
    return matmul(reduced_latent, transpose(value_up))[0]


def sinkhorn(matrix, rounds=20, epsilon=0.0):
    """Generic positive-matrix illustration; not GLM's complete mHC module."""
    rows, columns = _matrix_shape(matrix, "mixing matrix")
    if type(rounds) is not int or rounds < 1 or rows != columns:
        raise ValueError("use a nonempty square matrix and positive round count")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon < 0 or any(x <= 0 for row in matrix for x in row):
        raise ValueError("positive finite entries and nonnegative epsilon required")
    result = copy.deepcopy(matrix)
    for _ in range(rounds):
        denominators = [sum(row) + epsilon for row in result]
        if any(not math.isfinite(x) or x <= 0 for x in denominators):
            raise ValueError("row normalization overflow or zero denominator")
        result = [[x / denominators[i] for x in row] for i, row in enumerate(result)]
        columns = [sum(column) + epsilon for column in zip(*result)]
        if any(not math.isfinite(x) or x <= 0 for x in columns):
            raise ValueError("column normalization overflow or zero denominator")
        result = [[x / columns[j] for j, x in enumerate(row)] for row in result]
    return result


@dataclass
class HybridState:
    """Serving fixture: all fields describe one accepted prefix, and roll back together."""
    recurrent: list
    convolution: list
    latent_history: list
    prefix_length: int = 0

    def snapshot(self):
        return copy.deepcopy(self)

    def advance(self, scalar):
        # A deliberately tiny causal convolution -> KDA + latent-cache fixture.
        # These dimensions, weights and output mixing are invented for the lab.
        filtered = scalar + sum((i + 1) * x / 10 for i, x in enumerate(self.convolution))
        output, self.recurrent = kda_step(
            self.recurrent, [0.6, 0.8], [0.8, 0.6], [filtered], [0.9, 0.7], 0.4)
        self.convolution = (self.convolution + [scalar])[-3:]
        self.latent_history.append([scalar, filtered])
        self.prefix_length += 1
        return output[0] + sum(row[0] for row in self.latent_history) / self.prefix_length


def memory_ledger(context=131072, requests=1, tensor_parallel=8):
    """Lower-bound teaching ledger: FP32 KDA shards, BF16 MLA latents replicate."""
    if any(type(x) is not int or x < 1 for x in (context, requests, tensor_parallel)):
        raise ValueError("positive integer context, request count and TP degree required")
    config = json.loads(Path(__file__).with_name("checkpoint-config.json").read_text())["text_config"]
    linear = config["linear_attn_config"]
    if linear["num_heads"] % tensor_parallel:
        raise ValueError("this simple ledger requires an even KDA head partition")
    kda = len(linear["kda_layers"]) * linear["num_heads"] * linear["head_dim"] ** 2 * 4
    mla_per_token = len(linear["full_attn_layers"]) * config["kv_lora_rank"] * 2
    return {
        "context_tokens_per_request": context,
        "concurrent_requests": requests,
        "kda_all_heads_bytes_per_request": kda,
        "kda_bytes_per_gpu": requests * kda // tensor_parallel,
        "mla_bytes_per_gpu_if_replicated": requests * context * mla_per_token,
        "excluded": ["weights", "convolution state", "indexer state", "rollback copies", "workspace"],
    }


if __name__ == "__main__":
    output, state = kda_step([[2., 0.], [0., 4.]], [1., 0.], [1., 0.], [6., 1.], [1., 1.], 0.5)
    print(json.dumps({"worked_state": state, "worked_output": output,
                      "one_request_128k": memory_ledger(),
                      "two_requests_1m": memory_ledger(1048576, 2)}, indent=2))

"""Explicit decoder arithmetic and storage estimates, not device measurements."""
import json
from pathlib import Path

# (ggml enum, weights/block, bytes/block), reviewed from source 2026-10-18.
FORMATS = {"Q4_0": (2, 32, 18), "Q8_0": (8, 32, 34),
           "Q4_K": (12, 256, 144), "Q5_K": (13, 256, 176),
           "Q6_K": (14, 256, 210)}


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def quantized_matrix_bytes(rows, columns, fmt):
    positive_integer(rows, "rows")
    positive_integer(columns, "columns")
    _, block, size = FORMATS[fmt]
    if columns % block:
        raise ValueError("columns must be block-aligned; choose an explicit padding layout otherwise")
    return rows * (columns // block) * size


def projection_shapes(config):
    d, h, kv, head, ffn = [positive_integer(config[k], k) for k in
                           ("hidden_size", "num_attention_heads", "num_key_value_heads",
                            "head_dim", "intermediate_size")]
    if h % kv:
        raise ValueError("query heads must be divisible by KV heads")
    # Stored linear matrix shapes: [output, input]. h*head need not equal d.
    return {"q": (h * head, d), "k": (kv * head, d), "v": (kv * head, d),
            "o": (d, h * head), "gate": (ffn, d), "up": (ffn, d), "down": (d, ffn)}


def kv_bytes(config, context, batch=1, bytes_per_element=2):
    for name, value in (("context", context), ("batch", batch), ("bytes_per_element", bytes_per_element)):
        positive_integer(value, name)
    return 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * context * batch * bytes_per_element


def linear_intensity(m, k, batch, weight_bytes=2.0, activation_bytes=2):
    """Ideal one-read/one-write dense linear layer, excluding cache/block overhead."""
    for name, value in (("m", m), ("k", k), ("batch", batch)):
        positive_integer(value, name)
    if weight_bytes <= 0 or activation_bytes <= 0:
        raise ValueError("element sizes must be positive")
    flops = 2 * batch * m * k
    traffic = m * k * weight_bytes + batch * (k + m) * activation_bytes
    return flops / traffic


def matvec(weight, vector):
    if not weight or any(len(row) != len(vector) for row in weight):
        raise ValueError("nonempty rectangular weight must match the input dimension")
    return [sum(a * b for a, b in zip(row, vector)) for row in weight]


def report(config):
    shapes = projection_shapes(config)
    layers = config["num_hidden_layers"]
    elements = sum(m * k for m, k in shapes.values()) * layers
    embedding = config["vocab_size"] * config["hidden_size"]
    # Specific Qwen3 architecture: two d-wide norms and two shared head_dim
    # Q/K RMSNorm weights per layer, then one d-wide final norm. No biases.
    norms = layers * (2 * config["hidden_size"] + 2 * config["head_dim"]) + config["hidden_size"]
    matrix_bytes = sum(quantized_matrix_bytes(m, k, "Q4_K") for m, k in shapes.values()) * layers
    # Illustrative explicit mixture: all projection matrices Q4_K, tied table Q6_K.
    table_bytes = quantized_matrix_bytes(config["vocab_size"], config["hidden_size"], "Q6_K")
    weight_bytes = matrix_bytes + table_bytes + norms * 4
    cache = kv_bytes(config, 4096)
    bandwidth = 50e9  # hypothetical useful bandwidth; not a Jetson specification
    return {"scope": "Qwen3 config arithmetic; no weights loaded or hardware inference executed",
            "shapes_output_by_input": shapes, "unique_parameters_tied": elements + embedding + norms,
            "unique_parameters_if_untied": elements + 2 * embedding + norms,
            "lm_head_macs_per_decode_step": embedding,
            "formats": {name: {"enum": spec[0], "block": spec[1], "block_bytes": spec[2],
                                "bits_per_weight": 8 * spec[2] / spec[1]}
                        for name, spec in FORMATS.items()},
            "kv_fp16_4096_bytes": cache, "illustrative_mixed_weights_bytes": weight_bytes,
            "kv_fp16_max_config_context_bytes": kv_bytes(config, config["max_position_embeddings"]),
            "hypothetical_bandwidth_GB_per_s": 50,
            "optimistic_weight_only_tokens_per_s": bandwidth / weight_bytes,
            "optimistic_weights_plus_full_kv_read_tokens_per_s": bandwidth / (weight_bytes + cache),
            "fp16_linear_flops_per_byte_batch1": linear_intensity(4096, 2560, 1),
            "fp16_linear_flops_per_byte_batch32": linear_intensity(4096, 2560, 32)}


if __name__ == "__main__":
    config = json.loads(Path(__file__).with_name("qwen3-config.json").read_text())
    print(json.dumps(report(config), indent=2))

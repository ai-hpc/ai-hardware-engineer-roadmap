"""Metadata-only full models plus actual tiny random Qwen3 CPU inference.

No model weights, tokenizer, network or GPU required. Optional pinned dependencies.
"""
import hashlib
import inspect
import json
from pathlib import Path


def run():
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(123)
    torch.set_num_threads(1)
    here = Path(__file__).parent
    metadata = {}
    for name, config_class, model_class in (("qwen3", Qwen3Config, Qwen3ForCausalLM),
                                            ("qwen25", Qwen2Config, Qwen2ForCausalLM)):
        config = config_class.from_dict(json.loads((here / f"{name}-config.json").read_text()))
        with torch.device("meta"):
            model = model_class(config)
        metadata[name] = {"unique_parameters": sum(p.numel() for p in model.parameters()),
                          "named_parameters": len(list(model.named_parameters())),
                          "q_weight_shape": list(model.model.layers[0].self_attn.q_proj.weight.shape),
                          "tied_head": model.lm_head.weight is model.model.embed_tokens.weight,
                          "storage_device": str(next(model.parameters()).device)}
    assert metadata["qwen3"]["unique_parameters"] == 4022468096
    assert metadata["qwen25"]["unique_parameters"] == 72706203648

    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 7, 3, 9, 11]])
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        prefix = model(ids[:, :4], use_cache=True)
        cached = model(ids[:, 4:], past_key_values=prefix.past_key_values, use_cache=True)
        changed = model(torch.tensor([[1, 7, 3, 99, 100]]), use_cache=False).logits
        # New independent cache: the previous cached call mutated its own cache.
        bad_prefix = model(ids[:, :4], use_cache=True)
        wrong = model(ids[:, 4:], past_key_values=bad_prefix.past_key_values,
                      position_ids=torch.zeros((1, 1), dtype=torch.long), use_cache=True).logits
    cache_error = (full[:, -1] - cached.logits[:, -1]).abs().max().item()
    causal_error = (full[:, :3] - changed[:, :3]).abs().max().item()
    wrong_position_error = (full[:, -1] - wrong[:, -1]).abs().max().item()
    assert cache_error < 2e-6
    assert causal_error < 2e-6
    assert wrong_position_error > 1e-4
    assert cached.past_key_values.get_seq_length() == 5
    source = Path(inspect.getfile(Qwen3ForCausalLM)).read_bytes()
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "full_model_scope": "meta-device shape/parameter inspection only; no checkpoint weights",
            "full_model_metadata": metadata,
            "inference_scope": "random two-layer tiny Qwen3, FP32 CPU eager attention, seed123",
            "cached_vs_full_max_abs": cache_error,
            "changed_future_vs_earlier_logits_max_abs": causal_error,
            "deliberate_wrong_position_max_abs": wrong_position_error,
            "cache_sequence_length": cached.past_key_values.get_seq_length(),
            "qwen3_implementation_sha256": hashlib.sha256(source).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

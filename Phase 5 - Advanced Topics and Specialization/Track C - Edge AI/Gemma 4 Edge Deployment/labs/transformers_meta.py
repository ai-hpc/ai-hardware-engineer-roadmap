"""Optional full Gemma E2B architecture inspection; no pretrained tensors or inference."""
import hashlib
import inspect
import json
from pathlib import Path


def run():
    import torch
    import transformers
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    config=Gemma4Config.from_dict(json.loads(Path(__file__).with_name("target-config.json").read_text()))
    with torch.device("meta"):
        model=Gemma4ForConditionalGeneration(config)
    layers=model.model.language_model.layers
    total=sum(p.numel() for p in model.parameters())
    owners=[i for i,l in enumerate(layers) if not l.self_attn.is_kv_shared_layer]
    providers=[i for i,l in enumerate(layers) if l.self_attn.store_full_length_kv]
    assert total == 5104297504
    assert owners == list(range(15))
    assert providers == [13,14]
    return {"scope":"meta-device parameter/implementation inspection only; no model or cache tensors allocated",
            "torch":torch.__version__,"transformers":transformers.__version__,
            "unique_parameters":total,
            "component_parameters":{name:sum(p.numel() for p in module.parameters())
                                    for name,module in model.model.named_children()},
            "kv_owner_layers":owners,"shared_state_provider_layers":providers,
            "implementation_sha256":hashlib.sha256(Path(inspect.getfile(Gemma4ForConditionalGeneration)).read_bytes()).hexdigest()}


if __name__ == "__main__":
    print(json.dumps(run(),indent=2))

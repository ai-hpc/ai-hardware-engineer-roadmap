"""Optional Sionna 2.2.0/PyTorch CPU check; no RF, network calls, model downloads."""
import json
import math


def run():
    import sionna
    import torch
    from sionna.phy import config
    from sionna.phy.channel import AWGN
    from sionna.phy.mapping import Demapper, Mapper

    config.seed = 42
    torch.manual_seed(42)
    torch.set_num_threads(1)
    bits_per_symbol, ebn0_db, count = 2, 2.0, 100000
    bits = torch.randint(0, 2, (1, count), device="cpu").float()
    mapper = Mapper("qam", bits_per_symbol, device="cpu")
    demapper = Demapper("app", "qam", bits_per_symbol, device="cpu")
    symbols = mapper(bits)
    # QPSK Es=1; Eb=Es/2. Sionna AWGN no is COMPLEX variance N0,
    # so each real component has variance N0/2.
    no = torch.tensor(1 / (bits_per_symbol * 10 ** (ebn0_db / 10)))
    noiseless = demapper(symbols, no)
    assert torch.equal((noiseless > 0).float(), bits), "mapper/LLR convention mismatch"
    received = AWGN(device="cpu")(symbols, no)
    llrs = demapper(received, no)
    assert symbols.shape == (1, count // 2) and llrs.shape == bits.shape
    assert received.device.type == "cpu"
    errors = int(((llrs > 0).float() != bits).sum())
    expected = .5 * math.erfc(math.sqrt(10 ** (ebn0_db / 10)))
    standard_error = math.sqrt(expected * (1 - expected) / count)
    assert abs(errors / count - expected) < 5 * standard_error
    # Verify the optional library path really backpropagates through demapping.
    differentiable = received.detach().clone().requires_grad_(True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(demapper(differentiable, no), bits)
    loss.backward()
    assert torch.isfinite(differentiable.grad).all() and differentiable.grad.abs().sum() > 0
    return {"sionna": sionna.__version__, "torch": torch.__version__, "device": "cpu",
            "experiment": "uncoded synthetic QPSK AWGN", "seed": 42,
            "bits": count, "errors": errors, "ber": errors / count,
            "analytic_ber": expected, "complex_noise_variance": no.item(),
            "mean_symbol_energy": symbols.abs().square().mean().item(),
            "bce": loss.item(), "gradient_finite_nonzero": True}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

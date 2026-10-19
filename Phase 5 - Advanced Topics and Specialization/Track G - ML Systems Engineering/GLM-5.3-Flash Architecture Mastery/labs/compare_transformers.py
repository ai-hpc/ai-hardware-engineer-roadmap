"""Execute hash-pinned upstream KDA and KPool reference functions on CPU.

This is an explicit optional command, never a network call in portable tests.
Only the named mathematical functions are extracted; hub dispatch decorators
are removed so the comparison runs the upstream CPU fallback, not a GPU kernel.
The original upstream source remains Apache-2.0 licensed.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import torch
import torch.nn.functional as F

import hybrid_reference as ref


def load_reference(source_path):
    record = json.loads(Path(__file__).with_name("sources.json").read_text())["sources"][1]
    if source_path is None:
        source = urllib.request.urlopen(record["url"], timeout=30).read()
    else:
        source = Path(source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != record["sha256"]:
        raise ValueError("upstream source hash mismatch; review the revision before updating the record")
    names = {"l2norm", "recurrent_kimi_delta_attention", "chunk_kimi_delta_attention"}
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextTextIndexer":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in {
                    "get_visible_tokens", "get_pooled_states", "append_visible_tail"
                }:
                    method.decorator_list = []
                    selected.append(method)
    names |= {"get_visible_tokens", "get_pooled_states", "append_visible_tail"}
    if {node.name for node in selected} != names:
        raise ValueError("expected upstream reference functions missing")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=selected, type_ignores=[]), record["url"], "exec"), namespace)
    return namespace, record


def compare_pooling(functions):
    """Check actual upstream pool grouping/tails with left and right padding."""
    fixture = SimpleNamespace(head_dim=2, index_kpool=4,
                              index_kpool_compress_ape=torch.zeros(4, 2))
    comparisons = 0
    for length in (3, 4, 5, 7, 8, 9):
        for left_pad in (0, 2):
            total = left_pad + length + 2
            keys = torch.arange(total * 2, dtype=torch.float32).reshape(1, total, 2)
            valid = torch.zeros(1, total, dtype=torch.bool)
            valid[:, left_pad:left_pad+length] = True
            packed = torch.cat([keys, torch.zeros_like(keys), valid[..., None].float()], dim=-1)
            pooled, indices, pool_valid = functions["get_pooled_states"](fixture, packed)
            self_expected = torch.arange(left_pad, left_pad + (length//4)*4).reshape(1, -1, 4)
            torch.testing.assert_close(indices, self_expected)
            for pool in range(length//4):
                torch.testing.assert_close(pooled[0,pool], keys[0,left_pad+pool*4:left_pad+(pool+1)*4].mean(0))
            for count in range(1, length+1):
                visible = functions["get_visible_tokens"](fixture, valid, 1, left_pad+count)
                expected = torch.zeros_like(visible)
                expected[:, :, left_pad:left_pad+count] = True
                torch.testing.assert_close(visible, expected)
                # A complete pool is eligible only once its last token is visible.
                eligible = [pool for pool in range(length//4)
                            if bool(pool_valid[0,pool]) and bool(visible[0,0,indices[0,pool,-1]])]
                if eligible != list(range(count//4)):
                    raise AssertionError("pool selected before its end became visible")
                selected = indices[:,eligible].reshape(1,1,-1)
                with_tail = functions["append_visible_tail"](fixture, selected, visible, valid)
                real = with_tail[with_tail >= 0].tolist()
                if real != list(range(left_pad, left_pad+count)):
                    raise AssertionError(f"lost or future visible token: {real}")
                if with_tail.shape[-1] != (count//4)*4+3:
                    raise AssertionError("tail capacity must include three slots, including invalid sentinels")
                comparisons += 1
    return comparisons


def compare_weighted_mixed_pooling(functions):
    """Nonuniform per-channel gates and mixed padded requests, with scalar oracles."""
    ape = torch.log(torch.tensor([[1., 4.], [2., 3.], [3., 2.], [4., 1.]]))
    fixture = SimpleNamespace(head_dim=2, index_kpool=4, index_kpool_compress_ape=ape)
    keys = torch.tensor([[[1., 10.], [2., 20.], [3., 30.], [4., 40.]]])
    gates = torch.log(torch.tensor([[[2., 1.], [1., 3.], [3., 2.], [2., 4.]]]))
    packed = torch.cat([keys, gates, torch.ones(1, 4, 1)], dim=-1)
    pooled, indices, valid = functions["get_pooled_states"](fixture, packed)
    # Products of exp(gate) and exp(APE) give weights [2,2,9,8] / [4,9,4,4].
    torch.testing.assert_close(pooled, torch.tensor([[[65/21, 500/21]]]))
    torch.testing.assert_close(indices, torch.tensor([[[0, 1, 2, 3]]]))
    if not bool(valid[0, 0]):
        raise AssertionError("complete hand-worked pool rejected")
    if torch.allclose(pooled[0, 0], keys[0].mean(0)):
        raise AssertionError("nonuniform weighted pool collapsed to a plain average")

    batch, total = 3, 12
    ranges = [(2, 9), (0, 10), (0, 0)]  # Different left pads/lengths; third request entirely padded.
    keys = torch.tensor([[[100*b+t+1., 10*(100*b+t+1.)] for t in range(total)]
                         for b in range(batch)])
    gates = torch.tensor([[[.1*(t%4+1), .2*(3-t%4)] for t in range(total)]
                          for _ in range(batch)])
    valid_keys = torch.zeros(batch, total, dtype=torch.bool)
    for b, (start, end) in enumerate(ranges):
        valid_keys[b, start:end] = True
    packed = torch.cat([keys, gates, valid_keys[..., None].float()], dim=-1)
    pooled, indices, valid = functions["get_pooled_states"](fixture, packed)
    if pooled.shape != (3, 2, 2):
        raise AssertionError("mixed-batch pool retention has wrong shape")
    for b, (start, end) in enumerate(ranges):
        for group in range(2):
            complete = start + (group+1)*4 <= end
            if bool(valid[b, group]) != complete:
                raise AssertionError("a partial/missing pool became valid")
            if not complete:
                continue
            positions = list(range(start+group*4, start+(group+1)*4))
            torch.testing.assert_close(indices[b, group], torch.tensor(positions))
            expected = []
            for channel in range(2):
                weights = [math.exp(float(gates[b, t, channel])+float(ape[j, channel]))
                           for j, t in enumerate(positions)]
                expected.append(sum(weight*float(keys[b,t,channel]) for weight,t in zip(weights,positions))
                                / sum(weights))
            torch.testing.assert_close(pooled[b, group], torch.tensor(expected))
    torch.testing.assert_close(pooled[2], torch.zeros_like(pooled[2]))

    # Multi-query continuation: the last three query positions are 9, 10 and 11.
    visible = functions["get_visible_tokens"](fixture, valid_keys, 3, total)
    selected = torch.full((3, 3, 8), -1, dtype=torch.long)
    for b, (start, end) in enumerate(ranges):
        for q, position in enumerate((9, 10, 11)):
            wanted = list(range(start, min(end, position+1)))
            expected_visibility = torch.tensor([t in wanted for t in range(total)])
            torch.testing.assert_close(visible[b,q], expected_visibility)
            complete = wanted[:len(wanted)//4*4]
            selected[b,q,:len(complete)] = torch.tensor(complete, dtype=torch.long)
    with_tail = functions["append_visible_tail"](fixture, selected, visible, valid_keys)
    for b, (start, end) in enumerate(ranges):
        for q, position in enumerate((9, 10, 11)):
            real = with_tail[b,q][with_tail[b,q] >= 0].tolist()
            if real != list(range(start, min(end, position+1))):
                raise AssertionError("mixed-batch tail lost, duplicated, or exposed a future token")
    return {"nonuniform_hand_example": 1, "mixed_batch_pool_slots": 6,
            "mixed_batch_multiquery_tails": 9, "includes_all_padding_request": True}


def compare(source_path=None):
    torch.set_num_threads(1)
    functions, record = load_reference(source_path)
    recurrent = functions["recurrent_kimi_delta_attention"]
    chunked = functions["chunk_kimi_delta_attention"]
    generator = torch.Generator().manual_seed(20261018)
    worst_output = worst_state = worst_scalar = worst_continuation = 0.0
    cases = 0
    for length in (1, 3, 4, 5, 7, 8, 9, 65):
        for normalize in (False, True):
            q = torch.randn(2, length, 2, 3, generator=generator) * .3
            k = torch.randn(2, length, 2, 3, generator=generator) * .3
            v = torch.randn(2, length, 2, 5, generator=generator)
            g = -.01 - torch.rand(2, length, 2, 3, generator=generator) * .2
            beta = torch.rand(2, length, 2, generator=generator)
            initial = torch.randn(2, 2, 3, 5, generator=generator) * .2
            expected_o, expected_s = recurrent(q, k, v, g, beta, initial, True, normalize)
            for chunk in (1, 4, 8, 64):
                output, state = chunked(q, k, v, g, beta, chunk_size=chunk,
                                        initial_state=initial, output_final_state=True,
                                        use_qk_l2norm_in_kernel=normalize)
                worst_output = max(worst_output, (output - expected_o).abs().max().item())
                worst_state = max(worst_state, (state - expected_s).abs().max().item())
                torch.testing.assert_close(output, expected_o, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(state, expected_s, rtol=2e-5, atol=2e-6)
                cases += 1
            # Independent Python FP64 recurrence, for every batch and head.
            for batch in range(2):
                for head in range(2):
                    state = initial[batch, head].tolist()
                    for t in range(length):
                        qi, ki = q[batch, t, head].tolist(), k[batch, t, head].tolist()
                        if normalize:
                            qi = [x / math.sqrt(sum(y*y for y in qi) + 1e-6) for x in qi]
                            ki = [x / math.sqrt(sum(y*y for y in ki) + 1e-6) for x in ki]
                        qi = [x / math.sqrt(3) for x in qi]
                        output, state = ref.kda_step(state, qi, ki, v[batch, t, head].tolist(),
                                                     [math.exp(x) for x in g[batch, t, head].tolist()],
                                                     beta[batch, t, head].item())
                        error = max(abs(a-b) for a,b in zip(output, expected_o[batch,t,head].tolist()))
                        worst_scalar = max(worst_scalar, error)
                        if error > 2e-6:
                            raise AssertionError(f"scalar recurrence error {error}")
                    torch.testing.assert_close(torch.tensor(state), expected_s[batch,head], rtol=2e-5, atol=2e-6)
            # A chunked prefix must carry its state into the recurrent suffix.
            if length > 1:
                cut = length // 2
                _, carried = chunked(q[:, :cut], k[:, :cut], v[:, :cut], g[:, :cut], beta[:, :cut],
                                      chunk_size=4, initial_state=initial, output_final_state=True,
                                      use_qk_l2norm_in_kernel=normalize)
                output, state = recurrent(q[:, cut:], k[:, cut:], v[:, cut:], g[:, cut:], beta[:, cut:], carried, True, normalize)
                worst_continuation = max(worst_continuation, (output - expected_o[:, cut:]).abs().max().item())
                torch.testing.assert_close(output, expected_o[:, cut:], rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(state, expected_s, rtol=2e-5, atol=2e-6)
    pool_comparisons = compare_pooling(functions)
    weighted_pooling = compare_weighted_mixed_pooling(functions)
    return {"torch_version": torch.__version__, "device": "cpu", "upstream_source": record,
            "chunk_comparisons": cases, "sequence_lengths": [1,3,4,5,7,8,9,65],
            "pool_visibility_and_tail_comparisons": pool_comparisons,
            "weighted_and_mixed_pooling": weighted_pooling,
            "chunk_sizes": [1,4,8,64], "normalization_modes": [False, True],
            "max_chunk_output_absolute_error": worst_output,
            "max_chunk_final_state_absolute_error": worst_state,
            "max_scalar_output_absolute_error": worst_scalar,
            "max_continuation_output_absolute_error": worst_continuation,
            "scope": "Six extracted upstream mathematical CPU fallback functions/methods; no model weights, full layer, CUDA/FLA kernel, packed documents, or deployment."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local copy of the pinned upstream file; otherwise download it")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = json.dumps(compare(arguments.source), indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(result)
    print(result, end="")

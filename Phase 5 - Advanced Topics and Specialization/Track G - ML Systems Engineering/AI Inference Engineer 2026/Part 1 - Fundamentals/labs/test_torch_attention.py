import importlib.util
import unittest
from inference_contracts import causal_gqa

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F


@unittest.skipUnless(HAS_TORCH, 'optional PyTorch CPU comparison')
class TorchAttention(unittest.TestCase):
    def test_gqa_chunks_match_scalar_and_sdpa_full(self):
        torch.manual_seed(9)
        q = torch.randn(1, 4, 7, 3, dtype=torch.float64)
        k = torch.randn(1, 2, 7, 3, dtype=torch.float64)
        v = torch.randn(1, 2, 7, 3, dtype=torch.float64)
        # Explicit expansion supports CPU backends without assuming GQA dispatch.
        ke, ve = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
        full = F.scaled_dot_product_attention(q, ke, ve, is_causal=True, dropout_p=0)
        scalar = causal_gqa(q[0].transpose(0, 1).tolist(),
                           k[0].transpose(0, 1).tolist(), v[0].transpose(0, 1).tolist())
        torch.testing.assert_close(full[0].transpose(0, 1), torch.tensor(scalar, dtype=q.dtype))
        for chunk in (1, 2, 3, 7):
            pieces = []
            for start in range(0, 7, chunk):
                end = min(7, start+chunk)
                mask = torch.arange(end)[None, :] <= torch.arange(start, end)[:, None]
                pieces.append(F.scaled_dot_product_attention(q[:, :, start:end],
                    ke[:, :, :end], ve[:, :, :end], attn_mask=mask, dropout_p=0))
            torch.testing.assert_close(torch.cat(pieces, dim=2), full)

    def test_non_square_is_causal_is_not_decode_offset(self):
        q = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
        k = torch.zeros((1, 1, 3, 1), dtype=torch.float64)
        v = torch.tensor([1., 3., 8.], dtype=torch.float64).reshape(1, 1, 3, 1)
        wrong = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
        correct = F.scaled_dot_product_attention(q, k, v,
                      attn_mask=torch.ones((1, 3), dtype=torch.bool), dropout_p=0)
        self.assertEqual(wrong.item(), 1)
        self.assertEqual(correct.item(), 4)


if __name__ == '__main__':
    unittest.main()

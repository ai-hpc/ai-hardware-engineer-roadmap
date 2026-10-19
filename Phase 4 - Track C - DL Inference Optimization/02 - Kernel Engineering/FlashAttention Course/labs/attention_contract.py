"""Attention shape/mask/traffic contracts using only the standard library."""
import argparse
import json


def count(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{label} must be a positive integer')
    return value


def causal_mask(q_len: int, kv_len: int, alignment: str) -> list[list[bool]]:
    count(q_len, 'q_len'); count(kv_len, 'kv_len')
    if alignment not in ('upper_left', 'lower_right'):
        raise ValueError('alignment must be explicit')
    offset = kv_len - q_len if alignment == 'lower_right' else 0
    return [[j <= i + offset for j in range(kv_len)] for i in range(q_len)]


def traffic(q_len: int, kv_len: int, dim: int, value_dim: int, row_tile: int,
            element_bytes: int = 2, stats_bytes: int = 4) -> dict:
    for label, value in locals().copy().items(): count(value, label)
    input_output = (q_len*dim + kv_len*dim + kv_len*value_dim + q_len*value_dim)*element_bytes
    materialized = input_output + 4*q_len*kv_len*element_bytes
    tiles = (q_len + row_tile - 1)//row_tile
    # Explicit query-outer traversal: Q/O once, complete K/V once per query tile.
    # Exact logical element counts for tails; no claim about hardware cache lines.
    tiled = (q_len*(dim+value_dim) + tiles*kv_len*(dim+value_dim))*element_bytes + q_len*stats_bytes
    return {'matmul_flops_full': 2*q_len*kv_len*(dim+value_dim),
            'materialized_bytes': materialized, 'query_outer_logical_bytes': tiled,
            'compulsory_input_output_bytes': input_output, 'query_tiles': tiles,
            'score_matrix_bytes': q_len*kv_len*element_bytes,
            'scope': 'one head; dense traversal; no cache, allocator, padding, tensor-core padding or dropout model'}


def kv_bytes(length: int, kv_heads: int, dim: int, *, layers: int = 1, batch: int = 1, element_bytes: int = 2) -> int:
    for label, value in locals().copy().items(): count(value, label)
    return 2*length*kv_heads*dim*layers*batch*element_bytes


def gather_pages(pages: list[list[int]], table: list[int], length: int) -> list[int]:
    count(length, 'length')
    if not pages or not table or not pages[0]: raise ValueError('empty page storage')
    size = len(pages[0])
    if any(len(page) != size for page in pages): raise ValueError('inconsistent page size')
    required = (length+size-1)//size
    if len(table) != required: raise ValueError('table must describe exactly the live pages')
    if any(isinstance(i, bool) or not isinstance(i,int) or i < 0 or i >= len(pages) for i in table):
        raise ValueError('invalid physical page index')
    return [value for i in table for value in pages[i]][:length]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--length',type=int,default=8192)
    p.add_argument('--dim',type=int,default=64);p.add_argument('--row-tile',type=int,default=64);a=p.parse_args()
    print(json.dumps({'traffic':traffic(a.length,a.length,a.dim,a.dim,a.row_tile),
                     'decode_kv_bytes_16k_8heads_d128_32layers':kv_bytes(16384,8,128,layers=32),
                     'upper_left_2x5':causal_mask(2,5,'upper_left'),
                     'lower_right_2x5':causal_mask(2,5,'lower_right')},indent=2))

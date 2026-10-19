"""Compile and exercise actual C++ components with independently encoded fixtures."""
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def string(value):
    if isinstance(value, str): value = value.encode()
    return struct.pack('<Q', len(value)) + value


def entry(key, kind, value):
    return string(key) + struct.pack('<I', kind) + value


def fixture(metadata=(), tensors=(), alignment=32, payload=None):
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata)) + b''.join(metadata)
    end = 0
    for name, dims, kind, offset, size in tensors:
        header += string(name) + struct.pack('<I', len(dims))
        header += b''.join(struct.pack('<Q', value) for value in dims)
        header += struct.pack('<IQ', kind, offset)
        end = max(end, offset+size)
    if tensors:
        header += bytes((-len(header)) % alignment)
        header += bytes(end) if payload is None else payload
    return header


def vocabulary(model='course.greedy.bytes', tokens=('<bos>', '<eos>', 'a', 'b', 'ab', '<0x58>'), add_bos=True):
    return [entry('tokenizer.ggml.model', 8, string(model)),
            entry('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, len(tokens)) + b''.join(string(s) for s in tokens)),
            entry('tokenizer.ggml.bos_token_id', 4, struct.pack('<I', 0)),
            entry('tokenizer.ggml.eos_token_id', 4, struct.pack('<I', 1)),
            entry('tokenizer.ggml.add_bos_token', 7, bytes([add_bos]))]


class RuntimeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='course-runtime-cpu-')
        cls.addClassCleanup(cls.workspace.cleanup)
        cls.root = Path(cls.workspace.name)
        supplied = os.environ.get('JLLM_CPU_BUILD')
        if supplied:
            cls.build = Path(supplied)
        else:
            if not shutil.which('cmake') or not shutil.which('c++'):
                raise unittest.SkipTest('cmake and C++17 compiler required for actual native checks')
            cls.build = cls.root/'build'
            subprocess.run(['cmake', '-S', str(ROOT), '-B', str(cls.build),
                            '-DCMAKE_BUILD_TYPE=Release', '-DJLLM_BUILD_JETSON=OFF'],
                           check=True, capture_output=True, text=True)
            subprocess.run(['cmake', '--build', str(cls.build), '-j2', '--target',
                            'gguf-inspect', 'test_tokenizer', 'test_sampling', 'test_memory_contracts', 'test_allocator_contracts'],
                           check=True, capture_output=True, text=True)
        for binary in ('gguf-inspect', 'test_tokenizer', 'test_sampling', 'test_memory_contracts', 'test_allocator_contracts'):
            if not (cls.build/binary).is_file(): raise RuntimeError(f'missing native binary {binary}')

    def inspect(self, data, valid=True, message=None):
        path = self.root/'case.gguf'; path.write_bytes(data)
        result = subprocess.run([str(self.build/'gguf-inspect'), str(path)], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0 if valid else 2, result.stderr)
        if message: self.assertIn(message, result.stderr)
        return result.stdout

    def tokenize(self, data, prompt, status=0):
        path = self.root/'tokens.gguf'; path.write_bytes(data)
        result = subprocess.run([str(self.build/'test_tokenizer'), str(path), prompt], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, status, result.stderr)
        return result.stdout.strip()

    def test_scalar_and_flat_array_metadata_with_late_fields(self):
        metadata = [entry('a', 0, b'\xff'), entry('b', 1, b'\xff'), entry('c', 2, b'\x34\x12'),
                    entry('d', 3, b'\xff\xff'), entry('e', 4, struct.pack('<I', 123)),
                    entry('f', 5, struct.pack('<i', -123)), entry('g', 6, struct.pack('<f', .25)),
                    entry('h', 7, b'\x01'), entry('i', 8, string('hello')),
                    entry('j', 9, struct.pack('<IQ', 0, 3)+b'abc'),
                    entry('k', 9, struct.pack('<IQ', 8, 2)+string('alpha')+string('β')),
                    entry('l', 10, struct.pack('<Q', 2**60)), entry('m', 11, struct.pack('<q', -5)),
                    entry('n', 12, struct.pack('<d', .125))]
        metadata += [entry(f'extra.{i}', 4, struct.pack('<I', i)) for i in range(260)]
        metadata += [entry('last.field', 8, string('retained'))]
        self.assertIn('metadata=275', self.inspect(fixture(metadata)))

    def test_every_truncated_header_and_tensor_prefix_is_rejected(self):
        data = fixture([entry('general.alignment', 4, struct.pack('<I', 64))],
                       [('tiny', [2, 3], 0, 0, 24)], 64)
        self.inspect(data)
        for i in range(len(data)):
            with self.subTest(bytes=i): self.inspect(data[:i], False)

    def test_version_magic_and_entry_count_limits(self):
        base = fixture()
        for changed in (b'NOPE'+base[4:], base[:4]+struct.pack('<I', 2)+base[8:],
                        base[:8]+struct.pack('<Q', 65537)+base[16:],
                        base[:16]+struct.pack('<Q', 100001)):
            self.inspect(changed, False)

    def test_unknown_nested_and_invalid_bool_types_rejected(self):
        for item in (entry('unknown', 99, b''), entry('nested', 9, struct.pack('<IQ', 9, 0)),
                     entry('bool', 7, b'\x02'), entry('bools', 9, struct.pack('<IQ', 7, 2)+b'\x00\x02')):
            self.inspect(fixture([item]), False)

    def test_lengths_counts_and_duplicate_keys_rejected(self):
        key = entry('name', 8, string('first'))
        for data in (fixture([key, key]), fixture([entry('large', 8, struct.pack('<Q', 2**64-1))]),
                     fixture([entry('array', 9, struct.pack('<IQ', 8, 1000001))]),
                     fixture([entry('array', 9, struct.pack('<IQ', 8, 999999))])):
            self.inspect(data, False)

    def test_alignment_and_payload_bounds(self):
        for alignment in (0, 1, 2, 3, 4, 2**21):
            self.inspect(fixture([entry('general.alignment', 4, struct.pack('<I', alignment))]), False)
        self.inspect(fixture(tensors=[('a', [1], 0, 1, 4)]), False, 'unaligned')
        self.inspect(fixture(tensors=[('a', [2], 0, 0, 8)], payload=b'1234'), False, 'outside file')

    def test_utf8_strings_and_ascii_hierarchical_keys(self):
        # Includes overlong encodings, surrogate, out-of-range scalar, truncation,
        # stray continuation, and invalid leading byte; no locale-dependent parser.
        for raw in (b'\xc0\x80', b'\xe0\x80\x80', b'\xed\xa0\x80', b'\xf4\x90\x80\x80',
                    b'\xf0\x80\x80\x80', b'\xc2', b'\x80', b'\xff'):
            with self.subTest(raw=raw):
                self.inspect(fixture([entry('text', 8, string(raw))]), False, 'UTF-8')
                self.inspect(fixture([entry('array', 9, struct.pack('<IQ', 8, 1)+string(raw))]),
                             False, 'UTF-8')
        for key in ('UPPER', 'has space', '.start', 'end.', 'two..parts', 'naïve', 'has\x00nul'):
            with self.subTest(key=key):
                self.inspect(fixture([entry(key, 4, struct.pack('<I', 1))]), False, 'key')
        self.inspect(fixture([entry('valid_key.part_2', 8, string('A\x00β🙂\U0010ffff'))]))

    def test_tensor_name_limit_is_in_utf8_bytes(self):
        for name in ('a'*64, 'β'*32):
            self.inspect(fixture(tensors=[(name, [1], 0, 0, 4)]))
        for name in ('a'*65, 'β'*33):
            self.inspect(fixture(tensors=[(name, [1], 0, 0, 4)]), False, 'string exceeds limit')

    def test_tensor_rank_zero_extent_overflow_overlap_and_duplicates(self):
        for tensors in ([('a', [], 0, 0, 4)], [('a', [1]*5, 0, 0, 4)],
                        [('a', [0], 0, 0, 4)], [('a', [2**63, 3], 0, 0, 4)],
                        [('a', [16], 0, 0, 64), ('b', [16], 0, 32, 64)],
                        [('a', [1], 0, 0, 4), ('a', [1], 0, 32, 4)]):
            self.inspect(fixture(tensors=tensors), False)

    def test_real_quantization_block_sizes_and_row_divisibility(self):
        for kind, elements, size in ((0, 3, 12), (1, 3, 6), (2, 32, 18), (3, 32, 20),
                                    (6, 32, 22), (7, 32, 24), (8, 32, 34), (9, 32, 36),
                                    (10, 256, 84), (11, 256, 110), (12, 256, 144),
                                    (13, 256, 176), (14, 256, 210), (15, 256, 292)):
            with self.subTest(kind=kind):
                output = self.inspect(fixture(tensors=[('weight', [elements, 2], kind, 0, size*2)]))
                self.assertIn(f'bytes={size*2}', output)
        self.inspect(fixture(tensors=[('weight', [128, 2], 12, 0, 144)]), False, 'row')
        self.inspect(fixture(tensors=[('weight', [256], 99, 0, 144)]), False, 'unsupported')

    def test_toy_tokenizer_greedy_bytes_bos_and_reload(self):
        self.assertEqual(self.tokenize(fixture(vocabulary()), 'abaX'), '0 4 2 5')
        self.assertEqual(self.tokenize(fixture(vocabulary(add_bos=False)), 'abaX'), '4 2 5')
        self.assertEqual(self.tokenize(fixture(vocabulary()), ''), '0')
        self.tokenize(fixture(vocabulary()), 'z', 3)
        self.tokenize(fixture(vocabulary()), '<0x58>', 3)

    def test_unsupported_reference_tokenizers_and_invalid_vocab_fail(self):
        for model in ('gpt2', 'llama', 'qwen2'):
            self.tokenize(fixture(vocabulary(model)), 'ab', 2)
        for tokens in (('<bos>', '<eos>', 'a', 'a'), ('<bos>', '<eos>', ''), ()):
            self.tokenize(fixture(vocabulary(tokens=tokens)), 'a', 2)

    def test_actual_sampler_against_probability_and_failure_contracts(self):
        result = subprocess.run([str(self.build/'test_sampling')], check=True, text=True,
                                capture_output=True, timeout=10)
        self.assertIn('sampling contract checks passed', result.stdout)

    def test_model_config_exact_keys_head_dimension_and_storage(self):
        metadata = vocabulary()
        dimensions = {'block_count': 2, 'attention.head_count': 4, 'attention.head_count_kv': 2,
                      'embedding_length': 8, 'feed_forward_length': 12, 'context_length': 128,
                      'attention.key_length': 4}
        metadata += [entry('general.architecture', 8, string('qwen3'))]
        metadata += [entry('qwen3.'+key, 4, struct.pack('<I', value)) for key, value in dimensions.items()]
        # An unrelated suffix lookalike must not override a recognized field.
        metadata += [entry('foreign.block_count', 4, struct.pack('<I', 999))]
        data = fixture(metadata, [('token_embd.weight', [8, 6], 1, 0, 96)])
        path = self.root/'config.gguf'; path.write_bytes(data)
        result = subprocess.run([str(self.build/'gguf-inspect'), '--config', str(path)],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('layers=2 heads=4 kv_heads=2 head_dim=4 hidden=8 vocab=6 tensor_bytes=96', result.stdout)

    def test_model_config_rejects_guessing_missing_dimensions_or_vocab(self):
        metadata = vocabulary()+[entry('general.architecture', 8, string('llama'))]
        base = {'block_count': 2, 'attention.head_count': 4, 'attention.head_count_kv': 2,
                'embedding_length': 8, 'feed_forward_length': 12, 'context_length': 128}
        for change in ({'block_count': 0}, {'attention.head_count_kv': 3}, {'embedding_length': 7},
                       {'context_length': None}, {'vocab_size': 99}):
            current = dict(base); current.update(change)
            items = metadata+[entry('llama.'+key, 4, struct.pack('<I', value))
                              for key, value in current.items() if value is not None]
            path = self.root/'invalid-config.gguf'
            path.write_bytes(fixture(items, [('token_embd.weight', [8, 6], 1, 0, 96)]))
            result = subprocess.run([str(self.build/'gguf-inspect'), '--config', str(path)],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_native_memory_planning_arithmetic_and_bounds(self):
        result = subprocess.run([str(self.build/'test_memory_contracts')], check=True, text=True,
                                capture_output=True, timeout=10)
        self.assertIn('memory planning checks passed', result.stdout)

    def test_actual_allocator_control_logic_with_cpu_shim(self):
        result = subprocess.run([str(self.build/'test_allocator_contracts')], check=True,
                                text=True, capture_output=True, timeout=10)
        self.assertIn('allocator CPU-shim checks passed', result.stdout)

    def test_seeded_malformed_corpus_is_bounded(self):
        rng = random.Random(19)
        for _ in range(100):
            self.inspect(rng.randbytes(rng.randrange(0, 200)), False)


if __name__ == '__main__': unittest.main(verbosity=2)

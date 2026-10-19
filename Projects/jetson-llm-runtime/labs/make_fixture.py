"""Write a tiny, deliberately incomplete GGUF v3 file for the parser lesson."""
import argparse
from pathlib import Path
import struct


def text(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def make_fixture():
    items = []
    def add(key, kind, value): items.append(text(key)+struct.pack('<I', kind)+value)
    add('general.architecture', 8, text('llama'))
    add('general.name', 8, text('course parser fixture; not a complete model'))
    for key, value in {'block_count':2, 'attention.head_count':4, 'attention.head_count_kv':2,
                       'embedding_length':8, 'feed_forward_length':12, 'context_length':128}.items():
        add('llama.'+key, 4, struct.pack('<I', value))
    add('tokenizer.ggml.model', 8, text('course.greedy.bytes'))
    tokens = ('<bos>', '<eos>', 'a', 'b', 'ab', '<0x58>')
    add('tokenizer.ggml.tokens', 9, struct.pack('<IQ',8,len(tokens))+b''.join(text(t) for t in tokens))
    add('tokenizer.ggml.bos_token_id', 4, struct.pack('<I',0))
    add('tokenizer.ggml.eos_token_id', 4, struct.pack('<I',1))
    add('tokenizer.ggml.add_bos_token', 7, b'\x01')
    header = b'GGUF'+struct.pack('<IQQ',3,1,len(items))+b''.join(items)
    header += text('token_embd.weight')+struct.pack('<IQQIQ',2,8,6,1,0)
    return header+bytes((-len(header))%32)+bytes(8*6*2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.output.open('xb') as target: target.write(make_fixture())
    print(f'Created {args.output}: {args.output.stat().st_size} bytes; no trained weights')

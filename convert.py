#!/usr/bin/env python3
"""Maple-Preview -> packed checkpoint, streamed straight from Hugging Face (no 40 GB of BF16 on disk).

Ternary matrices (experts, q/k/v/o) hold one |w| per row, so each becomes 2-bit codes t+1 in {0,1,2} plus one fp32
scale per row. Codes are packed in 128-column groups: byte j of a group holds columns j, j+32, j+64, j+96 at bits
0, 2, 4, 6 (the TQ2_0 layout from llama.cpp, so AVX2 unpacks a group with four shifts and one mask).
Dense tensors (embeddings, lm_head, router, norms) are copied as raw BF16.

Usage: python3 convert.py OUTDIR [shard numbers...]     resumable: finished shards are skipped
Out:   OUTDIR/shard-0000N.bin + shard-0000N.json, then OUTDIR/index.json once all nine are done."""
import json, os, re, struct, subprocess, sys, time
import numpy as np

REPO = 'https://huggingface.co/deepgrove/maple-preview/resolve/ac1ddd79d2/'
NSHARDS = 9
TERN = re.compile(r'\.(mlp\.experts\.\d+\.(gate|up|down)_proj|self_attn\.(q|k|v|o)_proj)\.weight$')


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def pack_ternary(w):
    """w: float32 [rows, cols] with one |w| per row -> (packed uint8 [rows, cols/4], scale float32 [rows])."""
    a = np.abs(w)
    s = a.max(axis=1)
    if not np.all((a == 0) | (a == s[:, None])):
        raise ValueError('not ternary: a row holds more than one |w|')
    codes = (np.sign(w) + 1).astype(np.uint8)  # 0, 1, 2
    r, c = w.shape
    g = codes.reshape(r, c // 128, 4, 32)
    packed = g[:, :, 0] | (g[:, :, 1] << 2) | (g[:, :, 2] << 4) | (g[:, :, 3] << 6)
    return np.ascontiguousarray(packed.reshape(r, c // 4)), s.astype(np.float32)


def unpack_ternary(packed, scale, cols):
    r = packed.shape[0]
    p = packed.reshape(r, cols // 128, 32)
    g = np.stack([(p >> sh) & 3 for sh in (0, 2, 4, 6)], axis=2)  # [r, groups, 4, 32]
    return (g.reshape(r, cols).astype(np.float32) - 1) * scale[:, None]


def read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise IOError(f'stream ended: {len(buf)} of {n} bytes')
        buf += chunk
    return bytes(buf)


def convert_shard(k, out):
    name = f'model-{k:05d}-of-{NSHARDS:05d}.safetensors'
    done = os.path.join(out, f'shard-{k:05d}.json')
    if os.path.exists(done):
        print(f'shard {k}: already done', flush=True)
        return
    binp = os.path.join(out, f'shard-{k:05d}.bin')
    t0 = time.time()
    p = subprocess.Popen(['curl', '-sL', '--fail', '--retry', '8', '--retry-delay', '5', REPO + name],
                         stdout=subprocess.PIPE, bufsize=1 << 20)
    f = p.stdout
    n = struct.unpack('<Q', read_exact(f, 8))[0]
    head = json.loads(read_exact(f, n))
    head.pop('__metadata__', None)
    pos, index, nbytes = 0, {}, 0
    with open(binp + '.part', 'wb') as o:
        for tname, t in sorted(head.items(), key=lambda kv: kv[1]['data_offsets'][0]):
            a, b = t['data_offsets']
            if a > pos:
                read_exact(f, a - pos)
            raw = read_exact(f, b - a)
            pos = b
            nbytes += b - a
            if t['dtype'] != 'BF16':
                raise ValueError(f'{tname}: dtype {t["dtype"]}')
            off = o.tell()
            if TERN.search(tname):
                w = bf16_to_f32(raw).reshape(t['shape'])
                packed, scale = pack_ternary(w)
                o.write(packed.tobytes()); o.write(scale.tobytes())
                index[tname] = {'file': os.path.basename(binp), 'off': off, 'kind': 't2', 'shape': t['shape'],
                                'soff': off + packed.nbytes}
            else:
                o.write(raw)
                index[tname] = {'file': os.path.basename(binp), 'off': off, 'kind': 'bf16', 'shape': t['shape']}
            if len(index) % 500 == 0:
                mb = nbytes / 1e6
                print(f'shard {k}: {len(index)}/{len(head)} tensors, {mb:.0f} MB in, {mb / (time.time() - t0):.1f} MB/s', flush=True)
    p.wait()
    if len(index) != len(head):
        raise IOError(f'shard {k}: {len(index)} of {len(head)} tensors')
    os.replace(binp + '.part', binp)
    json.dump(index, open(done, 'w'))
    print(f'shard {k}: done, {len(index)} tensors, {nbytes / 1e9:.2f} GB in -> {os.path.getsize(binp) / 1e9:.2f} GB out, '
          f'{time.time() - t0:.0f} s', flush=True)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    ks = [int(x) for x in sys.argv[2:]] or list(range(1, NSHARDS + 1))
    for k in ks:
        for attempt in range(4):
            try:
                convert_shard(k, out)
                break
            except Exception as e:
                print(f'shard {k}: attempt {attempt + 1} failed: {e}', flush=True)
                time.sleep(10)
        else:
            sys.exit(f'shard {k}: gave up')
    parts = [os.path.join(out, f'shard-{k:05d}.json') for k in range(1, NSHARDS + 1)]
    if all(os.path.exists(x) for x in parts):
        index = {}
        for x in parts:
            index.update(json.load(open(x)))
        json.dump(index, open(os.path.join(out, 'index.json'), 'w'))
        print(f'index: {len(index)} tensors', flush=True)
    print('FIN', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""index.json -> layout.bin: the tensor table maple.c reads, in one fixed order, 24-byte records '<BBQQII'
(kind 't' ternary / 'b' bf16, shard file 0..8, data offset, scale offset, rows, cols).
Order: word_embeddings, lm_head, norm; then per layer: input_layernorm, post_attention_layernorm, q_norm, k_norm,
router, q, k, v, o, and for each of 256 experts gate, up, down."""
import json, os, struct, sys

d = sys.argv[1]
idx = json.load(open(os.path.join(d, 'index.json')))
names = ['model.word_embeddings.weight', 'lm_head.weight', 'model.norm.weight']
for L in range(24):
    p = f'model.layers.{L}.'
    names += [p + 'input_layernorm.weight', p + 'post_attention_layernorm.weight', p + 'self_attn.q_norm.weight',
              p + 'self_attn.k_norm.weight', p + 'mlp.gate.weight'] + [p + f'self_attn.{m}_proj.weight' for m in 'qkvo']
    for e in range(256):
        names += [p + f'mlp.experts.{e}.{m}_proj.weight' for m in ('gate', 'up', 'down')]
assert len(names) == len(idx), (len(names), len(idx))
with open(os.path.join(d, 'layout.bin'), 'wb') as o:
    for n in names:
        e = idx[n]; shape = e['shape'] + [1] * (2 - len(e['shape']))
        o.write(struct.pack('<BBQQII', ord('t' if e['kind'] == 't2' else 'b'), int(e['file'][6:11]) - 1,
                            e['off'], e.get('soff', 0), shape[0], shape[1]))
print(f'layout: {len(names)} tensors')

#!/usr/bin/env python3
"""Reference forward pass of Maple-Preview in plain numpy, fp32, on the packed checkpoint from convert.py.
Slow on purpose: it is the yardstick the fast runtime is checked against, not the runtime.

Follows modeling_maple.py: pre-norm decoder, GQA 16/4 with q/k RMSNorm per head, RoPE on the first 64 of 128
dims only in sliding layers (window 512, keys i-512..i), no RoPE in global layers, router softmax in fp32 -> top 8
-> renormalised, experts silu(min(gate, 7)) * clamp(up, -7, 7).

Usage: python3 ref.py CKPTDIR TOKENIZER.json "question" [max_new_tokens]
       python3 ref.py CKPTDIR TOKENIZER.json --ids 1,2,3 --dump logits.npy   (one forward over given ids)"""
import json, os, sys, time
import numpy as np

H, NL, NH, NKV, HD, NE, TOPK, VOCAB = 2048, 24, 16, 4, 128, 256, 8, 151936
ROT, WINDOW, EPS = 64, 512, 1e-6
GLOBAL = {3, 7, 11, 15, 19, 23}
EOS = 151645


class Ckpt:
    def __init__(self, d):
        self.d, self.idx, self.mm = d, json.load(open(os.path.join(d, 'index.json'))), {}

    def _map(self, f):
        if f not in self.mm:
            self.mm[f] = np.memmap(os.path.join(self.d, f), dtype=np.uint8, mode='r')
        return self.mm[f]

    def dense(self, name, rows=None):
        e = self.idx[name]; m = self._map(e['file']); shape = e['shape']
        if rows is not None:  # one row of a 2-D tensor (embedding lookup)
            n = shape[1]; raw = m[e['off'] + rows * n * 2: e['off'] + (rows + 1) * n * 2]
            return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        n = int(np.prod(shape)); raw = m[e['off']: e['off'] + n * 2]
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)

    def tern(self, name):
        e = self.idx[name]; m = self._map(e['file']); r, c = e['shape']
        packed = np.frombuffer(m[e['off']: e['off'] + r * c // 4], dtype=np.uint8).reshape(r, c // 128, 32)
        scale = np.frombuffer(m[e['soff']: e['soff'] + r * 4], dtype=np.float32)
        g = np.stack([(packed >> sh) & 3 for sh in (0, 2, 4, 6)], axis=2)
        return (g.reshape(r, c).astype(np.float32) - 1) * scale[:, None]


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


INV_FREQ = 1.0 / (10000.0 ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT))


def rope(x, pos):  # x [heads, 128]; rotate the first 64 dims, halves of 32
    f = pos * INV_FREQ
    cos = np.concatenate([np.cos(f), np.cos(f)]).astype(np.float32)
    sin = np.concatenate([np.sin(f), np.sin(f)]).astype(np.float32)
    r = x[:, :ROT]
    half = np.concatenate([-r[:, ROT // 2:], r[:, :ROT // 2]], axis=1)
    return np.concatenate([r * cos + half * sin, x[:, ROT:]], axis=1)


class Model:
    def __init__(self, ck):
        self.ck = ck
        self.k = [[] for _ in range(NL)]
        self.v = [[] for _ in range(NL)]
        self.norm_w = ck.dense('model.norm.weight')
        self.cache = {}

    def lw(self, name, tern=False):
        if name not in self.cache:
            self.cache[name] = self.ck.tern(name) if tern else self.ck.dense(name)
        return self.cache[name]

    def step(self, tok, pos):
        ck, P = self.ck, 'model.layers.'
        x = ck.dense('model.word_embeddings.weight', rows=tok).astype(np.float32)
        for L in range(NL):
            p = f'{P}{L}.'
            h = rms(x, self.lw(p + 'input_layernorm.weight'))
            q = (self.lw(p + 'self_attn.q_proj.weight', True) @ h).reshape(NH, HD)
            k = (self.lw(p + 'self_attn.k_proj.weight', True) @ h).reshape(NKV, HD)
            v = (self.lw(p + 'self_attn.v_proj.weight', True) @ h).reshape(NKV, HD)
            q = rms(q, self.lw(p + 'self_attn.q_norm.weight'))
            k = rms(k, self.lw(p + 'self_attn.k_norm.weight'))
            if L not in GLOBAL:
                q, k = rope(q, pos), rope(k, pos)
            self.k[L].append(k); self.v[L].append(v)
            K = np.stack(self.k[L]); V = np.stack(self.v[L])  # [T, NKV, HD]
            lo = 0 if L in GLOBAL else max(0, pos - WINDOW)
            K, V = K[lo: pos + 1], V[lo: pos + 1]
            att = np.empty((NH, HD), np.float32)
            for hq in range(NH):
                kv = hq // (NH // NKV)
                s = K[:, kv] @ q[hq] / np.sqrt(HD)
                s = np.exp(s - s.max()); s /= s.sum()
                att[hq] = s @ V[:, kv]
            x = x + self.lw(p + 'self_attn.o_proj.weight', True) @ att.reshape(-1)
            h = rms(x, self.lw(p + 'post_attention_layernorm.weight'))
            logits = self.lw(p + 'mlp.gate.weight') @ h
            pr = np.exp(logits - logits.max()); pr /= pr.sum()
            top = np.argsort(-pr)[:TOPK]
            wts = pr[top] / (pr[top].sum() + 1e-20)
            y = np.zeros(H, np.float32)
            for e, we in zip(top, wts):
                ep = f'{p}mlp.experts.{e}.'
                g = ck.tern(ep + 'gate_proj.weight') @ h
                u = ck.tern(ep + 'up_proj.weight') @ h
                a = (np.minimum(g, 7.0) / (1 + np.exp(-np.minimum(g, 7.0)))) * np.clip(u, -7.0, 7.0)
                y += we * (ck.tern(ep + 'down_proj.weight') @ a)
            x = x + y
        return rms(x, self.norm_w)

    def logits(self, hfinal):
        if 'lm_head' not in self.cache:
            self.cache['lm_head'] = self.ck.dense('lm_head.weight')
        return self.cache['lm_head'] @ hfinal


def main():
    ckdir, tokp = sys.argv[1], sys.argv[2]
    m = Model(Ckpt(ckdir))
    if sys.argv[3] == '--ids':
        ids = [int(t) for t in sys.argv[4].split(',')]
        out = []
        for i, t in enumerate(ids):
            out.append(m.logits(m.step(t, i)))
        np.save(sys.argv[6], np.stack(out))
        print('saved', np.stack(out).shape, 'argmax', [int(np.argmax(o)) for o in out])
        return
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tokp)
    prompt = f'<|im_start|>user\n{sys.argv[3]}<|im_end|>\n<|im_start|>assistant\n<think>\n'
    ids = tk.encode(prompt).ids
    n_new = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    t0 = time.time()
    for i, t in enumerate(ids):
        hf = m.step(t, i)
    print(f'prompt {len(ids)} tokens, {time.time() - t0:.1f} s', flush=True)
    out = []
    for j in range(n_new):
        nxt = int(np.argmax(m.logits(hf)))
        if nxt == EOS:
            break
        out.append(nxt)
        print(tk.decode([nxt]), end='', flush=True)
        hf = m.step(nxt, len(ids) + j)
    print(f'\n{len(out)} tokens, {time.time() - t0:.1f} s total', flush=True)


if __name__ == '__main__':
    main()

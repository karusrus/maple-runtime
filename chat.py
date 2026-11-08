#!/usr/bin/env python3
"""Ask Maple one question through our C runtime: tokenize, run ./maple, decode tokens as they arrive.
Usage: python3 chat.py CKDIR TOKENIZER.json MAPLE_BIN "question" [max_new_tokens] [threads] [extra maple flags...]"""
import subprocess, sys
from tokenizers import Tokenizer

ck, tokp, binp, question = sys.argv[1:5]
n_new = sys.argv[5] if len(sys.argv) > 5 else '256'
threads = sys.argv[6] if len(sys.argv) > 6 else '4'
tk = Tokenizer.from_file(tokp)
ids = tk.encode(f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n').ids
p = subprocess.Popen([binp, ck, '--ids', ','.join(map(str, ids)), '--n', n_new, '--threads', threads] + sys.argv[7:],
                     stdout=subprocess.PIPE, text=True)
out = []
for line in p.stdout:
    out.append(int(line))
    # decode the whole tail each time so multi-byte characters print once they are complete
    text = tk.decode(out)
    prev = tk.decode(out[:-1])
    print(text[len(prev):], end='', flush=True)
p.wait()
print()

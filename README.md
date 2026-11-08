# maple-runtime

A C runtime for [Maple-Preview](https://huggingface.co/deepgrove/maple-preview) (DeepGrove, MIT: 20B parameters, ~1B active per token, a mixture of 256 experts with ternary weights) on a 2014 iMac: Intel i5-4690 (AVX2), 16 GB DDR3, AMD Radeon R9 M290X 2 GB, macOS Big Sur.

The official code needs CUDA. DeepGrove's llama.cpp fork runs the model on CPUs. This runtime runs the CPU path with AVX2, and on the old Radeon it reads the whole prompt through Metal.

Independent work, not affiliated with DeepGrove.

## Against llama.cpp on the same machine

8 Nov 2026. DeepGrove's fork [deepgrove-ai/llama.cpp](https://github.com/deepgrove-ai/llama.cpp) at `7e30f3a`, with the official GGUF `maple-preview-TQ2_0-head-F16.gguf` (sha256 starts `2fad7b490378`). Same iMac, same token counts, 3 threads. A video-capture service took ~64% of one core during both runs. Raw log: [`bench/results-2026-11-08.log`](bench/results-2026-11-08.log); script: [`bench/bench_vs_llamacpp.sh`](bench/bench_vs_llamacpp.sh).

| tokens/s | llama.cpp fork, CPU | this runtime, CPU | this runtime, prompt on the GPU |
|---|---|---|---|
| prompt, 512 tokens | 78.0 | 78.6 | 184.6 |
| prompt, 2048 tokens | 73.7 | 73.0 | **320.7 (4.4×)** |
| generation, 128 tokens | 15.9 | **29.4–30.8 (1.8–1.9×)** | 27.7–31.5 |

- The prompt on the CPU is the same speed in both.
- Generation is ~1.9× faster here.
- With the prompt on the GPU, a 2,048-token prompt is read 4.4× faster. The fork has no TQ2_0 kernels for Metal.
- With 4 threads both get slower, because the fourth core is shared with the capture service: llama.cpp writes at 8.8 tokens/s, this runtime at 15.9–19.6.
- Greedy text check, same raw prompt: both start with "Rayleigh scattering." and then diverge. llama.cpp falls into a loop ("The sky is not blue because the atmosphere is blue."), this runtime keeps writing coherent text. That is a numerical difference after a few tokens, not evidence that either is more exact.

## What is in it

**v1.1 (3 Nov 2026), CPU:**
- Ternary weights packed 2 bits each and multiplied with int8 activations (`maddubs`, 32 weights per instruction).
- The prompt is read in chunks of 128 tokens.
- Attention runs in parallel over heads and tokens, with an fp16 cache and four query heads per pass over their key/value head.
- The BF16 output head runs on the GPU, one pass for a whole pack of conversations.
- Up to 64 conversations at once (`--multi`).

**v1.2 (6 Nov 2026), the prompt on the GPU (`--gpupre N`, chunks of up to N ≤ 2048 tokens).** Every layer runs on the Radeon. Only the choice of the 8 experts per token stays on the CPU, and the hidden state stays in VRAM. Each layer:
1. **On the card:** RMS norm, q/k/v (ternary; with the o projection, 63 MB for all 24 layers stay in VRAM), q/k norm and RoPE, causal attention over the card's own fp16 key/value cache, o projection added into the hidden state, the second norm, router logits. Then one wait.
2. **On the CPU:** softmax and top-8 per token, pairs grouped by expert (~1 ms per layer).
3. **On the card:** the experts (gate+up with SiLU, down with the routing weight) and their weighted sum added into the hidden state. This is queued without waiting.

The experts' 2-bit codes are not copied ahead. Each layer's span of the memory-mapped shards is a Metal buffer, and the kernels read the codes over PCIe. After the chunk, keys and values go back to the CPU's cache, and generation runs on the CPU.

Measured on the iMac, 6 Nov, capture service running. Prompt time only, `--gpuhead --ctx 4096`:

| Prompt | v1.1, 4 threads | v1.1, 3 threads | `--gpupre 2048`, 3 threads | vs best v1.1 |
|---|---|---|---|---|
| 618 tokens | 13.11 s | 8.41 s | 2.12 s | 4.0× |
| 843 tokens | 17.95 s | 11.83 s | 2.71 s | 4.4× |
| 1,042 tokens | 21.38 s | 14.52 s | 3.09 s | 4.7× |
| 2,568 tokens | 56.62 s | 38.21 s | 7.30 s | 5.2× |

**Where the card's time goes** (843 tokens, `GPRE_PROF=1`): experts 1.55–1.66 s (gate+up ~1.1, down ~0.5), q/k/v 0.24, o 0.17, attention 0.26, router 0.09, norms 0.03. The expert kernels run at ~35–40% of the card's FMA peak. They are bound by ALU work (decoding plus FMAs), not by the bus.

**Default kernel, `gpu_k4.metal`.** It decodes the codes in registers: `(word >> 2s) & 0x03030303`, then one byte conversion per weight, so 1.5 instructions per weight instead of 3. The token's sum of x is taken off at the end. It is 6–7% faster than the first kernels (`GPRE_K1=1`).

## Checks

- **`ref.py`** is a reference forward pass in plain numpy. With `--exact`, the logits are within 0.7% of the maximum (fp16 attention cache), with the same top-1 and top-5 tokens on 5 positions.
- **`GFULL_CHECK=1`** runs every layer on the CPU as well, from the card's own input, and prints the differences. Tested on the R9 M290X and an M1, 256–700 tokens:
  - attention and experts agree to ~1e-6;
  - keys and values differ by one fp16 step in ~0.5% of halves;
  - the top-8 experts differ on at most 1 token of 256 in a layer;
  - 24 generated tokens after a card prompt match the CPU path.

## Tried and kept off

| Flag | What it does | Result |
|---|---|---|
| `--gpux` | Expert cache on the GPU | Exact, but no faster: per-layer latency on this card (~0.25–0.3 ms) is about what the CPU needs for all 8 experts |
| `--gpuhead8` | int8 output head on the GPU | Slower than BF16 |
| `GPRE_K2=1` | Codes unpacked into threadgroup memory | 1.4× slower: fewer waves per compute unit |
| `GPRE_K3=1` | One weight row per thread | 2.3× slower |
| `GPRE_KT=1` | 16-token tiles with the old decoding | Same speed: the tiles alone gain nothing |
| `GPRE_STAGE=1` | Copies each layer's experts into VRAM first (8.7 GB/s) | Kernels only 9% faster: the work is ALU-bound |
| — | 12 or 16 experts per token instead of 8 | Worse answers on an extraction test; the model was trained with 8 |

## Limits

- Greedy decoding only.
- One process per run: the weights load in ~1.7 s once cached, and there is no server mode yet.
- Generation is on the CPU.
- Tested on the R9 M290X (Big Sur) and an M1 only.
- English only, like the model.

## Files

| File | What |
|---|---|
| `convert.py` | Streams the BF16 checkpoint (~40 GB) from Hugging Face and packs it (~5.8 GB), shard by shard |
| `layout.py` | `index.json` → `layout.bin`, the tensor table `maple.c` reads |
| `maple.c` | The runtime |
| `gpu_head.m` | Output head on the GPU (BF16, int8, packs of up to 64 vectors) |
| `gpu_exp.m` | Expert cache on the GPU (`--gpux`, off by default) |
| `gpu_pre.m` | Prompt on the GPU (`--gpupre`): ternary matmul kernels, regions over the mapped shards, the per-layer driver |
| `gpu_full.metal` | Norms, q/k norm with RoPE, attention, router, residual sums for `--gpupre` |
| `gpu_k4.metal` | Expert and q/k/v/o kernels for `--gpupre` (default) |
| `gpu_kt.metal`, `gpu_k2.metal`, `gpu_k3.metal` | Alternative kernels, measured slower or equal; the build embeds them too |
| `ref.py` | Reference forward pass in numpy |
| `chat.py` | One question: tokenize, run, decode |
| `gpu_micro*.m` | GPU microbenchmarks behind the decisions above |
| `bench/` | The comparison with llama.cpp and its raw log |

## Build and run

The tokenizer is `tokenizer.json` from [deepgrove/maple-preview](https://huggingface.co/deepgrove/maple-preview).

```sh
python3 convert.py ck && python3 layout.py ck
for f in gpu_full gpu_k4 gpu_kt gpu_k2 gpu_k3; do xxd -i $f.metal > ${f}_metal.h; done
cc -O3 -ffast-math -march=native -DGPU_HEAD -fobjc-arc -o maple_gpu maple.c gpu_head.m gpu_exp.m gpu_pre.m \
   -framework Metal -framework Foundation -lpthread
python3 chat.py ck tokenizer.json ./maple_gpu "Why is the sky blue?" 400 3 --gpuhead
./maple_gpu ck --ids 1,2,3 --n 200 --threads 3 --ctx 4096 --gpuhead --gpupre 2048   # prompt on the GPU
./maple_gpu ck --multi prompts.txt --n 64 --threads 4 --ctx 512 --gpuhead            # one prompt per line, ids comma-separated
```

The first run after a reboot reads 5.8 GB from disk (about 2 minutes on the iMac's Fusion drive).

## License

MIT. Maple-Preview itself is MIT, by DeepGrove.

#!/bin/zsh
# This runtime vs DeepGrove's llama.cpp fork (github.com/deepgrove-ai/llama.cpp) on the same machine.
# Same token counts for both: prompt 512 and 2048, generation 128, 3 and 4 threads.
# The fork has no TQ2_0 Metal kernels, so it runs on the CPU; this runtime is measured CPU-only and with the prompt on the GPU.
#
# Usage: bench_vs_llamacpp.sh <llama.cpp fork dir, built> <maple-preview-TQ2_0-head-F16.gguf> <ck dir> <tokenizer.json> <text file>
#   The text file only supplies tokens for the prompts (any English text of 2,048+ tokens).
#   Build the fork first: cmake -S . -B build-cpu -DGGML_METAL=OFF && cmake --build build-cpu --target llama-bench llama-simple
L=$1; G=$2; CK=$3; TOK=$4; TXT=$5; BIN=${MAPLE_BIN:-./maple_gpu}
log() { echo "$(date +%H:%M:%S) $*"; }
log "llama.cpp commit $(git -C $L rev-parse --short HEAD)"
for t in 3 4; do
  $L/build-cpu/bin/llama-bench -m $G -p 512,2048 -n 128 -t $t -r 2 -o md 2>&1 | grep -v "^ *$" | tail -6
done
IDS=$(mktemp)
python3 - "$TOK" "$TXT" > $IDS <<'E'
import sys; from tokenizers import Tokenizer
ids = Tokenizer.from_file(sys.argv[1]).encode(open(sys.argv[2]).read()).ids
assert len(ids) >= 2048, len(ids)
print(",".join(map(str, ids[:512]))); print(",".join(map(str, ids[:2048])))
E
for t in 3 4; do for k in 1 2; do
  ids=$(sed -n ${k}p $IDS)
  log "this runtime, CPU, $t threads, prompt $(echo $ids | tr , '\n' | wc -l | tr -d ' ')"
  $BIN $CK --ids $ids --n 128 --threads $t --ctx 4096 --gpuhead 2>&1 >/dev/null | grep "tok/s"
  log "this runtime, prompt on the GPU, $t threads"
  $BIN $CK --ids $ids --n 128 --threads $t --ctx 4096 --gpuhead --gpupre 2048 2>&1 >/dev/null | grep "tok/s"
done; done
rm -f $IDS
# sanity: greedy continuation of the same raw text from both
Q="The main reason the sky looks blue is"
log "llama.cpp greedy"
$L/build-cpu/bin/llama-simple -m $G -n 40 "$Q" 2>/dev/null | tail -c 400; echo
log "this runtime greedy"
ids=$(python3 -c "import sys; from tokenizers import Tokenizer; print(','.join(map(str, Tokenizer.from_file(sys.argv[1]).encode(sys.argv[2]).ids)))" "$TOK" "$Q")
$BIN $CK --ids $ids --n 40 --threads 4 --ctx 512 --gpuhead 2>/dev/null | python3 -c "import sys; from tokenizers import Tokenizer; print(Tokenizer.from_file(sys.argv[1]).decode([int(x) for x in sys.stdin.read().split()]))" "$TOK"

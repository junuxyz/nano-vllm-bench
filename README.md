# nano-vLLM benchmark

Minimal comparison between
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
and
[nano-vllm-v1](https://github.com/junuxyz/nano-vllm-v1).

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone --recurse-submodules \
  https://github.com/junuxyz/nano-vllm-bench.git
cd nano-vllm-bench

MAX_JOBS=4 uv sync
```

Download Qwen3-8B to the persistent workspace:

```bash
uv run hf download Qwen/Qwen3-8B \
  --local-dir /workspace/huggingface/Qwen3-8B
```

## Run

### Rate-limited

```bash
uv run python bench.py \
  --engine nano-vllm \
  --model /workspace/huggingface/Qwen3-8B \
  --num-requests 256 \
  --request-rate 2 \
  --min-input-len 128 \
  --max-input-len 2048 \
  --min-output-len 64 \
  --max-output-len 256 \
  --max-model-len 4096 \
  --token-budget 512 \
  --max-num-seqs 64 \
  --enforce-eager \
  --output results/nano-vllm-rate-2.json

uv run python bench.py \
  --engine nano-vllm-v1 \
  --model /workspace/huggingface/Qwen3-8B \
  --num-requests 256 \
  --request-rate 2 \
  --min-input-len 128 \
  --max-input-len 2048 \
  --min-output-len 64 \
  --max-output-len 256 \
  --max-model-len 4096 \
  --token-budget 512 \
  --max-num-seqs 64 \
  --enforce-eager \
  --output results/nano-vllm-v1-rate-2.json
```

### Maximum throughput

```bash
uv run python bench.py \
  --engine nano-vllm \
  --model /workspace/huggingface/Qwen3-8B \
  --num-requests 256 \
  --request-rate inf \
  --min-input-len 128 \
  --max-input-len 2048 \
  --min-output-len 64 \
  --max-output-len 256 \
  --max-model-len 4096 \
  --token-budget 512 \
  --max-num-seqs 64 \
  --enforce-eager \
  --output results/nano-vllm-rate-inf.json

uv run python bench.py \
  --engine nano-vllm-v1 \
  --model /workspace/huggingface/Qwen3-8B \
  --num-requests 256 \
  --request-rate inf \
  --min-input-len 128 \
  --max-input-len 2048 \
  --min-output-len 64 \
  --max-output-len 256 \
  --max-model-len 4096 \
  --token-budget 512 \
  --max-num-seqs 64 \
  --enforce-eager \
  --output results/nano-vllm-v1-rate-inf.json
```

Both engines are run with `--enforce-eager` due to CUDA graph issues in
nano-vLLM-v1 ([issue #2](https://github.com/slwang-ustc/nano-vllm-v1/issues/2),
[PR #3](https://github.com/slwang-ustc/nano-vllm-v1/pull/3)).

Results include throughput and p50/p90/p99 TTFT, TPOT, and ITL.

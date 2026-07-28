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

```bash
uv run python bench.py \
  --engine nano-vllm \
  --model /workspace/huggingface/Qwen3-8B \
  --num-requests 512 \
  --request-rate 1 \
  --min-input-len 128 \
  --max-input-len 2048 \
  --min-output-len 64 \
  --max-output-len 256 \
  --max-model-len 4096 \
  --token-budget 512 \
  --max-num-seqs 64 \
  --output results/nano-vllm.json

uv run python bench.py \
  --engine nano-vllm-v1 \
  --model /workspace/huggingface/Qwen3-8B \
  --num-requests 512 \
  --request-rate 1 \
  --min-input-len 128 \
  --max-input-len 2048 \
  --min-output-len 64 \
  --max-output-len 256 \
  --max-model-len 4096 \
  --token-budget 512 \
  --max-num-seqs 64 \
  --output results/nano-vllm-v1.json
```

Results include throughput and p50/p90/p99 TTFT, TPOT, and ITL.
Pass `--no-progress` to disable the warmup and request progress bars.

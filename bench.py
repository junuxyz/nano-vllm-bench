"""Minimal serving benchmark for nano-vLLM."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


ENGINES = ("nano-vllm", "nano-vllm-v1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--model", required=True, help="Local Qwen3 model path")
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_workload(args, vocab_size, sampling_params_class):
    rng = random.Random(args.seed)
    prompts = []
    sampling_params = []

    for _ in range(args.num_requests):
        input_len = rng.randint(args.min_input_len, args.max_input_len)
        output_len = rng.randint(args.min_output_len, args.max_output_len)
        prompts.append([rng.randrange(vocab_size) for _ in range(input_len)])
        sampling_params.append(
            sampling_params_class(
                temperature=1.0,
                max_tokens=output_len,
                ignore_eos=True,
            )
        )

    return prompts, sampling_params


def make_arrival_times(args):
    rng = random.Random(args.seed + 1)
    arrival_times = [0.0]
    for _ in range(1, args.num_requests):
        arrival_times.append(
            arrival_times[-1] + rng.expovariate(args.request_rate)
        )
    return arrival_times


def latency_summary(values):
    values_ms = np.asarray(values) * 1000
    return {
        "p50": float(np.percentile(values_ms, 50)),
        "p90": float(np.percentile(values_ms, 90)),
        "p99": float(np.percentile(values_ms, 99)),
    }


def main():
    args = parse_args()

    engine_path = Path(__file__).parent / "engines" / args.engine
    if not engine_path.is_dir():
        raise FileNotFoundError(f"Engine directory not found: {engine_path}")
    sys.path.insert(0, str(engine_path))

    import nanovllm
    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required")
    if not 0 < args.min_input_len <= args.max_input_len:
        raise ValueError("invalid input length range")
    if not 0 < args.min_output_len <= args.max_output_len:
        raise ValueError("invalid output length range")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be positive")
    if args.max_input_len + args.max_output_len > args.max_model_len:
        raise ValueError("input + output length exceeds --max-model-len")

    model_path = Path(args.model).expanduser().resolve()
    output_path = args.output or Path("results") / f"{args.engine}.json"

    llm_args = {
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.token_budget,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": 1,
    }
    if args.engine == "nano-vllm-v1":
        llm_args["chunked_prefill"] = True

    print(f"engine: {args.engine}")
    print(f"module: {nanovllm.__file__}")
    print(f"gpu: {torch.cuda.get_device_name(0)}")

    llm = LLM(str(model_path), **llm_args)

    # Warm up model execution and CUDA graphs with a different prompt.
    warmup = [random.Random(args.seed + 1).randrange(llm.tokenizer.vocab_size)]
    llm.generate(
        [warmup],
        SamplingParams(max_tokens=8, ignore_eos=True),
        use_tqdm=not args.no_progress,
    )

    prompts, sampling_params = make_workload(
        args,
        llm.tokenizer.vocab_size,
        SamplingParams,
    )
    arrival_times = make_arrival_times(args)
    input_tokens = sum(map(len, prompts))
    output_tokens = sum(params.max_tokens for params in sampling_params)

    torch.cuda.synchronize()
    started_at = time.perf_counter()
    records = {}
    active = {}
    next_request = 0
    progress = tqdm(
        total=args.num_requests,
        desc="Processing Requests",
        dynamic_ncols=True,
        disable=args.no_progress,
    )

    while next_request < args.num_requests or active:
        now = time.perf_counter()

        while (
            next_request < args.num_requests
            and now - started_at >= arrival_times[next_request]
        ):
            llm.add_request(
                prompts[next_request],
                sampling_params[next_request],
            )
            sequence = llm.scheduler.waiting[-1]
            record = {
                "arrival_time": started_at + arrival_times[next_request],
                "sequence": sequence,
                "token_times": [],
            }
            records[sequence.seq_id] = record
            active[sequence.seq_id] = record
            next_request += 1

        if active:
            before = {
                seq_id: record["sequence"].num_completion_tokens
                for seq_id, record in active.items()
            }
            llm.step()
            token_time = time.perf_counter()

            finished = []
            for seq_id, record in active.items():
                sequence = record["sequence"]
                new_tokens = sequence.num_completion_tokens - before[seq_id]
                record["token_times"].extend([token_time] * new_tokens)
                if sequence.is_finished:
                    finished.append(seq_id)
            for seq_id in finished:
                del active[seq_id]
            if finished:
                progress.update(len(finished))
                progress.set_postfix(
                    submitted=next_request,
                    active=len(active),
                )
        elif next_request < args.num_requests:
            delay = (
                started_at
                + arrival_times[next_request]
                - time.perf_counter()
            )
            if delay > 0:
                time.sleep(min(delay, 0.001))

    duration = time.perf_counter() - started_at
    progress.close()

    token_times = [record["token_times"] for record in records.values()]
    actual_output_tokens = sum(map(len, token_times))
    if actual_output_tokens != output_tokens:
        raise RuntimeError(
            f"expected {output_tokens} output tokens, "
            f"got {actual_output_tokens}"
        )

    ttft = [
        times[0] - record["arrival_time"]
        for record, times in zip(records.values(), token_times)
    ]
    per_request_itl = [np.diff(times) for times in token_times]
    itl = np.concatenate(per_request_itl)
    tpot = [float(np.mean(times)) for times in per_request_itl]

    result = {
        "engine": args.engine,
        "model": str(model_path),
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "num_requests": args.num_requests,
        "input_length": [args.min_input_len, args.max_input_len],
        "output_length": [args.min_output_len, args.max_output_len],
        "token_budget": args.token_budget,
        "request_rate": args.request_rate,
        "duration_seconds": duration,
        "input_tokens": input_tokens,
        "output_tokens": actual_output_tokens,
        "requests_per_second": args.num_requests / duration,
        "output_tokens_per_second": actual_output_tokens / duration,
        "total_tokens_per_second": (
            input_tokens + actual_output_tokens
        ) / duration,
        "ttft_ms": latency_summary(ttft),
        "tpot_ms": latency_summary(tpot),
        "itl_ms": latency_summary(itl),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"time: {duration:.2f}s")
    print(f"output throughput: {result['output_tokens_per_second']:.2f} tok/s")
    for metric in ("ttft_ms", "tpot_ms", "itl_ms"):
        values = result[metric]
        print(
            f"{metric.removesuffix('_ms').upper()} p90/p99: "
            f"{values['p90']:.2f}/{values['p99']:.2f} ms"
        )
    print(f"result: {output_path.resolve()}")


if __name__ == "__main__":
    main()

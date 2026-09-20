"""Official Qwen streaming API, in its own Python/vLLM environment.

The parent routes one acoustic segment at a time. Stdout is reserved for JSON;
all library output (including vLLM child processes) goes to the worker log.
"""
import argparse
import base64
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    args = parser.parse_args()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    import numpy as np
    from qwen_asr import Qwen3ASRModel

    root = Path(__file__).resolve().parents[1]
    config = dict(model=str(root / "models/qwen3-asr-0.6b"),
                  max_new_tokens=128, gpu_memory_utilization=0.45,
                  kv_cache_memory_bytes=512 * 1024**2,
                  max_model_len=4096, max_num_seqs=1,
                  max_num_batched_tokens=4096, enforce_eager=True,
                  enable_prefix_caching=False, mm_processor_cache_gb=0,
                  limit_mm_per_prompt={"audio": 1}, skip_mm_profiling=True,
                  seed=0)
    model = Qwen3ASRModel.LLM(**config)
    # Initialize audio preprocessing / CUDA kernels before the replay clock starts.
    # Synthetic silence carries no test content; its state and transcript are discarded.
    warmup_start = time.monotonic()
    warmup_state = model.init_streaming_state(chunk_size_sec=2.0)
    model.streaming_transcribe(np.zeros(32000, dtype=np.float32), warmup_state)
    model.finish_streaming_transcribe(warmup_state)
    del warmup_state
    warmup = dict(audio="2 seconds of synthetic silence; state discarded",
                  seconds=time.monotonic() - warmup_start, scope="Qwen only")
    # Keep the first four seconds revisable even when emitting every second.
    streaming = dict(chunk_size_sec=args.chunk_seconds,
                     unfixed_chunk_num=math.ceil(4 / args.chunk_seconds), unfixed_token_num=5)
    runtime = {name: importlib.metadata.version(name)
               for name in ("qwen-asr", "vllm", "torch", "transformers", "numpy")}
    runtime["python"] = sys.version.split()[0]
    runtime["pytorch_alloc_conf"] = os.environ.get("PYTORCH_ALLOC_CONF")
    state = None

    def send(value):
        protocol.write(json.dumps(value, ensure_ascii=False) + "\n")

    send(dict(ready=True, config=config, chunk_seconds=args.chunk_seconds,
              streaming=streaming, warmup=warmup, runtime=runtime))
    for line in sys.stdin:
        request = json.loads(line)
        start = time.monotonic()
        if request["op"] == "start":
            if state is not None:
                raise RuntimeError("Previous ASR segment was not closed")
            state = model.init_streaming_state(**streaming)
        else:
            if state is None:
                raise RuntimeError("No active ASR segment")
            pcm = np.frombuffer(base64.b64decode(request["pcm"], validate=True), dtype="<f4")
            model.streaming_transcribe(pcm, state)
            if request["op"] == "finish":
                model.finish_streaming_transcribe(state)
            elif request["op"] != "push":
                raise ValueError("Unknown operation")
        send(dict(text=state.text, text_mode="snapshot", language=state.language,
                  compute_s=time.monotonic() - start, padding_samples=0))
        if request["op"] == "finish":
            state = None


if __name__ == "__main__":
    main()

"""Small stdio bridge: Sortformer and Qwen retain independent dependencies."""
import base64
from datetime import datetime
from difflib import SequenceMatcher
import json
import os
import subprocess
import time

import numpy as np

from common import ROOT, SR


def _normalized_with_ends(text):
    normalized, ends = [], []
    for index, character in enumerate(text):
        if character.isalnum():
            for folded in character.casefold():
                normalized.append(folded)
                ends.append(index + 1)
    return "".join(normalized), ends


def remove_overlap_prefix(previous, current):
    """Remove a near-identical previous suffix from the current prefix."""
    before, _ = _normalized_with_ends(previous)
    after, ends = _normalized_with_ends(current)
    limit = min(80, len(after))
    for current_size in range(limit, 3, -1):
        low = max(4, current_size - 12)
        high = min(80, len(before), current_size + 12)
        for previous_size in range(low, high + 1):
            matcher = SequenceMatcher(None, before[-previous_size:], after[:current_size], autojunk=False)
            blocks = [block for block in matcher.get_matching_blocks() if block.size]
            matched = sum(block.size for block in blocks)
            if (blocks and blocks[0].b == 0
                    and blocks[-1].b + blocks[-1].size == current_size
                    and matched / current_size >= 0.85
                    and matched / previous_size >= 0.75):
                return current[ends[current_size - 1]:].lstrip(" \t\r\n,.;:!?，。；：！？")
    if after and len(after) < 4 and before.endswith(after):
        return ""
    return current


class QwenWorker:
    def __init__(self, chunk_seconds=1.0):
        self.chunk_samples = round(chunk_seconds * SR)
        self.log_path = ROOT / "logs" / ("qwen-worker-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".log")
        env = dict(os.environ, PYTHONNOUSERSITE="1", VLLM_WORKER_MULTIPROC_METHOD="spawn",
                   VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                   PYTORCH_ALLOC_CONF="expandable_segments:True")
        for key in ("PYTHONPATH", "LD_LIBRARY_PATH", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
                    "PYTORCH_CUDA_ALLOC_CONF"):
            env.pop(key, None)
        with self.log_path.open("w") as log:
            self.process = subprocess.Popen(
                [str(ROOT / ".venv-qwen/bin/python"), str(ROOT / "scripts/qwen_worker.py"),
                 "--chunk-seconds", str(chunk_seconds)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        try:
            self.config = self._read()
            if not self.config.get("ready"):
                raise RuntimeError("Qwen did not become ready")
        except BaseException:
            self.close()
            raise

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Qwen worker stopped; see " + str(self.log_path))
        return json.loads(line)

    def request(self, op, audio=None):
        request = dict(op=op)
        if audio is not None:
            request["pcm"] = base64.b64encode(np.asarray(audio, dtype="<f4").tobytes()).decode("ascii")
        start = time.monotonic()
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self._read()
        result["worker_compute_s"] = result["compute_s"]
        result["compute_s"] = time.monotonic() - start  # include IPC in the integration cost
        return result

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()


class QwenSegmentASR:
    def __init__(self, worker):
        self.worker = worker
        self.pending = np.empty(0, dtype=np.float32)
        self.closed = False
        self.overlap_context = ""
        self.deferred_compute_s = 0.0
        self.worker.request("start")

    def _visible(self, update):
        update = dict(update)
        if self.deferred_compute_s:
            update["compute_s"] += self.deferred_compute_s
            self.deferred_compute_s = 0.0
        if self.overlap_context:
            update["text"] = remove_overlap_prefix(self.overlap_context, update["text"])
        return update

    def prime(self, audio, previous_text):
        if self.closed or len(self.pending):
            raise ValueError("ASR overlap must be primed before new audio")
        self.overlap_context = previous_text
        self.pending = np.asarray(audio, dtype=np.float32).copy()
        while len(self.pending) >= self.worker.chunk_samples:
            count = self.worker.chunk_samples
            update = self.worker.request("push", self.pending[:count])
            self.deferred_compute_s += update["compute_s"]
            self.pending = self.pending[count:]

    def push(self, audio):
        if self.closed:
            raise ValueError("ASR segment already closed")
        self.pending = np.concatenate((self.pending, audio))
        updates = []
        while len(self.pending) >= self.worker.chunk_samples:
            count = self.worker.chunk_samples
            updates.append(self._visible(self.worker.request("push", self.pending[:count])))
            self.pending = self.pending[count:]
        return updates

    def finish(self):
        if self.closed:
            raise ValueError("ASR segment already closed")
        self.closed = True
        return self._visible(self.worker.request("finish", self.pending))

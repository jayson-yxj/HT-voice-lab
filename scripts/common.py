"""Shared audio clock and result format for the three independent runners."""
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def wav_info(path):
    info = sf.info(path)
    if info.samplerate != SR or info.channels != 1:
        raise ValueError("Run './lab prepare INPUT' first: expected 16 kHz mono audio")
    if info.frames == 0:
        raise ValueError("Empty audio")
    return info


def sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def replay(path, samples, session):
    """Only read a block after its last sample has become available on the clock."""
    with sf.SoundFile(path) as source:
        offset = 0
        while offset < len(source):
            count = min(samples, len(source) - offset)
            session.wait_until((offset + count) / SR)
            audio = source.read(count, dtype="float32")
            offset += len(audio)
            yield audio, offset / SR, offset == len(source)


def merge_segments(segments, duration):
    groups = {}
    for item in segments:
        start, end = max(0.0, float(item["start"])), min(duration, float(item["end"]))
        if end > start:
            groups.setdefault(str(item["speaker"]), []).append((start, end))
    result = []
    for speaker, spans in groups.items():
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        result.extend(dict(start=round(a, 6), end=round(b, 6), speaker=speaker) for a, b in merged)
    return sorted(result, key=lambda x: (x["start"], x["speaker"]))


class Session:
    def __init__(self, args, load_seconds, config):
        import torch
        self.args = args
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=False)
        self.duration = wav_info(args.audio).duration
        self.events = (self.out / "events.jsonl").open("w")
        self.costs, self.delays = [], []
        self.first_output = None
        self.meta = {
            "backend": args.backend, "audio": str(Path(args.audio).resolve()),
            "audio_seconds": self.duration, "realtime": not args.fast,
            "device": args.device, "model_load_seconds": load_seconds,
            "warmup_performed": config.get("warmup_performed", False), "config": config,
            "models": json.loads((ROOT / "models.json").read_text()),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "timing_note": "Output delay is publication time minus committed audio watermark; not turn-completion latency. Fast mode does not report wall-clock audio delay. Model loading and explicit warmup are excluded from replay metrics; see warmup_performed and config for warmup scope.",
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        write_json(self.out / "run.json", self.meta)
        self.origin = time.monotonic()

    def wait_until(self, audio_time):
        if not self.args.fast:
            time.sleep(max(0, self.origin + audio_time - time.monotonic()))

    def emit(self, available, committed, compute_seconds, final_flush=False, **payload):
        emitted = time.monotonic() - self.origin
        delay = max(0.0, emitted - committed) if not self.args.fast and committed is not None and committed > 0 else None
        item = dict(available_audio_s=available, committed_until_s=committed,
                    emitted_at_s=emitted, compute_s=compute_seconds,
                    output_delay_s=delay, final_flush=final_flush, **payload)
        self.events.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.events.flush()
        self.costs.append(compute_seconds)
        if delay is not None and not final_flush:
            self.delays.append(delay)
        if self.first_output is None and (payload.get("segments") or payload.get("text")):
            self.first_output = emitted

    def finish(self, segments=None, text=None, extra=None):
        import torch
        self.events.close()
        result = dict(self.meta, status="complete", elapsed_seconds=time.monotonic() - self.origin,
                      compute_seconds=sum(self.costs), compute_rtf=sum(self.costs) / self.duration,
                      first_nonempty_output_seconds=self.first_output,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
                      peak_gpu_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None)
        for label, values in [("compute_ms", [x * 1000 for x in self.costs]), ("output_delay_s", self.delays)]:
            result[label] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))} if values else None
        if segments is not None:
            merged = merge_segments(segments, self.duration)
            write_json(self.out / "segments.json", merged)
            with (self.out / "prediction.rttm").open("w") as f:
                for s in merged:
                    f.write("SPEAKER audio 1 {:.6f} {:.6f} <NA> <NA> {} <NA> <NA>\n".format(s["start"], s["end"] - s["start"], s["speaker"]))
            result["detected_speakers"] = sorted({s["speaker"] for s in merged})
        if text is not None:
            (self.out / "transcript.txt").write_text(text + "\n")
        if extra:
            result.update(extra)
        write_json(self.out / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

import argparse
import csv
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def doctor(args):
    import torch
    packages = {}
    for name in ["torch", "torchaudio", "torchvision", "funasr", "diart", "pyannote.audio", "nemo_toolkit", "numpy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    models = {}
    for name, spec in json.loads((ROOT / "models.json").read_text()).items():
        lock = ROOT / "models" / name / "download.json"
        meta = json.loads(lock.read_text()) if lock.is_file() else {}
        models[name] = meta.get("revision") == spec["revision"] and all(
            (ROOT / "models" / name / filename).is_file() and
            (ROOT / "models" / name / filename).stat().st_size == meta.get("files", {}).get(filename, {}).get("size")
            for filename in spec["files"])
    probes = {}
    if args.imports:
        for module in ["funasr", "diart", "nemo.collections.asr.models"]:
            try:
                importlib.import_module(module)
                probes[module] = "ok"
            except Exception as exc:
                probes[module] = "{}: {}".format(type(exc).__name__, exc)
    if torch.cuda.is_available():
        value = (torch.ones(8, device="cuda") * 2).sum().item()
        assert value == 16
    from models import token
    from groq_semantic import load_api_key
    result = dict(python=sys.version.split()[0], packages=packages, ffmpeg=shutil.which("ffmpeg"),
                  cuda_available=torch.cuda.is_available(),
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  hf_login_present=bool(token()), groq_login_present=bool(load_api_key()),
                  models_ready=models, import_checks=probes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(packages.values()) and all(x == "ok" for x in probes.values()) else 1


def prepare(args):
    from common import SR, wav_info, write_json
    source = Path(args.input).expanduser().resolve(strict=True)
    if args.start < 0 or (args.duration is not None and args.duration <= 0):
        raise ValueError("Invalid clipping interval")
    digest = hashlib.sha256()
    with source.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    identity = hashlib.sha256((checksum + str(args.start) + str(args.duration)).encode()).hexdigest()[:12]
    directory = ROOT / "data/prepared"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (source.stem + "-" + identity + ".wav")
    if not target.exists():
        partial = target.with_name(target.stem + ".part.wav")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source), "-ss", str(args.start)]
        if args.duration is not None:
            command += ["-t", str(args.duration)]
        command += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(partial)]
        subprocess.run(command, check=True)
        wav_info(partial)
        partial.replace(target)
    info = wav_info(target)
    write_json(target.with_suffix(".json"), dict(source=str(source), source_sha256=checksum,
               clip_start_s=args.start, requested_duration_s=args.duration,
               audio_seconds=info.duration, sample_rate=SR, channels=1))
    print(target)
    return 0


def check_groq(args):
    from groq_semantic import check_model_access
    models = args.model or ["qwen/qwen3.8-27b"]
    available = check_model_access(models, args.timeout, proxy=args.proxy)
    print(json.dumps(dict(status="ok", models=available), ensure_ascii=False, indent=2))
    return 0


def run(args):
    import random
    import numpy as np
    import torch
    from common import wav_info
    args.audio = str(Path(args.audio).resolve(strict=True))
    wav_info(args.audio)
    if not args.output:
        args.output = str(ROOT / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.backend))
    if Path(args.output).exists():
        raise ValueError("Output directory already exists; choose a new path")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    if not 0.5 <= args.latency <= 5:
        raise ValueError("diart latency must be between 0.5 and 5 seconds")
    if not 0 < args.threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if not math.isfinite(args.silence_gap) or args.silence_gap < 0.08:
        raise ValueError("ASR silence gap must be finite and at least 0.08 seconds")
    if args.asr_model == "qwen3" and not (
            math.isfinite(args.qwen_window_pause) and math.isfinite(args.qwen_window_limit)
            and math.isfinite(args.qwen_window_overlap)
            and 1 <= args.qwen_window_pause <= args.qwen_window_limit
            and 0 <= args.qwen_window_overlap < args.qwen_window_pause):
        raise ValueError("Qwen window settings require 1 <= pause <= limit and 0 <= overlap < pause")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    module = importlib.import_module("run_" + args.backend)
    module.run(args)
    return 0


def compare(args):
    base = Path(args.output or ROOT / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-comparison"))
    base.mkdir(parents=True, exist_ok=False)
    rows = []
    for backend in ["sortformer", "asr"]:
        output = base / backend
        command = [str(ROOT / "lab"), "run", str(Path(args.audio).resolve()), "--backend", backend,
                   "--device", args.device, "--output", str(output), "--latency", str(args.latency)]
        if args.fast:
            command.append("--fast")
        print("Running {}: {}".format(backend, base / (backend + ".log")), flush=True)
        with (base / (backend + ".log")).open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = dict(backend=backend, status="complete" if process.returncode == 0 else "failed", exit_code=process.returncode)
        summary = output / "summary.json"
        if summary.is_file():
            metrics = json.loads(summary.read_text())
            row.update({k: metrics.get(k) for k in ["compute_rtf", "model_load_seconds", "first_nonempty_output_seconds", "peak_rss_mib", "peak_gpu_allocated_mib"]})
            row["output_delay_p95_s"] = (metrics.get("output_delay_s") or {}).get("p95")
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (base / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(base / "comparison.csv")
    return 0 if all(r["exit_code"] == 0 for r in rows) else 2


def main():
    parser = argparse.ArgumentParser(description="HT voice lab: streaming speaker/ASR benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor", help="Check packages, GPU and local model availability")
    p.add_argument("--imports", action="store_true")
    p.add_argument("--output")
    p.set_defaults(func=doctor)
    p = sub.add_parser("selftest", help="Check metrics, audio boundaries and causal mel features")
    p.set_defaults(func=lambda a: subprocess.call([sys.executable, str(ROOT / "scripts/test_lab.py")]))
    p = sub.add_parser("prepare", help="Convert audio/video to 16 kHz mono WAV")
    p.add_argument("input")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float)
    p.set_defaults(func=prepare)
    for name in ["run", "compare"]:
        p = sub.add_parser(name)
        p.add_argument("audio")
        p.add_argument("--output")
        p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
        p.add_argument("--fast", action="store_true", help="Throughput/smoke only; not realtime latency")
        p.add_argument("--latency", type=float, default=1.0, help="diart buffering latency in seconds")
        if name == "run":
            p.add_argument("--backend", choices=["diart", "sortformer", "asr", "conversation"], default="sortformer",
                           help="Inference backend (default: sortformer)")
            p.add_argument("--threshold", type=float, default=0.5)
            p.add_argument("--asr-model", choices=["paraformer", "qwen3"], default="paraformer",
                           help="ASR model for --backend conversation")
            p.add_argument("--qwen-chunk-seconds", type=float, choices=[1.0, 2.0], default=1.0,
                           help="Qwen streaming decode interval; conversation backend only")
            p.add_argument("--qwen-window-pause", type=float, default=12.0,
                           help="Roll Qwen state at the next speaker pause after this many seconds")
            p.add_argument("--qwen-window-limit", type=float, default=15.0,
                           help="Hard limit for one cumulative Qwen acoustic window")
            p.add_argument("--qwen-window-overlap", type=float, default=1.5,
                           help="Audio pre-roll used to recover words across Qwen window boundaries")
            p.add_argument("--silence-gap", type=float, default=0.8,
                           help="Conversation ASR segment silence gap in seconds; not semantic turn completion")
            p.add_argument("--max-global-speakers", type=int, default=12,
                           help="Session-wide identities; Sortformer still handles at most four local speakers")
            p.add_argument("--speaker-episode-gap", type=float, default=6.0,
                           help="Silence that starts a new local-speaker episode")
            p.add_argument("--speaker-slot-recheck-gap", type=float, default=8.0,
                           help="Recheck a reused local slot after this many seconds")
            p.add_argument("--no-speaker-stabilization", dest="speaker_stabilization",
                           action="store_false", default=True,
                           help="Use raw Sortformer slots without CPU voiceprint stabilization")
        p.set_defaults(func=run if name == "run" else compare)
    p = sub.add_parser("fetch", help="Download revision-pinned model files")
    p.add_argument("--only", nargs="+", choices=list(json.loads((ROOT / "models.json").read_text())))
    p.set_defaults(func=lambda a: subprocess.call([sys.executable, str(ROOT / "scripts/models.py")] + (["--only", *a.only] if a.only else [])))
    p = sub.add_parser("login", help="Interactive local Hugging Face login; do not paste tokens into chat")
    p.set_defaults(func=lambda a: subprocess.call([str(Path(sys.executable).parent / "hf"), "auth", "login"]))
    p = sub.add_parser("login-groq", help="Securely save a Groq API key in the user config directory")
    p.set_defaults(func=lambda a: importlib.import_module("groq_semantic").login())
    p = sub.add_parser("check-groq", help="Check Groq key and model permissions before a demo")
    p.add_argument("--model", action="append")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--proxy")
    p.set_defaults(func=check_groq)
    p = sub.add_parser("score", help="Score diarization against manually verified RTTM")
    p.add_argument("--reference", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--collar", type=float, default=0.0)
    p.add_argument("--skip-overlap", action="store_true")
    p.add_argument("--output")
    p.set_defaults(func=lambda a: importlib.import_module("score").run(a))
    p = sub.add_parser("score-asr", help="Chinese character error rate against verified text")
    p.add_argument("--reference", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--output")
    p.set_defaults(func=lambda a: importlib.import_module("score").run_asr(a))
    p = sub.add_parser("analyze-dialogue", help="Track per-speaker semantic state and explicit reply requests")
    p.add_argument("speech_spans")
    p.add_argument("--assistant-alias", action="append",
                   help="Assistant name that must be spoken before a reply is allowed; repeat for aliases")
    p.add_argument("--invocation-context-seconds", type=float, default=30.0)
    p.add_argument("--semantic-provider", choices=["pending", "groq"], default="pending")
    p.add_argument("--semantic-model", default="qwen/qwen3.8-27b")
    p.add_argument("--semantic-timeout", type=float, default=10.0)
    p.add_argument("--semantic-proxy",
                   help="Optional HTTPS proxy URL for the semantic provider")
    p.add_argument("--semantic-minimum-interval", type=float, default=0.0,
                   help="Optional request spacing for offline runs under free-tier limits")
    p.add_argument("--output")
    p.set_defaults(func=lambda a: importlib.import_module("dialogue_state").run(a))
    p = sub.add_parser("build-memory", help="Compress completed speaker turns into session memory")
    p.add_argument("dialogue_state")
    p.add_argument("--session-id")
    p.add_argument("--memory-provider", choices=["local", "groq"], default="local")
    p.add_argument("--memory-model", default="qwen/qwen3.8-27b")
    p.add_argument("--memory-timeout", type=float, default=30.0)
    p.add_argument("--memory-proxy", help="Optional HTTPS proxy URL for the memory provider")
    p.add_argument("--output")
    p.set_defaults(func=lambda a: importlib.import_module("session_memory").run(a))
    p = sub.add_parser("build-participant-context",
                       help="Create anonymous participants and verified turn evidence")
    p.add_argument("session_memory")
    p.add_argument("--personal-provider", choices=["none", "groq"], default="none")
    p.add_argument("--personal-model", default="qwen/qwen3.8-27b")
    p.add_argument("--personal-timeout", type=float, default=30.0)
    p.add_argument("--personal-proxy", help="Optional HTTPS proxy URL for personal information extraction")
    p.add_argument("--personal-minimum-interval", type=float, default=0.0)
    p.add_argument("--output")
    p.set_defaults(func=lambda a: importlib.import_module("participant_context").run(a))
    p = sub.add_parser("eval-personal-identity",
                       help="Evaluate personal information extraction on the fixed fixture")
    p.add_argument("--fixture", default=str(ROOT / "fixtures" / "personal_identity_eval.json"))
    p.add_argument("--model", default="qwen/qwen3.8-27b")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--proxy")
    p.add_argument("--minimum-interval", type=float, default=0.2)
    p.add_argument("--output", default=str(ROOT / "results" / "personal_identity_eval.json"))
    p.set_defaults(func=lambda a: importlib.import_module("eval_personal_identity").run(a))
    p = sub.add_parser("visualize-memory", help="Watch session memory and serve a live relationship graph")
    p.add_argument("session_memory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="Do not open the graph in a browser")
    p.set_defaults(func=lambda a: importlib.import_module("memory_graph").run(a))
    p = sub.add_parser("demo", help="Run conversation, semantic memory and the live graph together")
    p.add_argument("audio")
    p.add_argument("--output")
    p.add_argument("--session-id")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--fast", action="store_true", help="Process audio without realtime waiting")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--asr-model", choices=["paraformer", "qwen3"], default="qwen3")
    p.add_argument("--qwen-chunk-seconds", type=float, choices=[1.0, 2.0], default=1.0)
    p.add_argument("--qwen-window-pause", type=float, default=12.0)
    p.add_argument("--qwen-window-limit", type=float, default=15.0)
    p.add_argument("--qwen-window-overlap", type=float, default=1.5)
    p.add_argument("--silence-gap", type=float, default=0.8)
    p.add_argument("--max-global-speakers", type=int, default=12)
    p.add_argument("--speaker-episode-gap", type=float, default=6.0)
    p.add_argument("--speaker-slot-recheck-gap", type=float, default=8.0)
    p.add_argument("--no-speaker-stabilization", dest="speaker_stabilization",
                   action="store_false", default=True)
    p.add_argument("--assistant-alias", action="append")
    p.add_argument("--invocation-context-seconds", type=float, default=30.0)
    p.add_argument("--semantic-model", default="qwen/qwen3.8-27b")
    p.add_argument("--semantic-timeout", type=float, default=10.0)
    p.add_argument("--semantic-minimum-interval", type=float, default=0.0)
    p.add_argument("--memory-model", default="qwen/qwen3.8-27b")
    p.add_argument("--memory-timeout", type=float, default=30.0)
    p.add_argument("--groq-proxy", help="Optional HTTPS proxy used by both semantic stages")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(func=lambda a: importlib.import_module("live_demo").run(a))
    args = parser.parse_args()
    try:
        return args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

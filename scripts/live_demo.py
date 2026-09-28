"""Run file-audio inference, semantic memory, and the live graph as one demo."""
from datetime import datetime
import json
from pathlib import Path
from queue import Full, Queue
import subprocess
import sys
import threading
import time
import webbrowser

from common import ROOT, write_json
from dialogue_state import DialogueStateTracker
from groq_semantic import GroqSemanticJudge, check_model_access
from memory_graph import create_server
from participant_context import (GroqPersonalInfoExtractor, build_from_path,
                                 refresh_with_personal_info)
from session_memory import GroqMemoryCompressor, build, empty, save


ARTIFICIAL_CLOSES = {"asr_window_pause", "asr_window_limit"}


class LiveMemoryUpdater:
    """Consume each stable speech span once and replace the memory snapshot."""

    def __init__(self, tracker, compressor, output, session_id, on_memory_saved=None):
        self.tracker = tracker
        self.compressor = compressor
        self.output = Path(output)
        self.dialogue_path = self.output / "dialogue_state.json"
        self.memory_path = self.output / "session_memory.json"
        self.session_id = session_id
        self.processed = []
        self.on_memory_saved = on_memory_saved

    def _save(self, memory):
        save(self.memory_path, memory)
        if self.on_memory_saved:
            self.on_memory_saved()

    def initialize(self):
        dialogue = self.tracker.result()
        write_json(self.dialogue_path, dialogue)
        self._save(empty(
            dialogue, self.dialogue_path, self.session_id, self.compressor))

    @staticmethod
    def _stable(spans):
        return [item for item in spans
                if item.get("status") == "final"
                and item.get("close_reason") not in ARTIFICIAL_CLOSES]

    @staticmethod
    def _signature(span):
        return span.get("id"), span.get("speaker"), span.get("text"), span.get("close_reason")

    def sync(self, spans):
        stable = self._stable(spans)
        signatures = [self._signature(item) for item in stable]
        if signatures[:len(self.processed)] != self.processed:
            raise RuntimeError("A finalized speech span changed after semantic processing")
        count = 0
        for span in stable[len(self.processed):]:
            print("[live] 正在分析 {} {:.2f}–{:.2f}s，请等待图更新…".format(
                span.get("speaker") or "重叠未分离",
                float(span.get("start_s", 0)), float(span.get("end_s", 0))), flush=True)
            record = self.tracker.push(span)
            dialogue = self.tracker.result()
            write_json(self.dialogue_path, dialogue)
            if any(item.get("turn_complete") is True for item in dialogue["contributions"]):
                print("[live] 正在压缩会话记忆…", flush=True)
                memory = build(dialogue, self.dialogue_path,
                               self.session_id, self.compressor)
            else:
                memory = empty(dialogue, self.dialogue_path,
                               self.session_id, self.compressor)
            self._save(memory)
            self.processed.append(self._signature(span))
            count += 1
            state = "完成" if record.get("turn_complete") is True else "待续说"
            print("[live] {} {}：{}".format(
                record.get("speaker") or "重叠未分离", state,
                record.get("text") or "（无文字）"), flush=True)
        return count


class LiveParticipantUpdater:
    """Coalesce memory snapshots and refresh only speakers with new evidence."""

    def __init__(self, memory_path, context_path, extractor):
        self.memory_path = Path(memory_path)
        self.context_path = Path(context_path)
        self.extractor = extractor
        self.pending = Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.context = None
        self.error = None
        self.started = False

    def start(self):
        self.thread.start()
        self.started = True

    def request(self):
        try:
            self.pending.put_nowait(True)
        except Full:
            pass

    def _build_latest(self):
        for attempt in range(5):
            try:
                return build_from_path(self.memory_path)
            except ValueError as exc:
                if "SHA-256" not in str(exc) or attempt == 4:
                    raise
                time.sleep(0.05)

    def _run(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                context, changed = refresh_with_personal_info(
                    self._build_latest(), self.extractor, self.context)
                write_json(self.context_path, context)
                self.context = context
                self.error = None
                if changed:
                    print("[personal] 已更新 {}：共 {} 条个人信息。".format(
                        "/".join(changed), context["stats"]["observations"]), flush=True)
            except Exception as exc:  # Keep audio and session memory running.
                self.error = exc
                print("[personal] 更新失败：{}".format(exc), file=sys.stderr, flush=True)
            finally:
                self.pending.task_done()

    def flush(self):
        self.pending.join()
        if self.error:
            raise RuntimeError("Personal context update failed: {}".format(self.error))

    def close(self):
        if not self.started:
            return
        self.pending.join()
        self.pending.put(None)
        self.thread.join()


def _inference_command(args, output):
    command = [
        sys.executable, str(ROOT / "scripts" / "cli.py"), "run", str(args.audio),
        "--backend", "conversation", "--output", str(output),
        "--device", args.device, "--threshold", str(args.threshold),
        "--asr-model", args.asr_model,
        "--qwen-chunk-seconds", str(args.qwen_chunk_seconds),
        "--qwen-window-pause", str(args.qwen_window_pause),
        "--qwen-window-limit", str(args.qwen_window_limit),
        "--qwen-window-overlap", str(args.qwen_window_overlap),
        "--silence-gap", str(args.silence_gap),
    ]
    if args.fast:
        command.append("--fast")
    if not args.speaker_stabilization:
        command.append("--no-speaker-stabilization")
    return command


def run(args):
    audio = Path(args.audio).expanduser().resolve(strict=True)
    args.audio = str(audio)
    output = (Path(args.output).expanduser().resolve() if args.output else
              ROOT / "results" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-demo"))
    if output.exists():
        raise ValueError("Output directory already exists: {}".format(output))
    if args.invocation_context_seconds <= 0:
        raise ValueError("Invocation context must be positive")
    print("Checking Groq model access…", flush=True)
    check_model_access(
        {args.semantic_model, args.memory_model},
        timeout=max(args.semantic_timeout, args.memory_timeout),
        proxy=args.groq_proxy)
    aliases = tuple(args.assistant_alias or ["小派"])
    judge = GroqSemanticJudge(
        args.semantic_model, args.semantic_timeout, proxy=args.groq_proxy,
        minimum_interval=args.semantic_minimum_interval)
    compressor = GroqMemoryCompressor(
        args.memory_model, args.memory_timeout, proxy=args.groq_proxy)
    personal_extractor = GroqPersonalInfoExtractor(
        args.memory_model, args.memory_timeout, proxy=args.groq_proxy,
        minimum_interval=args.semantic_minimum_interval)
    tracker = DialogueStateTracker(aliases, args.invocation_context_seconds, judge)
    personal = LiveParticipantUpdater(
        output / "session_memory.json", output / "participant_context.json",
        personal_extractor)
    updater = LiveMemoryUpdater(
        tracker, compressor, output, args.session_id or output.name,
        on_memory_saved=personal.request)

    server, _, url = create_server(
        output / "session_memory.json", args.host, args.port, require_file=False)
    viewer = threading.Thread(target=server.serve_forever, daemon=True)
    viewer.start()
    print("Live graph: {}".format(url), flush=True)
    print("Output: {}".format(output), flush=True)
    if not args.no_open:
        webbrowser.open(url)

    process = None
    try:
        process = subprocess.Popen(_inference_command(args, output))
        while not output.is_dir():
            if process.poll() is not None:
                raise RuntimeError("Conversation inference stopped before creating output")
            time.sleep(0.1)
        personal.start()
        updater.initialize()
        spans_path = output / "speech_spans.json"
        revision = None
        while True:
            if spans_path.is_file():
                stat = spans_path.stat()
                current = (stat.st_mtime_ns, stat.st_size)
                if current != revision:
                    spans = json.loads(spans_path.read_text())
                    if not isinstance(spans, list):
                        raise ValueError("Live speech spans must be a JSON list")
                    updater.sync(spans)
                    revision = current
            if process.poll() is not None:
                break
            time.sleep(0.2)
        if process.returncode:
            raise RuntimeError("Conversation inference failed with exit code {}".format(
                process.returncode))
        if spans_path.is_file():
            updater.sync(json.loads(spans_path.read_text()))
        personal.flush()
        print("Demo complete. The graph remains open; press Ctrl+C to stop.", flush=True)
        while viewer.is_alive():
            viewer.join(1)
    except KeyboardInterrupt:
        pass
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        server.shutdown()
        server.server_close()
        personal.close()
    return 0

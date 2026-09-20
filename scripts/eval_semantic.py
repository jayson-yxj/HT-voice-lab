"""Run the labelled semantic-turn fixture against a configured Groq model."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from groq_semantic import GroqSemanticJudge


STATUS = {True: "complete", False: "incomplete", None: "uncertain"}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--proxy")
    parser.add_argument("--minimum-interval", type=float, default=2.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cases = json.loads(args.fixture.read_text())
    judge = GroqSemanticJudge(args.model, args.timeout, proxy=args.proxy)
    previous = {}
    if args.resume and args.output.is_file():
        saved = json.loads(args.output.read_text())
        if saved.get("model") == args.model:
            previous = {row["id"]: row for row in saved.get("cases", [])
                        if row.get("error") is None}
    rows = []
    last_request = None
    for case in cases:
        if case["id"] in previous:
            rows.append(previous[case["id"]])
            print("{}: {} (cached)".format(case["id"], rows[-1]["actual"]), flush=True)
            continue
        context = [dict(id="{}-{:02d}".format(case["id"], index + 1), text=text,
                        start_s=float(index * 5), end_s=float(index * 5 + 3))
                   for index, text in enumerate(case["context"])]
        try:
            if last_request is not None:
                time.sleep(max(0, args.minimum_interval - (time.monotonic() - last_request)))
            last_request = time.monotonic()
            result = judge.assess(
                speaker="SPEAKER_00", contribution_ids=[item["id"] for item in context],
                text=context[-1]["text"], close_reason=case["close_reason"], context=context)
            actual = STATUS[result["turn_complete"]]
            rows.append(dict(
                id=case["id"], language=case["language"], expected=case["expected"],
                actual=actual, accepted=actual in case["expected"],
                confidence=result["confidence"], reason=result["reason"],
                latency_s=result["latency_s"], error=None))
        except Exception as exc:
            rows.append(dict(
                id=case["id"], language=case["language"], expected=case["expected"],
                actual=None, accepted=False, confidence=None, reason=None,
                latency_s=None, error="{}: {}".format(type(exc).__name__, exc)))
        print("{}: {}".format(case["id"], rows[-1]["actual"] or rows[-1]["error"]),
              flush=True)

    latencies = [row["latency_s"] for row in rows if row["latency_s"] is not None]
    summary = dict(
        fixture=str(args.fixture.resolve()), model=args.model, total=len(rows),
        accepted=sum(row["accepted"] for row in rows),
        accuracy=sum(row["accepted"] for row in rows) / len(rows),
        errors=sum(row["error"] is not None for row in rows),
        latency_s=(dict(mean=statistics.fmean(latencies), p50=percentile(latencies, 0.50),
                        p95=percentile(latencies, 0.95), maximum=max(latencies))
                   if latencies else None),
        cases=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("model", "total", "accepted", "accuracy", "errors", "latency_s")},
                     ensure_ascii=False, indent=2))
    print(args.output)
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    sys.exit(main())

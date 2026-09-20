import json
from pathlib import Path

from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate


def run(args):
    if args.collar < 0:
        raise ValueError("collar must be nonnegative")
    reference = load_rttm(args.reference)
    hypothesis = load_rttm(args.hypothesis)
    if len(reference) != 1 or len(hypothesis) > 1:
        raise ValueError("This command expects one recording in each RTTM")
    from pyannote.core import Annotation
    ref = next(iter(reference.values()))
    hyp = next(iter(hypothesis.values())) if hypothesis else Annotation(uri=ref.uri)
    if not len(ref):
        raise ValueError("Reference contains no speaker intervals")
    hyp.uri = ref.uri
    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    # Score the full reference/hypothesis extent, including false speech in silence.
    from pyannote.core import Segment, Timeline
    end = max(ref.get_timeline().extent().end, hyp.get_timeline().extent().end)
    details = metric(ref, hyp, uem=Timeline([Segment(0, end)]), detailed=True)
    result = {"collar_seconds": args.collar, "skip_overlap": args.skip_overlap,
              "reference": str(Path(args.reference).resolve()), "hypothesis": str(Path(args.hypothesis).resolve()),
              "scores": {k: float(v) for k, v in details.items()}}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0


def normalize_text(text):
    import unicodedata
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def run_asr(args):
    import jiwer
    ref = normalize_text(Path(args.reference).read_text())
    hyp = normalize_text(Path(args.hypothesis).read_text())
    if not ref:
        raise ValueError("Reference text is empty after normalization")
    scores = jiwer.process_characters(ref, hyp)
    result = dict(cer=scores.cer, substitutions=scores.substitutions,
                  deletions=scores.deletions, insertions=scores.insertions,
                  reference_characters=len(ref),
                  normalization="NFKC, lowercase, remove whitespace and Unicode punctuation; no number or traditional/simplified conversion")
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0

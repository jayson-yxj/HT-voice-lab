import math
import time

import numpy as np
import torch
from pyannote.core import SlidingWindow, SlidingWindowFeature

from common import ROOT, SR, Session, replay, sync


def run(args):
    from diart import SpeakerDiarization, SpeakerDiarizationConfig
    from diart.models import SegmentationModel, EmbeddingModel
    for name in ["segmentation", "embedding"]:
        if not (ROOT / "models" / name / "pytorch_model.bin").is_file():
            raise RuntimeError("Missing pyannote weights. Accept the model terms, run './lab login', then './lab fetch --only segmentation embedding'.")
    start = time.monotonic()
    config = SpeakerDiarizationConfig(
        segmentation=SegmentationModel.from_pyannote(str(ROOT / "models/segmentation/pytorch_model.bin")),
        embedding=EmbeddingModel.from_pyannote(str(ROOT / "models/embedding/pytorch_model.bin")),
        duration=5, step=0.5, latency=args.latency, max_speakers=8,
        device=torch.device(args.device))
    model = SpeakerDiarization(config)
    sync(args.device)
    session = Session(args, time.monotonic() - start,
                      dict(duration=5, step=0.5, latency=args.latency, max_speakers=8,
                           tau_active=config.tau_active, rho_update=config.rho_update,
                           delta_new=config.delta_new, startup="left zero padding", eof="right zero padding"))
    buffer = np.zeros(5 * SR, dtype=np.float32)
    segments = []
    virtual_end = 0.0

    def consume(audio, available, final_flush=False):
        nonlocal buffer, virtual_end
        start = time.monotonic()
        count = len(audio)
        buffer = np.concatenate((buffer[count:], audio))
        virtual_end += count / SR
        window = SlidingWindowFeature(buffer[:, None], SlidingWindow(start=virtual_end - 5,
                                      duration=1 / SR, step=1 / SR))
        with torch.inference_mode():
            annotation, waveform = model([window])[0]
        sync(args.device)
        output = []
        for span, _, label in annotation.itertracks(yield_label=True):
            a, b = max(0.0, span.start), min(session.duration, span.end)
            if b > a:
                output.append(dict(start=a, end=b, speaker=str(label)))
        segments.extend(output)
        committed = max(0.0, min(session.duration, waveform.extent.end))
        session.emit(available, committed, time.monotonic() - start,
                     final_flush=final_flush, segments=output)

    for audio, end, last in replay(args.audio, SR // 2, session):
        if len(audio) < SR // 2:
            audio = np.pad(audio, (0, SR // 2 - len(audio)))
        consume(audio, end, final_flush=last and end != virtual_end + 0.5)
    for _ in range(math.ceil(args.latency / 0.5)):
        consume(np.zeros(SR // 2, dtype=np.float32), session.duration, final_flush=True)
    session.finish(segments=segments)

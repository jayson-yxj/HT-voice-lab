"""Small checks for metrics, time boundaries and causal feature extraction."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from common import merge_segments, replay
from score import run as score_diarization, run_asr


def main():
    merged = merge_segments([
        dict(start=-0.2, end=1, speaker="A"),
        dict(start=1, end=2.5, speaker="A"),
        dict(start=0.5, end=1.5, speaker="B"),
    ], 2)
    assert merged == [dict(start=0.0, end=2, speaker="A"), dict(start=0.5, end=1.5, speaker="B")]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wave = np.arange(1601, dtype=np.float32) / 3200
        sf.write(root / "audio.wav", wave, 16000, subtype="FLOAT")
        waits = []
        blocks = list(replay(root / "audio.wav", 320, SimpleNamespace(wait_until=waits.append)))
        assert len(blocks) == 6 and blocks[-1][2] and len(blocks[-1][0]) == 1
        assert abs(waits[-1] - len(wave) / 16000) < 1e-9
        np.testing.assert_array_equal(np.concatenate([b[0] for b in blocks]), wave)
        ref, hyp, out = root / "ref.rttm", root / "hyp.rttm", root / "score.json"
        def rttm(a, b):
            return "SPEAKER test 1 0 1 <NA> <NA> {} <NA> <NA>\nSPEAKER test 1 1 1 <NA> <NA> {} <NA> <NA>\n".format(a, b)
        ref.write_text(rttm("A", "B"))
        for text, expected in [(rttm("X", "Y"), 0), (rttm("X", "X"), 0.5), ("", 1)]:
            hyp.write_text(text)
            args = SimpleNamespace(reference=str(ref), hypothesis=str(hyp), collar=0, skip_overlap=False, output=str(out))
            with contextlib.redirect_stdout(io.StringIO()):
                score_diarization(args)
            assert abs(json.loads(out.read_text())["scores"]["diarization error rate"] - expected) < 1e-9
        ref.write_text("小派，你觉得呢？")
        hyp.write_text("小派你觉得啊")
        with contextlib.redirect_stdout(io.StringIO()):
            run_asr(SimpleNamespace(reference=str(ref), hypothesis=str(hyp), output=str(out)))
        assert abs(json.loads(out.read_text())["cer"] - 1 / 6) < 1e-9
    print("PASS: segment bounds/overlap, replay tail, identity permutation/collapse/silence DER, Chinese CER")


def check_features():
    import torch
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
    from run_sortformer import feature_window
    pre = AudioToMelSpectrogramPreprocessor(sample_rate=16000, normalize="NA", window_size=0.025,
                                           window_stride=0.01, window="hann", features=128,
                                           n_fft=512, dither=0.0)
    model = SimpleNamespace(process_signal=lambda signal, length: pre(input_signal=signal, length=length))
    audio = np.random.default_rng(0).normal(0, 0.05, 48000).astype(np.float32)
    full, _ = model.process_signal(torch.from_numpy(audio)[None], torch.tensor([len(audio)]))
    for first, last in [(0, 104), (40, 152), (88, 200)]:
        raw_first = max(0, first - 3)
        wave = audio[raw_first * 160:last * 160 + 320]
        features, lengths = feature_window(model, wave, raw_first, first, last, "cpu")
        torch.testing.assert_close(features.transpose(1, 2), full[:, :, first:last], rtol=1e-5, atol=1e-5)
        assert lengths.item() == last - first
    print("PASS: bounded causal Sortformer mel windows match reference features")


if __name__ == "__main__":
    main()
    from test_conversation import check_conversation
    check_conversation()
    from test_dialogue_state import check_dialogue_state
    check_dialogue_state()
    from test_session_memory import check_session_memory
    check_session_memory()
    from test_personal_identity import check_personal_identity_fixture
    check_personal_identity_fixture()
    from test_participant_context import check_participant_context
    check_participant_context()
    check_features()

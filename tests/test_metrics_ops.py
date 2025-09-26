from __future__ import annotations

import numpy as np

from asr_eval.corruption.ops import noise_snr_db
from asr_eval.metrics.wer_cer import corpus_scores


def test_wer_zero():
    scores = corpus_scores(["hello world"], ["hello world"])
    assert scores.wer == 0.0
    assert scores.cer == 0.0
    assert scores.n_utts == 1


def test_noise_deterministic():
    wav = np.zeros(16_000, dtype=np.float32)
    out_a = noise_snr_db(wav, 16_000, snr_db=20, seed=123)
    out_b = noise_snr_db(wav, 16_000, snr_db=20, seed=123)
    assert np.allclose(out_a, out_b)

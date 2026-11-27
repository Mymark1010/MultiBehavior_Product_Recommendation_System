import numpy as np
import pandas as pd
import pytest

from rr.candidates import CandidateFilter, finalize_topk


@pytest.fixture
def flt():
    # vocabulary 10, 20, 30, 40; item 40 only known from t=500.
    # availability: 10 -> 0 at t=100, back to 1 at t=200; 20 always unknown; 30 -> 1 at t=50.
    return CandidateFilter(
        vocab_itemids=[10, 20, 30, 40], known_at_ms=[0, 0, 0, 500],
        avail_itemids=[10, 10, 30, 99], avail_ts=[100, 200, 50, 0], avail_values=[0, 1, 1, 0],
        use_availability=True)


def test_availability_as_of(flt):
    items = np.array([10, 10, 10, 10, 20, 30, 30, 99])
    times = np.array([99, 100, 199, 200, 1000, 49, 50, 1000])
    assert flt.availability(items, times).tolist() == [-1, 0, 0, 1, -1, -1, 1, -1]


def test_eligible_rules(flt):
    items = np.array([10, 10, 20, 40, 40, 99])
    times = np.array([150, 250, 0, 499, 500, 1000])
    # unavailable at 150; available at 250; unknown availability kept; known_at; out of vocab.
    assert flt.eligible(items, times).tolist() == [False, True, True, False, True, False]


def test_finalize_filters_dedupes_and_falls_back(flt):
    candidates = np.array([[10, 30, 30, 99, -1],
                           [40, 20, -1, -1, -1]])
    times = np.array([150, 600])
    topk, scores, fb = finalize_topk(candidates, times, flt, fallback_items=[20, 30, 10, 40], k=3)
    # row 0: 10 unavailable, 30 kept once, 99 OOV -> fallback 20 (30 already chosen, 10 unavailable at 150).
    assert topk[0].tolist() == [30, 20, -1]
    assert fb[0].tolist() == [False, True, False]
    # row 1: 40 known at 600, 20 kept, fallback adds 30.
    assert topk[1].tolist() == [40, 20, 30]
    assert fb[1].tolist() == [False, False, True]


def test_finalize_without_candidates(flt):
    topk, _, fb = finalize_topk(np.empty((2, 0), dtype=np.int64), np.array([0, 600]), flt,
                                fallback_items=[40, 10, 20], k=2)
    assert topk.tolist() == [[10, 20], [40, 10]] and fb.all()


def test_from_frames():
    vocab = pd.DataFrame({"itemid": [5, 6]})
    catalog = pd.DataFrame({"itemid": [5, 6, 7], "known_at_train_ms": [1, 2, 3]})
    history = pd.DataFrame({"itemid": [5, 6], "timestamp": [0, 0], "property": ["available", "categoryid"],
                            "value": ["0", "12"]})
    f = CandidateFilter.from_frames(vocab, catalog, history, "known_at_train_ms", use_availability=True)
    assert f.eligible(np.array([5, 6]), np.array([10, 10])).tolist() == [False, True]
    main = CandidateFilter.from_frames(vocab, catalog, history, "known_at_train_ms")
    assert main.eligible(np.array([5, 6]), np.array([10, 10])).tolist() == [True, True]
    assert main.eligible(np.array([5, 6]), np.array([1, 1])).tolist() == [True, False]  # known_at still applies


def test_chunking_is_transparent(flt):
    rng = np.random.default_rng(0)
    candidates = rng.choice([10, 20, 30, 40, 99, -1], size=(50, 4))
    candidates = np.sort(candidates, axis=1)[:, ::-1]  # -1 padding at the end
    times = rng.integers(0, 1000, 50)
    full = finalize_topk(candidates, times, flt, [10, 20, 30, 40], k=3, chunk=50)
    small = finalize_topk(candidates, times, flt, [10, 20, 30, 40], k=3, chunk=7)
    for a, b in zip(full, small):
        np.testing.assert_array_equal(a, b)


def test_matches_reference_loop_with_long_fallback():
    rng = np.random.default_rng(1)
    vocab = np.arange(1, 301)
    known = rng.integers(0, 500, len(vocab))
    f = CandidateFilter(vocab, known, avail_itemids=vocab[:50], avail_ts=np.zeros(50, int),
                        avail_values=np.zeros(50, int), use_availability=True)
    n, k = 40, 20
    cands = rng.integers(1, 400, (n, 30))
    cands[rng.random((n, 30)) < 0.3] = -1
    cands = np.array([np.r_[r[r != -1], r[r == -1]] for r in cands])  # padding at the end
    times = rng.integers(0, 1000, n)
    fallback = rng.permutation(vocab)  # 300 items > window
    topk, _, fb = finalize_topk(cands, times, f, fallback, k=k, chunk=7)
    for r in range(n):
        out, flags = [], []
        for src, is_fb in [(cands[r], False), (fallback, True)]:
            for it in src:
                if len(out) == k:
                    break
                if it == -1 or it in out or not f.eligible(np.array([it]), np.array([times[r]]))[0]:
                    continue
                out.append(it); flags.append(is_fb)
        assert topk[r][:len(out)].tolist() == out and (topk[r][len(out):] == -1).all()
        assert fb[r][:len(out)].tolist() == flags

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
        avail_itemids=[10, 10, 30, 99], avail_ts=[100, 200, 50, 0], avail_values=[0, 1, 1, 0])


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
    f = CandidateFilter.from_frames(vocab, catalog, history, "known_at_train_ms")
    assert f.eligible(np.array([5, 6]), np.array([10, 10])).tolist() == [False, True]

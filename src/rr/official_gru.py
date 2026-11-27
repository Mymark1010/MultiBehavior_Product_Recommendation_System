"""Bridge to the official GRU4Rec PyTorch code (Hidasi, hidasib/GRU4Rec_PyTorch_Official) for notebook 13.

The official code is NOT part of this repo (research/education licence, not redistributed): notebook 13
fetches the pinned commit into ``external/`` and :func:`load_official` checks the file hashes before
importing it. Nothing in the official code is modified; this module only

* turns project events into the official input frame (one row per event, ``SessionId, ItemId, Time``);
  events of one timestamp group are ordered by ``event_id`` -- a fake order the main protocol forbids,
  accepted by the user (2026-11-26) for this verification run only;
* repeats the body of ``GRU4Rec.fit`` one epoch at a time (same objects, same RNG order) so that the
  project's validation NDCG@20 can be measured after every epoch;
* runs validation prefixes through the official layers and returns the hidden state after the last
  in-vocabulary event of each prefix, so the project filter / fallback / metrics score the model exactly
  like notebook 07.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OFFICIAL_URL = "https://github.com/hidasib/GRU4Rec_PyTorch_Official.git"
OFFICIAL_COMMIT = "d1fc31105577665d2f105de1c8d9a38d223a9a22"
# SHA-256 of the files with line endings normalised to LF (a Windows checkout may convert them to CRLF)
OFFICIAL_SHA256 = {"gru4rec_pytorch.py": "d5b52620f76f36b13e229a53d11176d822d472270f3183fda25a034051e61278",
                   "evaluation.py": "2da286054b1317a73201f32709c4937364582e11416964b2d0a91355a5b42af4"}


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_official(code_dir: Path):
    """Import the pinned official modules from ``code_dir`` after checking their hashes."""
    code_dir = Path(code_dir)
    for name, digest in OFFICIAL_SHA256.items():
        got = lf_sha256(code_dir / name)
        if got != digest:
            raise ValueError(f"{name}: SHA-256 {got} khác bản đã ghim ({OFFICIAL_COMMIT[:7]}).")
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    gm = importlib.import_module("gru4rec_pytorch")
    gm.GRU4RecModel._init_numpy_weights = _init_numpy_weights_numpy1
    return gm, importlib.import_module("evaluation")


def _init_numpy_weights_numpy1(self, shape):
    """``GRU4RecModel._init_numpy_weights`` under NumPy 1 type promotion (the version the official code was tested with).

    The original line ``np.random.rand(*shape).astype('float32') * 2 * sigma - sigma`` stays float32 under
    NumPy 1, where the float64 scalar ``sigma`` is cast down to the array's type; NumPy 2 (NEP 50) promotes the
    result to float64 and ``Tensor.set_`` then refuses it. Casting ``sigma`` first gives the NumPy 1 values.
    """
    sigma = np.float32(np.sqrt(6.0 / (shape[0] + shape[1])))
    return np.random.rand(*shape).astype("float32") * np.float32(2) * sigma - sigma


def read_paramfile(path: Path) -> dict:
    """``gru4rec_params`` of an official paramfile, read as a literal (the file is not executed)."""
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "gru4rec_params" for t in node.targets):
            call = node.value
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "OrderedDict" and len(call.args) == 1:
                return dict(ast.literal_eval(call.args[0]))
    raise ValueError(f"{path}: không tìm thấy gru4rec_params = OrderedDict([...]).")


def to_official_frame(events: pd.DataFrame) -> pd.DataFrame:
    """One row per event, sorted by session, timestamp, then ``event_id`` inside a timestamp group."""
    frame = events.sort_values(["session_id", "timestamp", "event_id"], kind="stable")
    return pd.DataFrame({"SessionId": frame["session_id"].to_numpy(), "ItemId": frame["itemid"].to_numpy(),
                         "Time": frame["timestamp"].to_numpy()})


class OfficialTrainer:
    """``GRU4Rec.fit`` split into epochs: same objects, same order of random draws, same loop body."""

    def __init__(self, gm, params: dict, train: pd.DataFrame, seed: int, sample_cache_max_size: int = 10_000_000):
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cpu")
        gru = gm.GRU4Rec(**params, device=device)
        gru.error_during_train = False
        gru.data_iterator = gm.SessionDataIterator(train, gru.batch_size, n_sample=gru.n_sample, sample_alpha=gru.sample_alpha,
                                                   sample_cache_max_size=sample_cache_max_size, session_order="time",
                                                   device=device)
        if gru.logq and gru.loss == "cross-entropy":
            pop = train.groupby("ItemId").size()
            gru.P0 = torch.tensor(pop[gru.data_iterator.itemidmap.index.values], dtype=torch.float32, device=device)
        model = gm.GRU4RecModel(gru.data_iterator.n_items, gru.layers, gru.dropout_p_embed, gru.dropout_p_hidden,
                                gru.embedding, gru.constrained_embedding).to(device)
        model._reset_weights_to_compatibility_mode()
        gru.model = model
        self.gru, self.opt, self.epoch = gru, gm.IndexedAdagradM(model.parameters(), gru.learning_rate, gru.momentum), 0

    def run_epoch(self) -> dict:
        """The body of the official epoch loop (``GRU4Rec.fit``), unchanged."""
        gru, t0 = self.gru, time.time()
        H = [torch.zeros((gru.batch_size, n), dtype=torch.float32, requires_grad=False, device=gru.device) for n in gru.layers]
        c, cc = [], []
        n_valid = gru.batch_size
        reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
        for in_idx, out_idx in gru.data_iterator(enable_neg_samples=(gru.n_sample > 0), reset_hook=reset_hook):
            for h in H:
                h.detach_()
            gru.model.zero_grad()
            R = gru.model.forward(in_idx, H, out_idx, training=True)
            L = gru.loss_function(R, out_idx, n_valid) / gru.batch_size
            L.backward()
            self.opt.step()
            L = L.cpu().detach().numpy()
            c.append(L)
            cc.append(n_valid)
            if np.isnan(L):
                gru.error_during_train = True
                raise FloatingPointError(f"Epoch {self.epoch + 1}: loss NaN.")
        self.epoch += 1
        c, cc = np.array(c), np.array(cc)
        return {"epoch": self.epoch, "loss": float(np.sum(c * cc) / np.sum(cc)), "train_seconds": round(time.time() - t0, 1),
                "minibatches": int(len(c))}


@torch.no_grad()
def query_states(gru, events: pd.DataFrame, queries: pd.DataFrame) -> torch.Tensor:
    """Hidden state after the last in-vocabulary event of each prefix (zero state if it has none).

    ``events``: the evaluation split (``session_id, group_index, event_id, itemid, timestamp``);
    ``queries``: ``session_id, prefix_len_groups``. Out-of-vocabulary events are skipped, as in the official
    iterator. Sessions run in parallel, longest first, one event per step through the official layers.
    """
    model, imap = gru.model, gru.data_iterator.itemidmap
    ev = events[events["itemid"].isin(imap.index)].sort_values(["session_id", "timestamp", "event_id"], kind="stable")
    sess = ev["session_id"].to_numpy()
    lengths = pd.Series(sess).value_counts()
    order = lengths.index.to_numpy()                      # longest session first
    rank = pd.Series(np.arange(len(order)), index=order)
    pos = ev.groupby("session_id").cumcount().to_numpy()
    # ragged layout: the events at position t, ordered by session rank, are the inputs of step t
    # (sessions are ranked longest first, so the active sessions of step t are ranks 0 .. n_active[t] - 1)
    n_steps = int(lengths.max()) if len(lengths) else 0
    o = np.lexsort((rank.reindex(sess).to_numpy(), pos))
    step_items = imap.reindex(ev["itemid"].to_numpy()).to_numpy().astype(np.int64)[o]
    step_ptr = np.r_[0, np.cumsum(np.bincount(pos, minlength=n_steps))]
    # number of in-vocabulary prefix events of each query -> step at which its state is complete
    counts = ev.groupby(["session_id", "group_index"]).size().groupby(level=0).cumsum()
    q_keys = pd.MultiIndex.from_arrays([queries["session_id"].to_numpy(), queries["prefix_len_groups"].to_numpy()])
    known = counts.reindex(counts.index.union(q_keys)).groupby(level=0).ffill().fillna(0).reindex(q_keys).to_numpy()
    q_last = known.astype(np.int64) - 1
    q_rank = rank.reindex(queries["session_id"].to_numpy()).fillna(-1).to_numpy().astype(np.int64)
    out = torch.zeros((len(queries), gru.layers[-1]), dtype=torch.float32)
    H = [torch.zeros((len(order), n), dtype=torch.float32) for n in gru.layers]
    n_active = np.diff(step_ptr)
    ready = np.nonzero(q_last >= 0)[0]
    ready = ready[np.argsort(q_last[ready], kind="stable")]
    steps, starts = np.unique(q_last[ready], return_index=True)
    by_step = dict(zip(steps.tolist(), np.split(ready, starts[1:])))
    for t in range(n_steps):
        n = int(n_active[t])
        Hs = [h[:n] for h in H]
        X = torch.from_numpy(step_items[step_ptr[t]:step_ptr[t + 1]])
        E, _, _ = model.embed(X, Hs, None)
        if not (model.constrained_embedding or model.embedding):
            Hs[0] = E
        model.hidden_step(E, Hs, training=False)
        for h, hs in zip(H, Hs):
            h[:n] = hs
        rows = by_step.get(t)
        if rows is not None:
            out[torch.from_numpy(rows)] = H[-1][torch.from_numpy(q_rank[rows])]
    return out


@torch.no_grad()
def topn_items(gru, states: torch.Tensor, n: int, batch_size: int = 2048):
    """Top-``n`` item ids and scores (official scoring: state . Wy + By over the whole item map)."""
    model, ids = gru.model, gru.data_iterator.itemidmap.index.to_numpy()
    O, B = model.Wy.weight, model.By.weight          # what model.embed(X, H, None) returns as O, B in every mode
    items = np.empty((len(states), n), dtype=np.int64)
    scores = np.empty((len(states), n), dtype=np.float32)
    for i in range(0, len(states), batch_size):
        s, idx = torch.topk(model.score_items(states[i:i + batch_size], O, B), n, dim=1)
        items[i:i + batch_size], scores[i:i + batch_size] = ids[idx.numpy()], s.numpy()
    return items, scores

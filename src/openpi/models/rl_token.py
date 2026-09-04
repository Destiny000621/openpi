"""SubRL learned RL-token encoder — a trained readout over pi0.5's image tokens.

Replaces the mean-pool z_rl when ``SUBRL_RLTOKEN=<npz>`` is set at serve time
(policies/policy.py): a learned query cross-attends over the prefix image tokens
(2 layers, d=512, 8 heads) and projects back to 2048, so the learner/replay/agent
plumbing is untouched. Trained offline by ``scripts/train_rl_token.py`` with an
AR-style token-reconstruction bottleneck, wrist token block up-weighted
(reimplemented from the SubRL YAM earbud-era spec; the YAM reference code lives
only on that station — deviations: the reconstruction decoder predicts tokens
from (z, learned position query) instead of a full causal transformer).

Inference here is PURE NUMPY on the (already-extracted) float32 token embeddings
— it runs outside the jitted pi0.5 forward, costs ~1 ms, and keeps the npz the
single source of truth for both trainer and serve. ``train_rl_token.py`` asserts
torch-vs-numpy forward parity before exporting.

npz format (all float32):
  q0            (1, D)            learned query
  proj_in_w/b   (E, D) / (D,)
  For each layer i in 0..L-1:
    li_{qw,kw,vw,ow}  (D, D)      attention projections (row-vector convention:
                                   y = x @ w + b)
    li_{qb,kb,vb,ob}  (D,)
    li_ln1_{g,b}      (D,)        pre-attention LayerNorm on the query
    li_mlp1_w/b       (D, 4D)/(4D,)
    li_mlp2_w/b       (4D, D)/(D,)
    li_ln2_g/b        (D,)
  proj_out_w/b  (D, E) / (E,)
  meta          scalar json string: {"layers": L, "dim": D, "heads": H, "emb": E}
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-6) * g + b


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class RLTokenEncoder:
    """Numpy forward of the trained RL token (see module docstring for the npz)."""

    def __init__(self, weights: dict[str, np.ndarray], layers: int, dim: int, heads: int) -> None:
        self.w = {k: np.asarray(v, np.float32) for k, v in weights.items()}
        self.layers = layers
        self.dim = dim
        self.heads = heads

    def __call__(self, tokens: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """tokens [b, s, E] (+ bool mask [b, s]) -> z [b, E]."""
        w = self.w
        b, s, _ = tokens.shape
        d, h = self.dim, self.heads
        hd = d // h
        kv = tokens @ w["proj_in_w"] + w["proj_in_b"]  # [b, s, D]
        q = np.broadcast_to(w["q0"], (b, 1, d)).astype(np.float32)  # [b, 1, D]
        neg = np.float32(-1e9)
        attn_bias = None
        if mask is not None:
            attn_bias = np.where(mask[:, None, None, :], 0.0, neg)  # [b,1,1,s]
        for i in range(self.layers):
            p = f"l{i}_"
            qn = _ln(q, w[p + "ln1_g"], w[p + "ln1_b"])
            qh = (qn @ w[p + "qw"] + w[p + "qb"]).reshape(b, 1, h, hd).transpose(0, 2, 1, 3)
            kh = (kv @ w[p + "kw"] + w[p + "kb"]).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
            vh = (kv @ w[p + "vw"] + w[p + "vb"]).reshape(b, s, h, hd).transpose(0, 2, 1, 3)
            logits = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(hd)  # [b,h,1,s]
            if attn_bias is not None:
                logits = logits + attn_bias
            out = _softmax(logits) @ vh  # [b,h,1,hd]
            out = out.transpose(0, 2, 1, 3).reshape(b, 1, d) @ w[p + "ow"] + w[p + "ob"]
            q = q + out
            qn = _ln(q, w[p + "ln2_g"], w[p + "ln2_b"])
            mlp = np.maximum(qn @ w[p + "mlp1_w"] + w[p + "mlp1_b"], 0.0)
            q = q + (mlp @ w[p + "mlp2_w"] + w[p + "mlp2_b"])
        z = q[:, 0] @ w["proj_out_w"] + w["proj_out_b"]  # [b, E]
        return z.astype(np.float32)


def load_encoder(path: str) -> RLTokenEncoder:
    """Load the trained RL token npz; fails LOUDLY on a missing/invalid file
    (never silently fall back to the mean-pool — checkpoints are valid only with
    the npz they trained under)."""
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"SUBRL_RLTOKEN npz not found: {p}")
    data = dict(np.load(p, allow_pickle=False))
    meta = json.loads(str(data.pop("meta")))
    return RLTokenEncoder(
        data, layers=int(meta["layers"]), dim=int(meta["dim"]), heads=int(meta["heads"])
    )

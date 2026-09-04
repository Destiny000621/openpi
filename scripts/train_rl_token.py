"""Train the SubRL RL token on extracted pi0.5 image-token shards (wrist-weighted).

Objective (RLT eq.2-style bottleneck, reimplemented from the SubRL YAM spec): the
encoder compresses the 768 image tokens into one 2048-d z; a reconstruction head
(learned per-position queries + MLP on z) must rebuild every token from z alone.
The wrist token block gets ``--wrist-weight`` (default 4) in the loss — seating
evidence lives in the wrist view — and masked-out tokens (zeroed right wrist)
get 0. Exports the npz consumed by ``openpi.models.rl_token.load_encoder``
(serve: ``SUBRL_RLTOKEN=<npz>``), asserting torch-vs-numpy forward parity first.

Any npz change = new SubRL run lineage (z semantics change).

Run (torch venv; CPU works, GPU faster):
    ~/venvs/subrl/bin/python ~/Desktop/openpi/scripts/train_rl_token.py \
        --data ~/Desktop/SubRL/data/franka_rl_token_corpus \
        --out  ~/Desktop/SubRL/checkpoints/rl_token/franka_cable_v1.npz
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import numpy as np
import tyro

VIEW_NAMES = ("base", "left_wrist", "right_wrist")  # FrankaInputs image order


@dataclasses.dataclass
class Args:
    data: str = "~/Desktop/SubRL/data/franka_rl_token_corpus"
    out: str = "~/Desktop/SubRL/checkpoints/rl_token/franka_cable_v1.npz"
    layers: int = 2
    dim: int = 512
    heads: int = 8
    steps: int = 4000
    batch: int = 64
    lr: float = 3e-4
    wrist_weight: float = 4.0
    device: str = "cuda"
    seed: int = 0


def main(args: Args) -> None:  # noqa: PLR0915
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    data_dir = pathlib.Path(args.data).expanduser()
    shards = sorted(data_dir.glob("shard_*.npz"))
    assert shards, f"no shards under {data_dir} — run extract_prefix_tokens.py first"
    # Corpus stays float16 in RAM (float32 doubles it and OOMs first); batches are
    # cast to float32 on-device.
    tokens = np.concatenate([np.load(s)["tokens"] for s in shards])
    assert tokens.dtype == np.float16, tokens.dtype
    mask = np.load(shards[0])["mask"].astype(bool)
    n, s, e = tokens.shape
    print(f"{n} frames, {s} tokens x {e} dims; {int(mask.sum())} unmasked tokens/frame")

    per_view = s // len(VIEW_NAMES)
    loss_w = np.ones(s, np.float32)
    loss_w[per_view : 2 * per_view] = args.wrist_weight  # wrist block
    loss_w[~mask] = 0.0
    d, h = args.dim, args.heads

    dev = args.device if torch.cuda.is_available() else "cpu"

    class Encoder(nn.Module):
        """Torch mirror of openpi.models.rl_token.RLTokenEncoder (must stay exact)."""

        def __init__(self) -> None:
            super().__init__()
            self.q0 = nn.Parameter(torch.randn(1, d) * 0.02)
            self.proj_in = nn.Linear(e, d)
            self.blocks = nn.ModuleList()
            for _ in range(args.layers):
                blk = nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(d, eps=1e-6),
                        "q": nn.Linear(d, d),
                        "k": nn.Linear(d, d),
                        "v": nn.Linear(d, d),
                        "o": nn.Linear(d, d),
                        "ln2": nn.LayerNorm(d, eps=1e-6),
                        "mlp1": nn.Linear(d, 4 * d),
                        "mlp2": nn.Linear(4 * d, d),
                    }
                )
                self.blocks.append(blk)
            self.proj_out = nn.Linear(d, e)

        def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            b, seq, _ = x.shape
            hd = d // h
            kv = self.proj_in(x)
            q = self.q0.expand(b, 1, d)
            bias = torch.where(m[:, None, None, :], 0.0, -1e9)
            for blk in self.blocks:
                qn = blk["ln1"](q)
                qh = blk["q"](qn).view(b, 1, h, hd).transpose(1, 2)
                kh = blk["k"](kv).view(b, seq, h, hd).transpose(1, 2)
                vh = blk["v"](kv).view(b, seq, h, hd).transpose(1, 2)
                att = torch.softmax(qh @ kh.transpose(-2, -1) / hd**0.5 + bias, dim=-1)
                out = (att @ vh).transpose(1, 2).reshape(b, 1, d)
                q = q + blk["o"](out)
                qn = blk["ln2"](q)
                q = q + blk["mlp2"](torch.relu(blk["mlp1"](qn)))
            return self.proj_out(q[:, 0])

    class Decoder(nn.Module):
        """Reconstruction head: per-position queries + MLP on z -> token_i."""

        def __init__(self) -> None:
            super().__init__()
            self.pos = nn.Parameter(torch.randn(s, d) * 0.02)
            self.z_proj = nn.Linear(e, d)
            self.mlp = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, e))

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            b = z.shape[0]
            zc = self.z_proj(z)[:, None, :].expand(b, s, d)
            pc = self.pos[None].expand(b, s, d)
            return self.mlp(torch.cat([zc, pc], dim=-1))

    enc, dec = Encoder().to(dev), Decoder().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    x_all = torch.from_numpy(tokens)  # fp16, CPU
    m = torch.from_numpy(mask)[None].to(dev)
    w = torch.from_numpy(loss_w)[None, :, None].to(dev)

    for step in range(args.steps):
        idx = torch.randint(0, n, (args.batch,))
        x = x_all[idx].to(dev).float()
        z = enc(x, m.expand(len(idx), -1))
        recon = dec(z)
        loss = (w * (recon - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  weighted recon MSE {loss.item():.5f}")

    # ---- export + torch-vs-numpy parity gate ----
    weights: dict[str, np.ndarray] = {
        "q0": enc.q0.detach().cpu().numpy(),
        "proj_in_w": enc.proj_in.weight.T.detach().cpu().numpy(),
        "proj_in_b": enc.proj_in.bias.detach().cpu().numpy(),
        "proj_out_w": enc.proj_out.weight.T.detach().cpu().numpy(),
        "proj_out_b": enc.proj_out.bias.detach().cpu().numpy(),
    }
    for i, blk in enumerate(enc.blocks):
        p = f"l{i}_"
        for short, lin in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")):
            weights[p + short + "w"] = blk[lin].weight.T.detach().cpu().numpy()
            weights[p + short + "b"] = blk[lin].bias.detach().cpu().numpy()
        weights[p + "ln1_g"] = blk["ln1"].weight.detach().cpu().numpy()
        weights[p + "ln1_b"] = blk["ln1"].bias.detach().cpu().numpy()
        weights[p + "ln2_g"] = blk["ln2"].weight.detach().cpu().numpy()
        weights[p + "ln2_b"] = blk["ln2"].bias.detach().cpu().numpy()
        weights[p + "mlp1_w"] = blk["mlp1"].weight.T.detach().cpu().numpy()
        weights[p + "mlp1_b"] = blk["mlp1"].bias.detach().cpu().numpy()
        weights[p + "mlp2_w"] = blk["mlp2"].weight.T.detach().cpu().numpy()
        weights[p + "mlp2_b"] = blk["mlp2"].bias.detach().cpu().numpy()

    sys.path.insert(0, str(pathlib.Path("~/Desktop/openpi/src").expanduser()))
    from openpi.models.rl_token import RLTokenEncoder  # numpy reference forward

    np_enc = RLTokenEncoder(weights, layers=args.layers, dim=d, heads=h)
    probe = tokens[:4].astype(np.float32)
    with torch.no_grad():
        z_t = enc(torch.from_numpy(probe).to(dev), m.expand(4, -1)).cpu().numpy()
    z_n = np_enc(probe, np.broadcast_to(mask, (4, s)))
    err = np.abs(z_t - z_n).max()
    assert err < 1e-2, f"torch-vs-numpy forward mismatch: max |dz| = {err}"
    print(f"parity gate: max |z_torch - z_numpy| = {err:.2e}  OK")

    out = pathlib.Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        **weights,
        meta=json.dumps({"layers": args.layers, "dim": d, "heads": h, "emb": e}),
    )
    print(f"wrote {out}\nServe with: SUBRL_RETURN_EMBED=1 SUBRL_RLTOKEN={out}")
    print("REMINDER: a new npz = a NEW SubRL run lineage (fresh buffer + learner).")


if __name__ == "__main__":
    main(tyro.cli(Args))

#!/usr/bin/env python3
"""Offline gates for the DSRL noise/prefix-rep wire layer — NO ROBOT REQUIRED.

DSRL (`nakamotoo/dsrl_pi0`, CoRL 2025) does RL in the *flow-matching noise* of a
frozen pi0.5: SAC picks the noise vector, the VLA turns it into an action chunk.
That only works if (a) the serve honours a caller-supplied noise deterministically
and (b) the noise the client sends still lands the policy on its SFT behaviour.
This script proves both against a **live serve** using **real demo frames** from
`double_cable_100_r6_v21`, before any of it is pointed at the arm.

Gates
  G1 determinism   same obs + same noise twice -> identical chunk; noise=None -> differs
  G2 back-compat   the stock (unpatched) openpi_client still gets a valid chunk
  G3 prefix_rep    shape/dtype/finiteness/determinism + the embedding width to
                   hardwire into the SAC state_dim (upstream's hardcoded 2032 is
                   wrong twice over — never trust the constant, measure it)
  G4 fidelity      chunk-vs-EXPERT MAE under three noise regimes. This is the gate
                   that matters. Upstream DSRL's SAC action is ONE (1, 32) noise row
                   TILED across the horizon -- in the aloha lineage that is the tile
                   to 50 at `train_utils_sim.py:233-242`, which is the lineage this
                   pi0.5-SFT belongs to (H=50 absolute chunked actions), NOT the
                   pi0-DROID one (H=10 joint velocity). The YAM port MEASURED that
                   tiling to break pi0.5-SFT (0.45 rad joint MAE vs expert, against
                   0.011 for iid noise). If `tiled` is much worse than `iid50` here,
                   DSRL on this checkpoint must run with noise_rows = action_horizon,
                   i.e. a (50, 32) SAC action space -- a DummyEnv Box shape change,
                   not an algorithm change (jaxrl2 reads action_chunk_shape from it).
  G5 latency       get_prefix_rep vs infer, per decision (DSRL pays both)

Usage (pixi shell -e droid-openpi), with the wcrop serve up on :8111:
    python scripts/dsrl_verify_noise.py
    python scripts/dsrl_verify_noise.py --episodes 3 --frames 6 --samples 8
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

# pi0.5 flow-matching latent: Pi0Config(pi05=True) defaults action_horizon=50,
# action_dim=32. Dims 10..31 are PAD dims of our rot6d10 action space but they are
# causally live inside the model (32-wide action_in_proj + joint Euler integration)
# — never zero them.
ACTION_HORIZON = 50
NOISE_DIM = 32
# Executed action space: [xyz(3), rot6d(6), gripper(1)] absolute, rot6d raw.
ACTION_DIM = 10
PROMPT = "Unplug the two cables from the right router, then insert them into the left router"
# The SFT demo set, for G4's expert chunks. First hit wins; override with --dataset.
DATASET_CANDIDATES = (
    pathlib.Path.home() / "Desktop/openpi/double_cable_100_r6_v21",           # avant-pc04
    pathlib.Path("/mnt/localssd/Sichang/lerobot_home/local/double_cable_100_r6_v21"),  # H200
    pathlib.Path.home() / ".cache/huggingface/lerobot/local/double_cable_100_r6_v21",
)


def _default_dataset() -> pathlib.Path:
    for cand in DATASET_CANDIDATES:
        if (cand / "meta/info.json").exists():
            return cand
    return DATASET_CANDIDATES[0]


def _decode_frames(video: pathlib.Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Decode the requested frame indices from a LeRobot mp4 as RGB uint8 HWC."""
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video))
    out: dict[int, np.ndarray] = {}
    try:
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {idx} of {video}")
            out[idx] = np.ascontiguousarray(frame[..., ::-1])  # BGR -> RGB
    finally:
        cap.release()
    return out


def _build_request(side_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray) -> dict:
    """The EXACT wire dict FrankaEEPi05Agent sends for the wcrop checkpoint.

    wcrop contract: the wrist frame goes out RAW (the serve applies the trained
    fractional crop (0.339, 0.17, 0.761, 0.92) itself and rejects a pre-resized
    224x224 wrist); the side frame is pad-resized client-side.
    """
    from openpi_client import image_tools  # noqa: PLC0415

    return {
        "observation/image": image_tools.resize_with_pad(side_rgb, 224, 224),
        "observation/wrist_image": np.ascontiguousarray(wrist_rgb),
        "observation/state": np.asarray(state, np.float32),
        "prompt": PROMPT,
    }


def _mae(chunk: np.ndarray, expert: np.ndarray) -> tuple[float, float, float]:
    """(position MAE in metres, rot6d MAE, gripper MAE in rad) over the horizon."""
    n = min(len(chunk), len(expert))
    d = np.abs(chunk[:n, :ACTION_DIM] - expert[:n, :ACTION_DIM])
    return float(d[:, :3].mean()), float(d[:, 3:9].mean()), float(d[:, 9].mean())


def _connect(host: str, port: int):
    """The avantbot DSRL client if this box has avantbot, else the patched openpi one.

    Both speak the same envelope. The fallback lets this script run on a GPU box
    (e.g. the H200) that has openpi but no robot stack — the gates test the SERVE,
    not the robot.
    """
    import sys  # noqa: PLC0415

    avantbot = pathlib.Path(__file__).resolve().parents[1] / "vendor/avantbot"
    if avantbot.exists():
        sys.path.insert(0, str(avantbot))
        try:
            from avantbot.policies.pi0.dsrl_client import DsrlWebsocketClient  # noqa: PLC0415

            print("client: avantbot DsrlWebsocketClient")
            return DsrlWebsocketClient(host, port)
        except Exception as exc:  # noqa: BLE001
            print(f"client: avantbot unavailable ({exc.__class__.__name__}), using openpi_client")

    from openpi_client import websocket_client_policy  # noqa: PLC0415

    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    if not hasattr(client, "get_prefix_rep"):
        raise SystemExit(
            "openpi_client has no get_prefix_rep — this checkout predates the DSRL "
            "wire layer. Install the openpi-client from the Franka_DSRL branch."
        )

    class _Shim:
        """Flattens get_prefix_rep to a 1-D vector, matching the avantbot client."""

        def __init__(self, inner):
            self._inner = inner

        def get_server_metadata(self):
            return self._inner.get_server_metadata()

        def infer(self, obs, *, noise=None):
            return self._inner.infer(obs) if noise is None else self._inner.infer(obs, noise=noise)

        def get_prefix_rep(self, obs):
            out = self._inner.get_prefix_rep(obs)
            rep = out["prefix_rep"] if isinstance(out, dict) else out
            return np.asarray(rep, np.float32).reshape(-1)

    print("client: openpi_client.WebsocketClientPolicy (DSRL envelope)")
    return _Shim(client)


def main() -> int:  # noqa: PLR0915, PLR0912, C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8111)
    ap.add_argument("--dataset", type=pathlib.Path, default=_default_dataset())
    ap.add_argument("--episodes", type=int, default=2, help="demo episodes to sample frames from")
    ap.add_argument("--frames", type=int, default=4, help="frames per episode")
    ap.add_argument("--samples", type=int, default=4, help="noise draws per frame per regime (G4)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    client = _connect(args.host, args.port)
    print(f"serve metadata: {client.get_server_metadata()}\n")

    # ---- load real demo frames + expert action chunks ------------------------
    import pandas as pd  # noqa: PLC0415

    cases: list[tuple[str, dict, np.ndarray]] = []  # (label, request, expert chunk)
    for ep in range(args.episodes):
        pq = args.dataset / f"data/chunk-000/episode_{ep:06d}.parquet"
        if not pq.exists():
            raise SystemExit(f"missing {pq} — point --dataset at the local LeRobot set")
        df = pd.read_parquet(pq)
        states = np.stack(df["observation.state"].to_numpy())
        actions = np.stack(df["action"].to_numpy())
        n = len(df) - ACTION_HORIZON
        idx = sorted(rng.choice(np.arange(10, n), size=args.frames, replace=False).tolist())
        wrist = _decode_frames(args.dataset / f"videos/chunk-000/observation.images.camera0/episode_{ep:06d}.mp4", idx)
        side = _decode_frames(args.dataset / f"videos/chunk-000/observation.images.camera1/episode_{ep:06d}.mp4", idx)
        for t in idx:
            cases.append(
                (
                    f"ep{ep}/f{t}",
                    _build_request(side[t], wrist[t], states[t]),
                    actions[t : t + ACTION_HORIZON],
                )
            )
    print(f"loaded {len(cases)} demo frames from {args.episodes} episodes\n")

    label0, req0, expert0 = cases[0]
    failures: list[str] = []

    # ---- G1 determinism ------------------------------------------------------
    print("=" * 78)
    print("G1  determinism of caller-supplied noise")
    noise = rng.standard_normal((ACTION_HORIZON, NOISE_DIM)).astype(np.float32)
    a1 = np.asarray(client.infer(dict(req0), noise=noise)["actions"])
    a2 = np.asarray(client.infer(dict(req0), noise=noise)["actions"])
    same = float(np.abs(a1 - a2).max())
    b1 = np.asarray(client.infer(dict(req0))["actions"])
    b2 = np.asarray(client.infer(dict(req0))["actions"])
    free = float(np.abs(b1 - b2).max())
    print(f"    chunk shape                 : {a1.shape}  (expect ({ACTION_HORIZON}, {ACTION_DIM}))")
    print(f"    same noise, max|delta|      : {same:.3e}   (expect exactly 0)")
    print(f"    server-drawn, max|delta|    : {free:.3e}   (expect > 0)")
    if a1.shape != (ACTION_HORIZON, ACTION_DIM):
        failures.append(f"G1 chunk shape {a1.shape}")
    if same != 0.0:
        failures.append(f"G1 same-noise chunks differ by {same:.3e}")
    if free == 0.0:
        failures.append("G1 server-drawn noise produced identical chunks (RNG not advancing)")
    print(f"    -> {'PASS' if not failures else 'FAIL'}")

    # ---- G2 backward compatibility ------------------------------------------
    print("=" * 78)
    print("G2  unpatched openpi_client still works against the patched serve")
    try:
        from openpi_client import websocket_client_policy  # noqa: PLC0415

        stock = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        c = np.asarray(stock.infer(dict(req0))["actions"])
        ok = c.shape == (ACTION_HORIZON, ACTION_DIM) and np.isfinite(c).all()
        print(f"    plain-dict chunk            : {c.shape}, finite={np.isfinite(c).all()}")
        if not ok:
            failures.append("G2 stock client got a bad chunk")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"G2 stock client raised {exc!r}")
        print(f"    ERROR: {exc!r}")
        ok = False
    print(f"    -> {'PASS' if ok else 'FAIL'}")

    # ---- G3 prefix_rep -------------------------------------------------------
    print("=" * 78)
    print("G3  get_prefix_rep (DSRL z_rl)")
    z1 = client.get_prefix_rep(dict(req0))
    z2 = client.get_prefix_rep(dict(req0))
    zother = client.get_prefix_rep(dict(cases[-1][1]))
    print(f"    shape / dtype               : {z1.shape} / {z1.dtype}")
    print(f"    finite                      : {bool(np.isfinite(z1).all())}")
    print(f"    repeat max|delta|           : {float(np.abs(z1 - z2).max()):.3e}   (expect 0)")
    print(f"    differs across observations : {float(np.abs(z1 - zother).max()):.3e}   (expect > 0)")
    print(f"    norm                        : {float(np.linalg.norm(z1)):.3f}")
    print(f"    >>> SAC state_dim = {ACTION_DIM} proprio + {z1.shape[0]} embed = {ACTION_DIM + z1.shape[0]}")
    if z1.dtype != np.float32 or z1.ndim != 1:
        failures.append(f"G3 prefix_rep is {z1.dtype} {z1.shape}")
    if not np.isfinite(z1).all():
        failures.append("G3 prefix_rep not finite")
    if float(np.abs(z1 - z2).max()) != 0.0:
        failures.append("G3 prefix_rep not deterministic")
    if float(np.abs(z1 - zother).max()) == 0.0:
        failures.append("G3 prefix_rep identical across different observations (constant feature!)")
    print(f"    -> {'PASS' if not [f for f in failures if f.startswith('G3')] else 'FAIL'}")

    # ---- G4 fidelity vs expert ----------------------------------------------
    print("=" * 78)
    print("G4  chunk-vs-expert MAE by noise regime  (the noise_rows decision)")
    print("    server : serve draws its own iid noise         = stock SFT behaviour")
    print("    iid50  : client sends iid (50,32)              = same distribution, our wire")
    print(f"    tiled  : client sends ONE (32,) row x{ACTION_HORIZON}      = upstream DSRL's action space")
    print("    expert : dataset action[t:t+50], absolute [xyz, rot6d, gripper]")
    regimes = ("server", "iid50", "tiled")
    acc: dict[str, list[tuple[float, float, float]]] = {r: [] for r in regimes}
    for _label, req, expert in cases:
        for _ in range(args.samples):
            row = rng.standard_normal((1, NOISE_DIM)).astype(np.float32)
            payloads = {
                "server": None,
                "iid50": rng.standard_normal((ACTION_HORIZON, NOISE_DIM)).astype(np.float32),
                "tiled": np.repeat(row, ACTION_HORIZON, axis=0),
            }
            for r in regimes:
                chunk = np.asarray(client.infer(dict(req), noise=payloads[r])["actions"])
                acc[r].append(_mae(chunk, expert))
    print()
    print(f"    {'regime':<8}{'pos MAE (mm)':>16}{'rot6d MAE':>14}{'grip MAE (rad)':>18}")
    means = {}
    for r in regimes:
        arr = np.asarray(acc[r])
        means[r] = arr.mean(axis=0)
        print(
            f"    {r:<8}{means[r][0] * 1000:>13.2f} +-{arr[:, 0].std() * 1000:>4.1f}"
            f"{means[r][1]:>14.4f}{means[r][2]:>18.4f}"
        )
    ratio = means["tiled"][0] / max(means["iid50"][0], 1e-9)
    print()
    print(f"    tiled/iid50 position-MAE ratio : {ratio:.2f}x")
    if ratio > 1.5:
        print("    >>> TILED NOISE DEGRADES THIS CHECKPOINT — run DSRL with noise_rows=50")
        print("        (full-chunk latent). A (1,32) SAC action would steer a policy")
        print("        that is already off its SFT manifold before RL starts.")
    else:
        print("    >>> tiled noise is comparable to iid — upstream's (1,32) action space is safe.")
    drift = means["iid50"][0] / max(means["server"][0], 1e-9)
    print(f"    iid50/server position-MAE ratio: {drift:.3f}   (expect ~1.0 — same distribution)")
    if not 0.8 < drift < 1.25:
        failures.append(f"G4 client-supplied iid noise shifts behaviour ({drift:.2f}x vs server-drawn)")
    print(f"    -> {'PASS' if not [f for f in failures if f.startswith('G4')] else 'FAIL'}")

    # ---- G5 latency ----------------------------------------------------------
    print("=" * 78)
    print("G5  per-decision latency (DSRL pays prefix_rep + infer at every replan)")
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        client.get_prefix_rep(dict(req0))
    t_rep = (time.perf_counter() - t0) / reps * 1000
    t0 = time.perf_counter()
    for _ in range(reps):
        client.infer(dict(req0), noise=noise)
    t_inf = (time.perf_counter() - t0) / reps * 1000
    print(f"    get_prefix_rep              : {t_rep:7.1f} ms")
    print(f"    infer (with noise)          : {t_inf:7.1f} ms")
    print(f"    total per decision          : {t_rep + t_inf:7.1f} ms")
    # Both upstream lineages decide once per FULL chunk (aloha --query_freq 50 of
    # horizon 50; real-DROID --query_freq 10 of horizon 10), so the Franka budget is
    # query_freq/30 Hz = 1.67 s at query_freq=50.
    budget_ms = ACTION_HORIZON / 30.0 * 1000
    print(f"    -> replan budget at 30 Hz, query_freq={ACTION_HORIZON}: {budget_ms:.0f} ms "
          f"({'OK' if t_rep + t_inf < 0.5 * budget_ms else 'TIGHT'})")

    print("=" * 78)
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

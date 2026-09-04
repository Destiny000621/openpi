"""Seed the SubRL replay with SUCCESSFUL insert segments from the wcrop eval rollouts.

For every successful insert attempt in data_log_eval_wcrop, this builds RLTTransition
windows byte-compatible with the live loop's pushes and feeds them through the normal
replay wire (`/extend`), so the learner treats them exactly like robot episodes
(their `done` terminals count toward the 25-episode warmup gate):

  - segment  = the LIVE selector's entry rule (gripper-held band + >=60 mm EE
               displacement since the grasp = plug physically out of the right
               router) ... release at the left router + a short tail. The unplug
               prefix is cut.
  - z_rl / ref_chunk = recomputed by the FROZEN policy itself: one infer() per
               window anchor (~every anchor_stride ticks, the live replan cadence)
               on the recorded frames + state -> the serve's wrist-weighted
               embedding and (50, 10) chunk. Requires SUBRL_RETURN_EMBED=1 and
               SUBRL_EMBED_WEIGHTS to be set to the SAME values as the live serve.
  - action_chunk = zeros (pure pi0.5 rollouts: the executed residual was zero;
               success-BC anchors the actor to pi0.5 on states where it succeeds).
  - reward 1 on the terminal window ONLY for candidates the pair-diff VLM verifier
               (benched zero-false-success) confirms with 2 votes; unverified
               candidates are SKIPPED entirely.

Run with the SERVE STOPPED (needs the GPU) and the LEARNER STOPPED (the script
spawns its own replay on the target journal, pushes, verifies /stats, and exits):

    cd ~/Desktop/openpi
    SUBRL_RETURN_EMBED=1 SUBRL_EMBED_WEIGHTS=1,4,1 \
    SUBRL_RLT_SRC=~/Desktop/openpi-RLT/rlt_online_rl/src \
    uv run python scripts/seed_franka_insert_replay.py \
        --journal ~/subrl_runs/dryrun_v0/replay_journal.pkl
"""

import base64
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import cv2
import numpy as np
import requests
import tyro

WRIST_CROP = (0.339, 0.17, 0.761, 0.92)
PROMPT = "Unplug the two cables from the right router, then insert them into the left router"
Z, C, A = 2048, 50, 10


@dataclasses.dataclass
class Args:
    root: str = "/home/boyuan/Desktop/Haply_Franka/data_log_eval_wcrop"
    push: bool = False  # default = PREVIEW ONLY: dump frames + verdicts for human review
    review_dir: str = "~/Desktop/SubRL/data/franka_seed_review"
    include: tuple[str, ...] = ()  # push stage: seed ONLY these segment names (empty = all verified)
    exclude: tuple[str, ...] = ()  # push stage: drop these segment names after review
    journal: str = "~/subrl_runs/dryrun_v0/replay_journal.pkl"
    config: str = "pi05_franka_double_cable_100_r6_rawrot_wcrop"
    dir: str = "~/.cache/openpi/hf/pi05_franka_double_cable_100_wcrop_10k"
    artifacts: str = "~/Desktop/SubRL/real_robot/artifacts/franka_cable_insert"
    replay_port: int = 9102
    anchor_stride: int = 15  # ticks between windows (live replan cadence)
    hold_w: float = 0.35
    open_w: float = 0.9
    min_extract_mm: float = 60.0
    tail_ticks: int = 10  # keep a short post-release tail so the seat is in-window
    llm_model: str = "gemini-3.7-flash"


# ---------------------------------------------------------------- mining


def _spans(mask: np.ndarray, min_len: int = 30) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _candidates(args: Args) -> list[dict]:
    """Successful-looking insert attempts: hold span -> displaced -> release."""
    cands = []
    for ep in sorted(pathlib.Path(args.root).expanduser().glob("*/episode_*")):
        s = np.load(ep / "arm0_states.npz")
        g = s["gripper_pos"][:, 0]
        pos = s["ee_pose"][:, 4:7]
        ts = np.load(ep / "timestamps.npy")
        wts = np.load(ep / "wrist_timestamps.npy") / 1000.0
        for a, b in _spans(g < args.hold_w):
            if b >= len(g) or g[b : b + 90].max(initial=0.0) < args.open_w:
                continue  # no release after this hold
            disp = np.linalg.norm(pos - pos[a], axis=1) * 1000.0
            entry = next((t for t in range(a, b) if disp[t] >= args.min_extract_mm), None)
            if entry is None:
                # Mid-task recordings (172731 sessions) START with the cable already
                # held above the left router — no unplug inside the recording, so
                # the displacement never fires. The segment starts at the hold start.
                if a > 10:
                    continue  # a real in-recording grasp that never got 60 mm clear
                entry = a
            if b - entry < 30:
                continue
            cands.append(
                {
                    "ep": ep,
                    "name": f"{ep.parent.name[-6:]}_{ep.name[-4:]}",
                    "start": int(entry),
                    "end": int(min(b + args.tail_ticks, len(ts) - 1)),
                    "ts": ts,
                    "wts": wts,
                    "states": s,
                }
            )
    return cands


def _frame(ep: pathlib.Path, cam: str, tick: int, ts: np.ndarray, cts: np.ndarray) -> np.ndarray:
    idx = int(np.argmin(np.abs(cts - ts[min(tick, len(ts) - 1)])))
    cap = cv2.VideoCapture(str(ep / f"{cam}.mp4"))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"frame read failed: {ep} {cam} #{idx}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _state10(states: np.lib.npyio.NpzFile, i: int) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    ee = states["ee_pose"][i]
    r6 = Rotation.from_quat([ee[1], ee[2], ee[3], ee[0]]).as_matrix()[:, :2].T.reshape(-1)
    grip = float(np.clip(1.0 - states["gripper_pos"][i, 0], 0.0, 1.0)) * 0.7929
    return np.concatenate([ee[4:7], r6, [grip]]).astype(np.float32)


# ---------------------------------------------------------------- VLM verification


def _crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = WRIST_CROP
    return img[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)]


def _seated_prompt(artifacts: pathlib.Path) -> str:
    ns: dict = {}
    exec(compile((artifacts / "primitives.py").read_text(), "primitives.py", "exec"), ns)  # noqa: S102
    return ns["SEATED_HELD_PROMPT"]


def _vlm_pair(args: Args, key: str, prompt: str, before: np.ndarray, after: np.ndarray) -> str:
    def url(img: np.ndarray) -> str:
        img = _crop(img)
        h, w = img.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        assert ok
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    body = {
        "model": args.llm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url(before)}},
                    {"type": "image_url", "image_url": {"url": url(after)}},
                ],
            }
        ],
        "max_tokens": 2000,
        "temperature": 0.0,
    }
    for attempt in (0, 1):
        try:
            r = requests.post(
                "https://litellm.avantrobotics.ai/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {key}"},
                timeout=30,
                verify=False,  # noqa: S501
            )
            if r.status_code in (401, 403, 429) or r.status_code >= 500:
                raise RuntimeError(f"gateway {r.status_code}")
            r.raise_for_status()
            m = re.search(r"\{.*\}", r.json()["choices"][0]["message"]["content"], re.DOTALL)
            return "parse_error" if m is None else json.loads(m.group(0)).get("result", "?")
        except Exception:  # noqa: BLE001, PERF203
            if attempt:
                return "error"
            time.sleep(2.0)
    return "error"


# ---------------------------------------------------------------- main


def main(args: Args) -> None:  # noqa: PLR0915
    import urllib3

    urllib3.disable_warnings()
    if args.push:  # the frozen-policy pass must reproduce the LIVE serve's z_rl
        assert os.environ.get("SUBRL_RETURN_EMBED") == "1", "run with SUBRL_RETURN_EMBED=1"
        assert os.environ.get("SUBRL_EMBED_WEIGHTS"), (
            "run with SUBRL_EMBED_WEIGHTS matching the live serve (e.g. 1,4,1)"
        )
    journal = pathlib.Path(args.journal).expanduser()
    journal.parent.mkdir(parents=True, exist_ok=True)
    artifacts = pathlib.Path(args.artifacts).expanduser()
    llm_key = (pathlib.Path.home() / ".subrl_llmkey").read_text().strip()
    prompt = _seated_prompt(artifacts)

    # 1. Mine + VLM-verify candidates (cheap; before loading the policy).
    review = pathlib.Path(args.review_dir).expanduser()
    review.mkdir(parents=True, exist_ok=True)
    cands = _candidates(args)
    print(f"{len(cands)} candidate insert attempts")
    verified, report = [], []
    seen: dict[str, int] = {}
    for c in cands:
        seen[c["name"]] = seen.get(c["name"], 0) + 1
        c["seg"] = f"{c['name']}_s{seen[c['name']]}_t{c['start']}"
        before = _frame(c["ep"], "wrist", c["start"], c["ts"], c["wts"])
        after = _frame(c["ep"], "wrist", c["end"], c["ts"], c["wts"])
        votes = [_vlm_pair(args, llm_key, prompt, before, after) for _ in range(3)]
        ok = votes.count("success") >= 2
        print(f"  {c['seg']} t{c['start']}-{c['end']}: votes={votes} -> {'SEED' if ok else 'skip'}")
        for tag, img in (("entry", before), ("terminal", after)):
            cv2.imwrite(
                str(review / f"{c['seg']}_{tag}.jpg"),
                cv2.cvtColor(_crop(img), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
        report.append(
            {
                "segment": c["seg"],
                "episode": str(c["ep"]),
                "start_tick": c["start"],
                "end_tick": c["end"],
                "duration_s": round(float(c["ts"][c["end"]] - c["ts"][c["start"]]), 1),
                "votes": votes,
                "verified": ok,
            }
        )
        if ok:
            verified.append(c)
        time.sleep(0.4)
    (review / "segments.json").write_text(json.dumps(report, indent=2))
    print(f"{len(verified)} verified successful segments; review dump -> {review}")

    if not args.push:
        print("\nPREVIEW ONLY — nothing was pushed. Inspect the review dir, then rerun")
        print("with --push (optionally --include/--exclude segment names).")
        return

    if args.include:
        verified = [c for c in verified if c["seg"] in args.include]
    if args.exclude:
        verified = [c for c in verified if c["seg"] not in args.exclude]
    print(f"pushing {len(verified)} segments after include/exclude")
    if not verified:
        sys.exit("nothing to seed")

    # 2. Load the frozen policy (GPU must be free of the serve).
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), pathlib.Path(args.dir).expanduser()
    )

    # 3. Spawn the replay on the target journal, push, verify, stop.
    sys.path.insert(0, str(pathlib.Path("~/Desktop/openpi/packages/openpi-client/src").expanduser()))
    from openpi_client import msgpack_numpy

    replay = subprocess.Popen(
        [
            str(pathlib.Path.home() / "venvs/subrl/bin/python"),
            str(pathlib.Path.home() / "Desktop/SubRL/real_robot/services/serve_replay.py"),
            "--port", str(args.replay_port), "--journal", str(journal),
        ]
    )
    base = f"http://127.0.0.1:{args.replay_port}"
    try:
        for _ in range(30):
            try:
                if requests.get(f"{base}/healthz", timeout=1).ok:
                    break
            except Exception:  # noqa: BLE001, S110, PERF203
                time.sleep(0.5)
        else:
            raise RuntimeError("replay did not come up")
        start_size = requests.get(f"{base}/stats", timeout=5).json()["size"]

        total = 0
        for ep_id, c in enumerate(verified):
            ext_ts = np.load(c["ep"] / "external_right_timestamps.npy") / 1000.0
            anchors = list(range(c["start"], c["end"], args.anchor_stride))
            feats = []
            for t in anchors:
                example = {
                    "observation/state": _state10(c["states"], t),
                    "observation/image": cv2.resize(
                        _frame(c["ep"], "external_right", t, c["ts"], ext_ts),
                        (224, 224),
                        interpolation=cv2.INTER_AREA,
                    ),
                    "observation/wrist_image": _frame(c["ep"], "wrist", t, c["ts"], c["wts"]),
                    "prompt": PROMPT,
                }
                out = policy.infer(example)
                feats.append(
                    {
                        "z_rl": np.asarray(out["image_embedding"], np.float32),
                        "proprio": _state10(c["states"], t),
                        "ref_chunk": np.asarray(out["actions"], np.float32)[:C, :A],
                    }
                )
            windows = []
            for k, f in enumerate(feats):
                last = k == len(feats) - 1
                nxt = feats[k + 1] if not last else f
                rew = np.zeros(C, np.float32)
                if last:
                    executed = min(c["end"] - anchors[k], C) - 1
                    rew[max(0, executed)] = 1.0  # terminal seat reward
                windows.append(
                    {
                        "z_rl": f["z_rl"],
                        "proprio": f["proprio"],
                        "ref_chunk": f["ref_chunk"],
                        "action_chunk": np.zeros((C, A), np.float32),  # pure pi0.5
                        "rewards": rew,
                        "done": bool(last),
                        "next_z_rl": nxt["z_rl"],
                        "next_proprio": nxt["proprio"],
                        "next_ref_chunk": nxt["ref_chunk"],
                        "source": 1,
                        "collection_phase": "online_rl",
                        "success": 1,
                        "intervention_flag": False,
                        "episode_id": ep_id,
                        "step_id": k,
                    }
                )
            r = requests.post(
                f"{base}/extend",
                data=msgpack_numpy.Packer().pack({"transitions": windows}),
                headers={"Content-Type": "application/octet-stream"},
                timeout=30,
            )
            r.raise_for_status()
            total += len(windows)
            print(f"  seeded {c['name']}: {len(windows)} windows")
        stats = requests.get(f"{base}/stats", timeout=5).json()
        assert stats["size"] == start_size + total, stats
        print(
            f"\nSEEDED {len(verified)} successful episodes / {total} windows "
            f"(replay size {start_size} -> {stats['size']}, journal {journal})"
        )
        print("These episodes count toward the 25-episode warmup gate.")
    finally:
        replay.terminate()
        try:
            replay.wait(timeout=10)
        except subprocess.TimeoutExpired:
            replay.kill()


if __name__ == "__main__":
    main(tyro.cli(Args))

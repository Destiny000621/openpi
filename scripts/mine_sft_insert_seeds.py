"""Mine FIRST-cable insert segments from the SFT LeRobot dataset (seed candidates).

For each dc100 demo episode: cut the first-insert sub-task — segment start = the
LIVE selector's entry rule (gripper closed on the cable + >=60 mm EE displacement
since the grasp = plug out of the right router), segment end = the first release
(+ tail). Verify each candidate with the pair-diff gemini judge (2-of-3 seated
votes on cropped wrist frames) and dump entry/terminal frames + verdicts for
human review. PREVIEW ONLY — pushing happens later via the seeding pipeline.

LeRobot frames are tick-aligned with the videos (no timestamp matching):
state = [x,y,z, r6_0..r6_5, gripper_knuckle_rad(0=open, ~0.7 closed-on-cable)],
camera0 = wrist, camera1 = side.

Run:  ~/venvs/subrl/bin/python scripts/mine_sft_insert_seeds.py
"""

import base64
import dataclasses
import json
import pathlib
import re
import time

import cv2
import numpy as np
import pandas as pd
import requests
import tyro

WRIST_CROP = (0.339, 0.17, 0.761, 0.92)


@dataclasses.dataclass
class Args:
    dataset: str = "~/Desktop/openpi/double_cable_100_r6_v21"
    review_dir: str = "~/Desktop/SubRL/data/franka_sft_seed_review"
    artifacts: str = "~/Desktop/SubRL/real_robot/artifacts/franka_cable_insert"
    held_rad: float = 0.5  # knuckle rad above this = closed on the cable
    open_rad: float = 0.15
    min_extract_mm: float = 60.0
    tail_ticks: int = 10
    llm_model: str = "gemini-3.7-flash"
    votes: int = 3


def _crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = WRIST_CROP
    return img[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)]


def _frame(video: pathlib.Path, idx: int) -> np.ndarray:
    """LeRobot videos are AV1 — cv2 can't decode them here; use ffmpeg's libdav1d."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-c:v", "libdav1d",
                "-i", str(video), "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", tmp.name,
            ],
            check=True,
        )
        frame = cv2.imread(tmp.name)
    if frame is None:
        raise RuntimeError(f"frame read failed: {video} #{idx}")
    return frame  # BGR


def _vlm(args: Args, key: str, prompt: str, before: np.ndarray, after: np.ndarray) -> str:
    def url(img_bgr: np.ndarray) -> str:
        img = _crop(img_bgr)
        h, w = img.shape[:2]
        scale = 640 / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
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


def main(args: Args) -> None:
    import urllib3

    urllib3.disable_warnings()
    ds = pathlib.Path(args.dataset).expanduser()
    review = pathlib.Path(args.review_dir).expanduser()
    review.mkdir(parents=True, exist_ok=True)
    key = (pathlib.Path.home() / ".subrl_llmkey").read_text().strip()
    ns: dict = {}
    art = pathlib.Path(args.artifacts).expanduser()
    exec(compile((art / "primitives.py").read_text(), "primitives.py", "exec"), ns)  # noqa: S102
    prompt = ns["SEATED_HELD_PROMPT"]

    report = []
    n_ok = 0
    for p in sorted((ds / "data/chunk-000").glob("episode_*.parquet")):
        ep = int(p.stem.split("_")[-1])
        st = np.stack(pd.read_parquet(p, columns=["observation.state"])["observation.state"].to_numpy())
        g, pos = st[:, 9], st[:, :3]
        held = g > args.held_rad
        # first hold span (>=30 ticks) followed by an open
        i, seg = 0, None
        while i < len(g) and seg is None:
            if held[i]:
                j = i
                while j < len(g) and held[j]:
                    j += 1
                if j - i >= 30 and j < len(g) and g[j : j + 90].min(initial=1.0) < args.open_rad:
                    seg = (i, j)
                i = j
            else:
                i += 1
        if seg is None:
            report.append({"segment": f"ep{ep:03d}", "verified": False, "votes": ["no_segment"]})
            continue
        a, b = seg
        disp = np.linalg.norm(pos - pos[a], axis=1) * 1000.0
        entry = next((t for t in range(a, b) if disp[t] >= args.min_extract_mm), a)
        end = min(b + args.tail_ticks, len(g) - 1)

        wrist = ds / f"videos/chunk-000/observation.images.camera0/episode_{ep:06d}.mp4"
        before = _frame(wrist, entry)
        after = _frame(wrist, end)
        votes = [_vlm(args, key, prompt, before, after) for _ in range(args.votes)]
        ok = votes.count("success") >= 2
        n_ok += ok
        seg_name = f"sft_ep{ep:03d}_t{entry}"
        for tag, img in (("entry", before), ("terminal", after)):
            cv2.imwrite(
                str(review / f"{seg_name}_{tag}.jpg"), _crop(img), [cv2.IMWRITE_JPEG_QUALITY, 88]
            )
        report.append(
            {
                "segment": seg_name,
                "episode": ep,
                "start_tick": int(entry),
                "end_tick": int(end),
                "duration_s": round((end - entry) / 30.0, 1),
                "votes": votes,
                "verified": ok,
            }
        )
        print(f"ep{ep:03d} t{entry}-{end}: {votes} -> {'SEED' if ok else 'skip'}", flush=True)
        time.sleep(0.3)

    (review / "segments.json").write_text(json.dumps(report, indent=2))
    print(f"\n{n_ok} verified of {len(report)} candidates -> {review}")


if __name__ == "__main__":
    main(tyro.cli(Args))

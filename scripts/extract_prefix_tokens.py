"""Extract pi0.5 prefix IMAGE TOKENS over recorded episodes (SubRL RL-token data).

Walks episode_recorder-format episode dirs (the raw demo sessions AND the SFT
eval rollouts share it: wrist.mp4 + external_right.mp4 + timestamps), samples
frames at --hz, rebuilds the exact wcrop serve inputs (raw wrist frame, 224 side,
10-D r6 state, task prompt) and runs the jitted ``extract_image_tokens`` of the
FROZEN checkpoint — the SAME forward the serve's z_rl uses. Shards go out as
float16 (float32 corpora OOM at ~10k frames).

Run with the serve STOPPED (needs the GPU):
    cd ~/Desktop/openpi
    uv run python scripts/extract_prefix_tokens.py \
        --roots '/home/boyuan/Desktop/Haply_Franka/data_log/double_lan_insertion_uniform*' \
                '/home/boyuan/Desktop/Haply_Franka/data_log_eval_wcrop/*' \
        --out ~/Desktop/SubRL/data/franka_rl_token_corpus
"""

import dataclasses
import glob
import json
import pathlib

import numpy as np
import tyro

PROMPT = "Unplug the two cables from the right router, then insert them into the left router"


@dataclasses.dataclass
class Args:
    roots: tuple[str, ...] = (
        "/home/boyuan/Desktop/Haply_Franka/data_log/double_lan_insertion_uniform*",
        "/home/boyuan/Desktop/Haply_Franka/data_log_eval_wcrop/*",
    )
    out: str = "~/Desktop/SubRL/data/franka_rl_token_corpus"
    config: str = "pi05_franka_double_cable_100_r6_rawrot_wcrop"
    dir: str = "~/.cache/openpi/hf/pi05_franka_double_cable_100_wcrop_10k"
    hz: float = 2.0
    shard_frames: int = 512
    max_frames_per_episode: int = 200


def _episode_dirs(roots: tuple[str, ...]) -> list[pathlib.Path]:
    out = []
    for root in roots:
        for session in sorted(glob.glob(str(pathlib.Path(root).expanduser()))):
            out += sorted(pathlib.Path(session).glob("episode_*"))
    return [e for e in out if (e / "wrist.mp4").exists() and (e / "arm0_states.npz").exists()]


def _state10(states: np.lib.npyio.NpzFile, i: int) -> np.ndarray:
    """[xyz, rot6d, gripper_knuckle_rad] from the recorded [qw,qx,qy,qz,x,y,z]."""
    from scipy.spatial.transform import Rotation

    ee = states["ee_pose"][i]
    rot = Rotation.from_quat([ee[1], ee[2], ee[3], ee[0]])  # wxyz -> xyzw
    r6 = rot.as_matrix()[:, :2].T.reshape(-1)
    grip = float(np.clip(1.0 - states["gripper_pos"][i, 0], 0.0, 1.0)) * 0.7929
    return np.concatenate([ee[4:7], r6, [grip]]).astype(np.float32)


def main(args: Args) -> None:
    import cv2
    import jax

    from openpi.models import model as _model
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as _config

    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), pathlib.Path(args.dir).expanduser()
    )
    extract = nnx_utils.module_jit(policy._model.extract_image_tokens)  # noqa: SLF001
    rng = jax.random.key(0)

    shard_tokens: list[np.ndarray] = []
    shard_meta: list[dict] = []
    shard_id = 0
    token_mask: np.ndarray | None = None  # constant view pattern (right wrist masked)

    def flush() -> None:
        nonlocal shard_id, shard_tokens, shard_meta
        if not shard_tokens:
            return
        np.savez_compressed(
            out / f"shard_{shard_id:04d}.npz",
            tokens=np.stack(shard_tokens).astype(np.float16),
            mask=token_mask,
            meta=json.dumps(shard_meta),
        )
        print(f"wrote shard_{shard_id:04d} ({len(shard_tokens)} frames)")
        shard_id += 1
        shard_tokens, shard_meta = [], []

    episodes = _episode_dirs(args.roots)
    print(f"{len(episodes)} episodes")
    for ep in episodes:
        states = np.load(ep / "arm0_states.npz")
        ts = np.load(ep / "timestamps.npy")
        wrist_ts = np.load(ep / "wrist_timestamps.npy") / 1000.0
        ext_ts = np.load(ep / "external_right_timestamps.npy") / 1000.0
        success = json.loads((ep / "metadata.json").read_text()).get("success", False)
        step = max(1, int(round((1.0 / args.hz) / max(np.median(np.diff(ts)), 1e-3))))
        ticks = list(range(0, len(ts), step))[: args.max_frames_per_episode]

        wcap = cv2.VideoCapture(str(ep / "wrist.mp4"))
        ecap = cv2.VideoCapture(str(ep / "external_right.mp4"))
        try:
            for tick in ticks:
                widx = int(np.argmin(np.abs(wrist_ts - ts[tick])))
                eidx = int(np.argmin(np.abs(ext_ts - ts[tick])))
                wcap.set(cv2.CAP_PROP_POS_FRAMES, widx)
                ok_w, wrist = wcap.read()
                ecap.set(cv2.CAP_PROP_POS_FRAMES, eidx)
                ok_e, ext = ecap.read()
                if not (ok_w and ok_e):
                    continue
                example = {
                    "observation/state": _state10(states, tick),
                    "observation/image": cv2.cvtColor(
                        cv2.resize(ext, (224, 224), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB,
                    ),
                    "observation/wrist_image": cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB),
                    "prompt": PROMPT,
                }
                inputs = policy._input_transform(example)  # noqa: SLF001
                inputs = jax.tree.map(lambda x: np.asarray(x)[np.newaxis, ...], inputs)
                obs = _model.Observation.from_dict(inputs)
                tokens, mask = extract(rng, obs)
                mask_np = np.asarray(mask)[0]
                if token_mask is None:
                    token_mask = mask_np
                else:
                    assert (token_mask == mask_np).all(), "token mask pattern changed mid-corpus"
                shard_tokens.append(np.asarray(tokens)[0])
                shard_meta.append({"episode": str(ep), "tick": int(tick), "success": bool(success)})
                if len(shard_tokens) >= args.shard_frames:
                    flush()
        finally:
            wcap.release()
            ecap.release()
        print(f"{ep.name}: {len(ticks)} frames queued (success={success})")
    flush()
    print(f"done -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))

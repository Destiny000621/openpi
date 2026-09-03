"""SubRL embed-hook gate (franka_port_plan.md §8.1).

Verifies, without a serve, that SUBRL_RETURN_EMBED=1:
  1. leaves the sampled action chunk BIT-IDENTICAL to a flag-off policy, and
  2. returns an `image_embedding` of shape (2048,) float32,
and that SUBRL_EMBED_WEIGHTS changes the embedding but never the actions.

Run (GPU must be free of the live serve; ~14 GB needed):
    cd ~/Desktop/openpi
    uv run python scripts/subrl_embed_probe.py \
        --config pi05_franka_double_cable_100_r6_rawrot_wcrop \
        --dir ~/.cache/openpi/hf/pi05_franka_double_cable_100_wcrop_10k
"""

import dataclasses
import gc
import os
import pathlib

import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    config: str = "pi05_franka_double_cable_100_r6_rawrot_wcrop"
    dir: str = "~/.cache/openpi/hf/pi05_franka_double_cable_100_wcrop_10k"
    weights: str = "1,4,1"


def _wcrop_example() -> dict:
    """Obs matching the wcrop serve contract: 10-D r6 state, RAW 720p wrist, 224 side."""
    rng = np.random.default_rng(1234)
    state = np.zeros(10, np.float32)
    state[:3] = [0.4, 0.0, 0.35]
    state[3:9] = [1, 0, 0, 0, 1, 0]  # identity rot6d
    state[9] = 0.4
    return {
        "observation/state": state,
        "observation/image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(256, size=(720, 1280, 3), dtype=np.uint8),
        "prompt": "Unplug the two cables from the right router, then insert them into the left router",
    }


def _make_policy(args: Args):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    ckpt_dir = pathlib.Path(args.dir).expanduser()
    # No rng arg: Policy defaults to jax.random.key(0) on every load, which is what
    # makes the cross-load bit-identity comparison valid.
    return _policy_config.create_trained_policy(train_config, ckpt_dir)


def main(args: Args) -> None:
    example = _wcrop_example()

    # Pass 1: flag off (baseline actions).
    os.environ.pop("SUBRL_RETURN_EMBED", None)
    os.environ.pop("SUBRL_EMBED_WEIGHTS", None)
    policy = _make_policy(args)
    base = policy.infer(example)
    assert "image_embedding" not in base, "flag off must not return an embedding"
    del policy
    gc.collect()

    # Pass 2: flag on, uniform mean-pool.
    os.environ["SUBRL_RETURN_EMBED"] = "1"
    policy = _make_policy(args)
    on = policy.infer(example)
    del policy
    gc.collect()

    # Pass 3: flag on + wrist-weighted pooling.
    os.environ["SUBRL_EMBED_WEIGHTS"] = args.weights
    policy = _make_policy(args)
    weighted = policy.infer(example)
    del policy
    gc.collect()

    a0, a1, a2 = (np.asarray(x["actions"]) for x in (base, on, weighted))
    assert a0.shape == a1.shape == a2.shape, (a0.shape, a1.shape, a2.shape)
    assert np.array_equal(a0, a1), f"actions changed with flag on (max diff {np.abs(a0 - a1).max()})"
    assert np.array_equal(a0, a2), f"actions changed with weights on (max diff {np.abs(a0 - a2).max()})"

    z1 = np.asarray(on["image_embedding"])
    z2 = np.asarray(weighted["image_embedding"])
    assert z1.shape == (2048,) and z1.dtype == np.float32, (z1.shape, z1.dtype)
    assert z2.shape == (2048,) and z2.dtype == np.float32, (z2.shape, z2.dtype)
    assert np.isfinite(z1).all() and np.isfinite(z2).all()
    assert not np.array_equal(z1, z2), "SUBRL_EMBED_WEIGHTS had no effect on z_rl"

    print(f"actions: shape {a0.shape}, bit-identical across all three passes  OK")
    print(f"z_rl mean-pool: shape {z1.shape} {z1.dtype}, |z| = {np.linalg.norm(z1):.3f}  OK")
    print(f"z_rl weighted({args.weights}): |z| = {np.linalg.norm(z2):.3f}, differs from mean-pool  OK")
    print("SUBRL embed-hook gate PASSED")


if __name__ == "__main__":
    main(tyro.cli(Args))

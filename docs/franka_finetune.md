# Fine-tune pi0.5 on single-arm Franka FR3 (EE-space)

End-to-end recipe for fine-tuning pi0.5 (`pi05_base`) on a single-arm Franka
FR3 dataset collected with [avantbot](https://avant-us.github.io/avantbot/latest/)
+ GELLO/Haply teleop. Single-arm analogue of `docs/yam_finetune.md`, but
**EE-space** (see below).

## Why EE-space, not joint-space

The raw recordings expose `joint_targets` and `target_pose`. In this dataset the
**arm columns of `joint_targets` are dead** — std = 0 across every episode; the
teleop/CRISP stack never stamped arm joint commands (only the gripper column
moves). The actual arm command lives in **`target_pose`** — the absolute EE
target the runner stamps in both Cartesian and joint-IK mode — which varies
properly and leads the measured `ee_pose` by ~7–10 mm. So we train on EE-space
actions.

Representation: **8-D** `[qw, qx, qy, qz, x, y, z, gripper]` — i.e. `target_pose`
+ gripper, used directly. We keep the raw quaternion: it is empirically clean
(0 sign-flips in 72k frames), so no 6D re-encoding is needed. Quaternions are
canonicalized per episode to a single hemisphere (qw≥0) for cross-episode
consistency.

## TL;DR — how this differs from YAM

| | YAM (bimanual) | Franka FR3 (this doc) |
|---|---|---|
| space | joint (14-D) | **EE-space (8-D)** `[quat, xyz, gripper]` |
| action source | `action` (joint) | `target_pose` (+ gripper) |
| data path | `LeRobotAlohaDataConfig` | `LeRobotFrankaDataConfig` (`FrankaInputs`) |
| delta mask | `make_bool_mask(6,-1,6,-1)` | `make_bool_mask(-4,3,-1)` (xyz delta; quat + gripper absolute) |
| cameras | 3 (head + 2 wrist) | 2 (camera0=wrist, camera1=side) |
| base weights | `pi05_base` | **`pi05_base`** (padded to 32-dim internally) |
| steps | 5k–15k | 5k |

pi0.5 runs on a fixed 32-dim action space; the 8-D vectors are zero-padded up to
32 (`PadStatesAndActions`) and sliced back at output. Poses are stored
**absolute** in the dataset; the delta-vs-state conversion happens at train time.

## What this branch adds vs. upstream openpi

| File | What it does |
|---|---|
| `src/openpi/policies/franka_policy.py` | `FrankaInputs` / `FrankaOutputs` — 8-D EE-space, 2-camera transforms |
| `src/openpi/training/config.py` | `LeRobotFrankaDataConfig` + `pi05_franka_lan_insertion` `TrainConfig` |
| `scripts/convert_franka_raw_to_lerobot.py` | Raw FR3 recordings → EE-space LeRobot v2.1 |
| `docs/franka_finetune.md` | This doc |

## Prerequisites

```bash
cd /mnt/localssd/Sichang/openpi
source env.sh                       # conda activate openpi + env vars
export HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home
```

## 1. Convert the dataset → LeRobot v2.1

### Why not the avantbot 2-step path

avantbot's `convert_lerobot` emits a LeRobot **v3.0** dataset that (1) bundles
many episodes into one ~100 MB parquet and (2) stores state/action as *flattened
per-dimension* columns (`observation.state.0..N`). openpi's lerobot is pinned at
**0.1.0** (v2.1): one parquet per episode, single `fixed_size_list` columns.
`scripts/convert_v3_to_v21.py` only relayouts files (symlinks) — it neither
splits the concatenated episodes nor reassembles the columns, and it would carry
over the dead `joint_targets`. This script reads the raw recordings, builds the
EE-space representation, and writes clean v2.1 directly.

### Direct converter (recommended)

Reads per episode: `arm0_states.npz` (`ee_pose`, gripper from `gripper_pos` or
`joint_pos[:,7]`), `arm0_actions.npz` (`target_pose`, gripper from `gripper_target`
or `joint_targets[:,7]`), `camera0.mp4` (wrist), `camera1.mp4` (side).

```bash
uv run python scripts/convert_franka_raw_to_lerobot.py \
    --input-dir "/mnt/localssd/Sichang/Autel Haply Dataset" \
    --repo-id local/lan_insertion_v21 \
    --task "insert the LAN cable" \
    --fps 30 --success-only          # optional: only SUCCESS-marked episodes
```

Output schema (compare `local/vials_4_aug_8ml46_v21`):

```
observation.state           float32 (8,)  [qw,qx,qy,qz, x,y,z, gripper]
action                      float32 (8,)  [qw,qx,qy,qz, x,y,z, gripper]
observation.images.camera0  video         wrist
observation.images.camera1  video         side
```

## 2. `TrainConfig` (already added)

`pi05_franka_lan_insertion` in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_franka_lan_insertion",
    model=pi0_config.Pi0Config(pi05=True),          # action_horizon=50, action_dim=32
    data=LeRobotFrankaDataConfig(
        repo_id="local/lan_insertion_v21",
        default_prompt="insert the LAN cable",
        use_delta_joint_actions=True,               # DeltaActions(make_bool_mask(-4,3,-1))
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=5_000,
    batch_size=32,
    num_workers=8,
    checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
    assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
)
```

Key choices:

1. **No `AssetsConfig`.** Norm-stats are computed fresh from this dataset (step 3).
2. **`use_delta_joint_actions=True`.** xyz → delta vs. state; quaternion + gripper
   absolute (`make_bool_mask(-4,3,-1)`).
3. **Camera mapping.** The repack maps `camera1` (side) → `base_0_rgb` and
   `camera0` (wrist) → `left_wrist_0_rgb`; `right_wrist_0_rgb` is zeroed + masked.

## 3. Compute norm-stats

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py pi05_franka_lan_insertion
```

Streams the dataset through repack → `FrankaInputs` → `DeltaActions(-4,3,-1)` and
writes `norm_stats.json` to
`<assets_base_dir>/pi05_franka_lan_insertion/local/lan_insertion_v21/`. Stats are
over the **post-delta** distribution (small xyz deltas + absolute quat/gripper).

## 4. Train

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_franka_lan_insertion \
    --exp-name=v1 --no-resume
```

Checkpoints land at `<checkpoint_base_dir>/pi05_franka_lan_insertion/v1/<step>/`.

## 5. Serve

```bash
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_franka_lan_insertion \
  --policy.dir=/mnt/localssd/Sichang/openpi-checkpoints/pi05_franka_lan_insertion/v1/4999
```

Wire format the server expects:
`{"state": np.f32(8,), "images": {"base_0_rgb": np.uint8(3,H,W), "left_wrist_0_rgb": ...}, "prompt": "..."}`
— `base_0_rgb` = side camera, `left_wrist_0_rgb` = wrist camera. State is the 8-D
EE vector `[quat, xyz, gripper]`; the model returns an 8-D absolute `target_pose`
+ gripper chunk (renormalize the quaternion to unit on the robot side).

## Deploy on the robot — ⚠️ client mismatch

Your intended inference client
([avantbot policy-inference](https://avant-us.github.io/avantbot/latest/capabilities/policy-inference/),
`config/sessions/policy/droid_pi05_fr3.yaml`) is the **DROID Pi0 client**. It is
**joint-space**: it sends `observation/joint_position` (7) + `observation/gripper_position`
(1) + `exterior_image_1_left` / `wrist_image_left`, and integrates the returned
action as joint velocities/deltas. That does **not** match this EE-space policy
(8-D `[quat, xyz, gripper]` in/out). Recommended: write/extend an avantbot client
that

- sends `base_0_rgb` = side cam, `left_wrist_0_rgb` = wrist cam, and `state` = the
  8-D EE vector (canonicalize the live `ee_pose` quaternion to qw≥0), and
- takes the 8-D action (`target_pose` + gripper) and commands the EE pose through
  avantbot's existing IK (`fr3_mink_droid`) → joint targets for the Hybrid Joint
  Impedance controller. avantbot already resolves EE pose → joints for spacemouse
  intervention, so the IK path exists. Because `target_pose` is what the runner
  dispatched (Cartesian *and* joint-IK mode), this closes the loop cleanly.

Match the client control rate to the dataset (`fps=30`); `action_horizon=50`.

## Gotchas

1. **`joint_targets` arm columns are dead** (std=0). The arm signal is
   `target_pose`; only the gripper column of `joint_targets` is usable.
2. **Quaternions are clean here** (0 sign-flips) but sit in both hemispheres
   across episodes — the converter canonicalizes to qw≥0 per episode. Renormalize
   to unit length on the robot side after inference.
3. **Absolute in dataset, delta at train time.** Only xyz is deltafied; do not
   pre-delta the poses.
4. **Gripper is raw knuckle radians** `[0, 0.7929]` (0 = open). Norm-stats handles
   scaling; the robot side maps back to its Robotiq convention. Note this dataset
   stores the gripper in `joint_pos[:,7]` / `joint_targets[:,7]` (no separate
   `gripper_pos` key), which the converter handles.
5. **Disk.** `pi05_base` weights are 11.6 GB → symlink `~/.cache/openpi` to a big
   drive before the first run (see `docs/yam_finetune.md` gotcha 3).

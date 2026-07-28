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

Representation:

- **Actions (8-D)**: `[qw, qx, qy, qz, x, y, z, gripper]` — `target_pose` +
  gripper target, used directly.
- **State (29-D)**: `[qw, qx, qy, qz, x, y, z, gripper, j0..j6, j0_vel..j6_vel,
  gripper_vel, fx, fy, fz, tx, ty, tz]` — `ee_pose` + gripper (dims 0–7,
  mirroring the action layout so `DeltaActions` lines up), then the arm
  `joint_pos` columns, `joint_vel`, and the external `wrench`. All of these
  signals are present in 77/77 recorded episodes (the avantbot capabilities
  doc's claim that Franka CRISP omits `joint_vel` is stale — it landed in
  v0.5.0, and `wrench`/`joint_eff` are recorded too).

The model observation is images + state + prompt: both cameras go in as image
inputs (see camera mapping below), the 29-D state is quantile-normalized and —
because pi0.5 uses discrete state input — also discretized into prompt tokens.

Quaternion canonicalization (converter): the driver emits per-sample quats that
jump hemisphere whenever the physical qw crosses 0, and this task sits at qw≈0 —
the raw data has persistent within-episode sign flips (58 events across 13
episodes), episodes split across both hemispheres, and 10 episodes where the
state and action streams disagree. The converter repairs all three: it enforces
sign continuity along each stream, flips the action stream to agree with the
state stream, and flips whole episodes onto a dataset-level reference
orientation (saved as `quat_reference.json` inside the dataset root for the
deploy client). Never canonicalize by `sign(qw)` here — with qw≈0 that decision is
sensor noise.

## TL;DR — how this differs from YAM

| | YAM (bimanual) | Franka FR3 (this doc) |
|---|---|---|
| action space | joint (14-D) | **EE-space (8-D)** `[quat, xyz, gripper]` |
| state | 14-D joints (mirrors actions) | **29-D** `[quat, xyz, gripper, joint_pos, joint_vel, wrench]` |
| action source | `action` (joint) | `target_pose` (+ gripper) |
| data path | `LeRobotAlohaDataConfig` | `LeRobotFrankaDataConfig` (`FrankaInputs`) |
| delta mask | `make_bool_mask(6,-1,6,-1)` | `make_bool_mask(-4,3,-1)` (xyz delta; quat + gripper absolute) |
| cameras | 3 (head + 2 wrist) | 2 (camera0=wrist, camera1=side) |
| base weights | `pi05_base` | **`pi05_base`** (padded to 32-dim internally) |
| steps | 5k–15k | 5k |

pi0.5 runs on a fixed 32-dim action space; the 29-D state and 8-D actions are
zero-padded up to 32 (`PadStatesAndActions`) and actions are sliced back to 8 at
output. Poses are stored **absolute** in the dataset; the delta-vs-state
conversion happens at train time and only ever touches the first 8 dims.

## What this branch adds vs. upstream openpi

| File | What it does |
|---|---|
| `src/openpi/policies/franka_policy.py` | `FrankaInputs` / `FrankaOutputs` — 29-D state / 8-D EE actions, 2-camera transforms |
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

Reads per episode: `arm0_states.npz` (`ee_pose`, `joint_pos`, `joint_vel`,
`wrench`; gripper from `gripper_pos` or `joint_pos[:,7]`), `arm0_actions.npz`
(`target_pose`, gripper from `gripper_target` or `joint_targets[:,7]`),
`camera0.mp4` (wrist), `camera1.mp4` (side). A cheap npz-only pre-pass computes
the dataset-level quaternion reference before the main video-encoding pass.

```bash
uv run python scripts/convert_franka_raw_to_lerobot.py \
    --input-dir "/mnt/localssd/Sichang/Autel Haply Dataset" \
    --repo-id local/lan_insertion_s29_v21 \
    --task "insert the LAN cable" \
    --fps 30 --success-only          # optional: only SUCCESS-marked episodes
```

Output schema:

```
observation.state           float32 (29,) [qw,qx,qy,qz, x,y,z, gripper,
                                           j0..j6, j0_vel..j6_vel, gripper_vel,
                                           fx,fy,fz, tx,ty,tz]
action                      float32 (8,)  [qw,qx,qy,qz, x,y,z, gripper]
observation.images.camera0  video         wrist
observation.images.camera1  video         side
quat_reference.json                       canonical hemisphere for deploy
```

The repo id is deliberately new (`_s29`): the 29-D state schema is incompatible
with the old 8-D datasets/checkpoints. `FrankaInputs` accepts the full 29-D
state (sliced to its `state_dim`) or exactly `state_dim` dims (lean inference
clients); anything else is rejected loudly.

## 2. `TrainConfig` (already added)

`pi05_franka_lan_insertion` in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_franka_lan_insertion",
    model=pi0_config.Pi0Config(pi05=True),          # action_horizon=50, action_dim=32
    data=LeRobotFrankaDataConfig(
        repo_id="local/lan_insertion_s29_v21",
        default_prompt="Unplug the cable from the current port, then insert it into the blue port",
        use_delta_joint_actions=True,               # DeltaActions(make_bool_mask(-4,3,-1))
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=5_000,
    batch_size=64,
    fsdp_devices=8,
    num_workers=8,
    checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
    assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
)
```

Key choices:

1. **No `AssetsConfig`.** Norm-stats are computed fresh from this dataset (step 3).
2. **The training prompt is `default_prompt`,** not the converter's `--task`
   string (`prompt_from_task` is False, so the LeRobot task is metadata only).
   Serve-time requests that omit `"prompt"` get `default_prompt` injected — use
   the same wording on the robot client.
3. **`use_delta_joint_actions=True`.** xyz → delta vs. state; quaternion + gripper
   absolute (`make_bool_mask(-4,3,-1)`).
4. **Camera mapping.** The repack maps `camera1` (side) → `base_0_rgb` and
   `camera0` (wrist) → `left_wrist_0_rgb`; `right_wrist_0_rgb` is zeroed + masked.

### 8-D-state ablation: `pi05_franka_lan_insertion_s8`

Identical to the config above except `state_dim=8`: the model sees only dims
0-7 of the same s29 dataset — `[qw,qx,qy,qz, x,y,z, gripper]` (ee_pose +
gripper), no joint_pos/joint_vel/wrench proprio. Same episodes, same actions,
same transforms, so a head-to-head run isolates the value of the extra proprio.
No reconversion needed; `FrankaInputs` slices the state on the fly. Norm-stats
are per-config, so the 8-D stats live under
`<assets_base_dir>/pi05_franka_lan_insertion_s8/` (state stats verified to match
dims 0-7 of the 29-D config's stats exactly).

At serve time an s8 checkpoint accepts either the full 29-D state (sliced
server-side) or exactly 8 dims — a lean client can send just
`[quat, xyz, gripper]`.

### Wrist-only + 8-D-state ablation: `pi05_franka_lan_insertion_s8_wrist`

`wrist_camera_only=True` + `state_dim=8`: the model sees only the wrist camera
(the side camera / `base_0_rgb` is zeroed and masked off, the standard
missing-camera pattern) and only the lean `[quat, xyz, gripper]` state. Same
s29 dataset, no reconversion. 4k steps; checkpoints land at 2000 and 3999
(final) via `save_interval=2_000` + `keep_period=2_000`.

Norm-stats are image-independent (state/actions only), so this config's stats
are byte-identical to `_s8`'s — copied rather than recomputed, at
`<assets_base_dir>/pi05_franka_lan_insertion_s8_wrist/`.

At serve time, a client may omit `observation/image` entirely and send just the
wrist image + 8-D state.

## 3. Compute norm-stats

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py --config-name pi05_franka_lan_insertion

# 8-D-state ablation (same dataset, sliced state)
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py --config-name pi05_franka_lan_insertion_s8
```

Streams the dataset through repack → `FrankaInputs` → `DeltaActions(-4,3,-1)` and
writes `norm_stats.json` to
`<assets_base_dir>/pi05_franka_lan_insertion/local/lan_insertion_s29_v21/`.
Action stats are over the **post-delta** distribution (small xyz deltas +
absolute quat/gripper); state stats cover all 29 dims (absolute — the delta
transform never touches the state).

## 4. Train

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_franka_lan_insertion \
    --exp-name=v1 --no-resume
```

Checkpoints land at `<checkpoint_base_dir>/pi05_franka_lan_insertion/v1/<step>/`.

## Push checkpoints to huggingface

```bash
uv run python scripts/push_to_hub.py \
  --checkpoint=/mnt/localssd/Sichang/openpi-checkpoints/pi05_franka_lan_insertion/v1/4999 \
  --repo=Sichang0621/Franka-cable-pi05-v2-5k
```

```bash
uv run --with "huggingface_hub[hf_transfer]" --with hf_transfer \
  env HF_HUB_ENABLE_HF_TRANSFER=1 \
  huggingface-cli download \
    Sichang0621/Franka-cable-pi05-v2-5k \
    --local-dir ~/.cache/openpi/hf/Franka-cable-pi05-v2-5k \
    --exclude "train_state/*"
```

## 5. Serve

```bash
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_franka_lan_insertion \
  --policy.dir=<checkpoint_base_dir>/pi05_franka_lan_insertion/<exp>/<step>
```

(Older 8-D-state checkpoints such as `Franka-cable-pi05-v1-5k` cannot be served
with this config — see gotcha 6.)

Wire format the server expects — the first server-side transform is
`FrankaInputs`, so the client must send *its* keys (there is no repack at
serving time; the `base_0_rgb`/`left_wrist_0_rgb` mapping happens inside
`FrankaInputs`):

```python
{
    "observation/state":       np.float32 (29,), # [qw,qx,qy,qz, x,y,z, gripper,
                                                 #  j0..j6, j0_vel..j6_vel, gripper_vel,
                                                 #  fx,fy,fz, tx,ty,tz]
    "observation/image":       np.uint8,         # side / third-person camera
    "observation/wrist_image": np.uint8,         # wrist camera
    "prompt": "...",
}
```

The state is assembled live from the robot: `ee_pose` (quat canonicalized — see
Deploy), gripper position (`joint_pos[7]`), the 7 arm `joint_pos` columns, all 8
`joint_vel` columns, and the 6-D external `wrench`. Images may be HWC or CHW
(auto-detected) at any resolution — the server resizes to 224x224. The response
is `{"actions": np.f32(50, 8)}` — a chunk of absolute `target_pose` + gripper
actions (renormalize the quaternion to unit on the robot side).

## Deploy on the robot — ⚠️ client mismatch

Your intended inference client
([avantbot policy-inference](https://avant-us.github.io/avantbot/latest/capabilities/policy-inference/),
`config/sessions/policy/droid_pi05_fr3.yaml`) is the **DROID Pi0 client**. It is
**joint-space**: it sends `observation/joint_position` (7) + `observation/gripper_position`
(1) + `exterior_image_1_left` / `wrist_image_left`, and integrates the returned
action as joint velocities/deltas. That does **not** match this policy
(29-D state in, 8-D EE `[quat, xyz, gripper]` out). Recommended: write/extend an
avantbot client that

- sends `observation/image` = side cam, `observation/wrist_image` = wrist cam,
  and `observation/state` = the 29-D vector above. Canonicalize the live
  `ee_pose` quaternion with the dataset's `quat_reference.json`: flip the whole
  quat (`q = -q`) whenever `dot(q, quat_reference) < 0`. Do **not** canonicalize
  by `sign(qw)` — this task's orientation sits at qw≈0, and
- takes the 8-D action (`target_pose` + gripper) and commands the EE pose through
  avantbot's existing IK (`fr3_mink_droid`) → joint targets for the Hybrid Joint
  Impedance controller. avantbot already resolves EE pose → joints for spacemouse
  intervention, so the IK path exists. Because `target_pose` is what the runner
  dispatched (Cartesian *and* joint-IK mode), this closes the loop cleanly.

Match the client control rate to the dataset (`fps=30`); `action_horizon=50`.

## Gotchas

1. **`joint_targets` arm columns are dead** (std=0). The arm signal is
   `target_pose`; only the gripper column of `joint_targets` is usable. (This
   constrains *actions* only — the state's `joint_pos` columns are live.)
2. **Quaternions are NOT clean.** Measured on the raw recordings: 58 persistent
   within-episode sign flips across 13 episodes (the driver re-canonicalizes
   per sample and this task's qw≈0 crosses zero), all 77 episodes have
   `|qw[0]| < 0.2`, and the raw episodes split across both hemispheres (the
   same physical orientation appears as +q and −q). The converter repairs this
   (continuity fix + shared state/action sign + dataset reference hemisphere);
   the deploy client must apply the same `quat_reference.json` flip to live
   quats. Renormalize output quats to unit length on the robot side.
3. **Absolute in dataset, delta at train time.** Only action xyz is deltafied;
   do not pre-delta the poses. The 29-D state is always absolute.
4. **Gripper is raw knuckle radians** `[0, 0.7929]` (0 = open). Norm-stats handles
   scaling; the robot side maps back to its Robotiq convention. Note this dataset
   stores the gripper in `joint_pos[:,7]` / `joint_targets[:,7]` (no separate
   `gripper_pos` key), which the converter handles.
5. **`gripper_vel` (state dim 22) is all-zeros** in the current recordings — the
   backend never populates it. It is kept for schema stability; at train time
   quantile normalization maps it to a constant, which is harmless. But the
   all-zero column makes its quantile stats degenerate (`q01 == q99 == 0`), so a
   checkpoint trained on this data cannot absorb live nonzero values — they
   normalize to huge magnitudes and saturate the state tokens. If a future
   backend starts filling `gripper_vel`, either re-convert + recompute
   norm-stats + retrain, or zero that dim in the deploy client to match the
   checkpoint.
6. **Old 8-D artifacts are incompatible.** Datasets, norm-stats, and checkpoints
   produced before the 29-D state (e.g. `local/lan_insertion_v21`,
   `Franka-cable-pi05-v1-5k`) cannot be mixed with this pipeline; `FrankaInputs`
   raises on 8-D states. Re-convert, recompute norm-stats, retrain.
7. **Disk.** `pi05_base` weights are 11.6 GB → symlink `~/.cache/openpi` to a big
   drive before the first run (see `docs/yam_finetune.md` gotcha 3).

## Appendix — reference: RLinf's `pi0_realworld` Franka data format

[RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/franka_pi0_sft_deploy.html)
runs π₀ SFT + deployment on a real Franka (bin relocation). Their format choices
are a useful contrast to ours — recorded here from their docs and source
(`rlinf/models/embodiment/openpi/dataconfig/realworld_dataconfig.py`,
`.../policies/realworld_policy.py`, `rlinf/envs/realworld/franka/franka_env.py`).

**Setup**: π₀ (`Pi0Config(action_horizon=10)`, z-score norm, continuous state
input), SpaceMouse teleop at 10 Hz (`step_frequency=10`), SERL impedance
controller (`serl_franka_controllers`), keyboard success labeling, LeRobot
export with columns `image`, `extra_view_image`, `state (19,)`, `actions (7,)`,
`task` (`prompt_from_task=True`).

**State (19-D, composition)** — `RealworldInputs` asserts shape `(19,)`:

| Signal | Dims | Notes |
|---|---|---|
| `tcp_pose` | 6 | `[x, y, z, rx, ry, rz]` — quat→**Euler xyz** via `Quat2EulerWrapper` |
| `tcp_vel` | 6 | EE twist |
| `tcp_force` | 3 | external force (libfranka `K_F_ext_hat_K`) |
| `tcp_torque` | 3 | external torque |
| `gripper_position` | 1 | |

**Actions (7-D, per-step relative)**: `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`.
Each step the env applies `pos += Δxyz·scale` and composes the rotation
properly: `R.from_euler("xyz", Δrpy·scale) * R.from_quat(current)`, then clips
the commanded pose into a configured workspace box (`ee_pose_limit_min/max`).
Because the recorded actions are already per-step deltas, their openpi pipeline
uses **no `DeltaActions` transform**. Camera mapping: `base_0_rgb ← image`,
`left_wrist_0_rgb ← extra_view_image`, right wrist zeroed + masked (same
pattern as ours); state/actions padded to 32; outputs sliced to `[:, :7]`.

**Key contrasts with this pipeline**:

1. **Rotation representation.** Euler state + small Euler *increments* as action
   targets → no quaternion double-cover problem at all (their workspace clamps
   `rx` away from the ±π wrap). We regress absolute quaternions, which is why
   the hemisphere canonicalization above exists.
2. **Delta reference.** Their deltas are per-step (each w.r.t. the current
   pose, integrated at 10 Hz); ours are per-chunk (all 50 xyz deltas w.r.t. the
   chunk-start state), so our late-chunk targets are cm-scale, not mm-scale.
3. **Proprio.** Their 19-D state carries velocity + force/torque (SERL,
   contact-rich heritage) — the same motivation as our 29-D state's
   `joint_vel` + `wrench`.
4. **Deploy safety.** They clamp every commanded pose into a workspace box
   before execution — our avantbot client should do the same before IK.

For another upstream reference: openpi's own `pi05_libero` uses an **8-D state**
`[eef_pos(3), axis-angle(3), gripper_qpos(2)]` with 7-D per-step relative
actions, `action_horizon=10`, and `discrete_state_input=False` (continuous
state only — our configs keep the pi0.5 default of discretized state tokens).
Low-dimensional EE state with a 3-D rotation parameterization + relative
actions is the common upstream pattern.

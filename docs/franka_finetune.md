# Fine-tune pi0.5 on single-arm Franka FR3 (relative EE-space, rot6d)

End-to-end recipe for fine-tuning pi0.5 (`pi05_base`) on a single-arm Franka
FR3 dataset collected with [avantbot](https://avant-us.github.io/avantbot/latest/)
+ GELLO/Haply teleop. Single-arm analogue of `docs/yam_finetune.md`, but
**EE-space with the rot6d10 representation** — the only supported Franka
representation (the legacy quat8 configs were removed; see
[Removed legacy configs](#removed-legacy-quat8-configs)).

## Why EE-space, not joint-space

The raw recordings expose `joint_targets` and `target_pose`. In these datasets
the **arm columns of `joint_targets` are dead** — std = 0 across every episode;
the teleop/CRISP stack never stamped arm joint commands (only the gripper
column moves). The actual arm command lives in **`target_pose`** — the absolute
EE target the runner stamps in both Cartesian and joint-IK mode — which varies
properly and leads the measured `ee_pose` by ~7–50 mm. So we train on EE-space
actions.

## The rot6d10 representation

```
state  (10-D) = [ x, y, z, r6_0..r6_5, gripper ]   from ee_pose
action (10-D) = [ x, y, z, r6_0..r6_5, gripper ]   from target_pose, stored ABSOLUTE
```

`r6` is the Zhou-et-al 6D rotation: the **first two columns of the rotation
matrix**, concatenated `[c0x, c0y, c0z, c1x, c1y, c1z]`. Train-time
`DeltaActions(make_bool_mask(9,-1))` makes **all pose dims — translation AND
rotation — relative to the current state**; only the gripper is absolute.

Why rot6d (and why quat8 was removed): the FR3 driver emits per-sample
quaternions whose sign jumps hemisphere whenever the physical qw crosses 0, and
these tasks sit at qw≈0 — raw recordings show persistent within-episode sign
flips, episodes split across both hemispheres, and state/action streams that
disagree. The quat8 pipeline repaired this with a three-stage canonicalization
plus a `quat_reference.json` the deploy client had to replicate — fragile
machinery that rot6d makes structurally unnecessary: `R(-q) = R(q)`, so the
encoding is quaternion-sign-invariant and **no canonicalization exists anywhere
in the rot6d path** (converter or client).

### Normalization: rot6d bypasses it

The standard (`normalize_rot6d=False`, used by `pi05_franka_double_cable_left_r6_rawrot`):
**xyz and gripper are quantile-normalized; the six rot6d dims are NOT** — they
pass through raw, since rotation-matrix entries already live in [-1, 1] and the
network's rot6d output feeds `rotation_6d_to_matrix()` / Gram-Schmidt directly.
Actions remain 10-D including rot6d — only the normalization map changes.

Mechanism: the data config replaces dims 3:9 of the loaded state/actions stats
with identity-mapping values (`q01=-1, q99=+1` makes the quantile map
`(x-q01)/(q99-q01)*2-1` exactly `x → x`; `mean=0, std=1` for z-score). Because
train-time `Normalize`, the checkpoint's baked `assets/` stats (written from
the same in-memory dict), and serve-time `Unnormalize` all consume
`data_config.norm_stats`, the override is consistent across the whole
lifecycle; the `norm_stats.json` on disk keeps the true computed statistics.
`FrankaInputs` additionally clamps state rot6d dims to [-1, 1] (the pi05
discrete-state tokenizer's `np.digitize` underflows values strictly below -1
to a stray `"-1"` token, so float-eps excursions are clipped).

Known trade-off: with rot6d deltas at ~±0.17 spread vs normalized dims at ~±1,
rotation dims receive ~35x less gradient signal (the flow-matching loss has no
per-dim weighting). Deliberate design choice — monitor rotation accuracy when
evaluating; a per-dim loss weight is the remedy if it lags.

## TL;DR — how this differs from YAM

| | YAM (bimanual) | Franka FR3 (this doc) |
|---|---|---|
| action space | joint (14-D) | **EE-space (10-D)** `[xyz, rot6d, gripper]` |
| state | 14-D joints (mirrors actions) | 10-D (mirrors actions) |
| action source | `action` (joint) | `target_pose` (+ gripper) |
| data path | `LeRobotAlohaDataConfig` | `LeRobotFrankaDataConfig` (`FrankaInputs`) |
| delta mask | `make_bool_mask(6,-1,6,-1)` | `make_bool_mask(9,-1)` (xyz+rot6d relative; gripper absolute) |
| normalization | quantile on everything | quantile on xyz+gripper; **rot6d raw** |
| cameras | 3 (head + 2 wrist) | 2 (camera0=wrist, camera1=side) |
| base weights | `pi05_base` | `pi05_base` (padded to 32-dim internally) |

pi0.5 runs on a fixed 32-dim action space; the 10-D state/actions are
zero-padded to 32 (`PadStatesAndActions`) and actions are sliced back to 10 at
output. Poses are stored **absolute** in the dataset; the delta-vs-state
conversion happens at train time.

## What this branch adds vs. upstream openpi

| File | What it does |
|---|---|
| `src/openpi/policies/franka_policy.py` | `FrankaInputs` / `FrankaOutputs` — 10-D rot6d state/actions, 2-camera transforms |
| `src/openpi/training/config.py` | `LeRobotFrankaDataConfig` (+ `_identity_rot6d_norm_stats`) and the `pi05_franka_double_cable_left_r6*` TrainConfigs |
| `scripts/convert_franka_raw_to_lerobot.py` | Raw FR3 recordings → rot6d10 LeRobot v2.1 (`--rep rot6d10`, default) |
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

### Direct converter

Reads per episode: `arm0_states.npz` (`ee_pose`; gripper from `gripper_pos` or
`joint_pos[:,7]`), `arm0_actions.npz` (`target_pose`, gripper from
`gripper_target` or `joint_targets[:,7]`), `camera0.mp4` (wrist),
`camera1.mp4` (side). No quaternion pre-pass — rot6d needs none.

```bash
uv run python scripts/convert_franka_raw_to_lerobot.py \
    --input-dir /mnt/localssd/Sichang/double_cable_insert_left \
    --repo-id local/double_cable_insert_left_r6_v21 \
    --task "Unplug the two cables from the right router, then insert them into the left router" \
    --fps 30 --rep rot6d10 \
    --episode-list <file with one episode dir name per line>   # optional curation
```

Output schema:

```
observation.state           float32 (10,) [x,y,z, r6_0..r6_5, gripper]
action                      float32 (10,) [x,y,z, r6_0..r6_5, gripper]
observation.images.camera0  video         wrist   (letterboxed to 640 long side)
observation.images.camera1  video         side
```

`--rep quat8` still exists only to reproduce the legacy `_s29` datasets for a
git-restored config; do not use it for new training. Always verify
`meta/info.json:fps` against the recordings (see YAM gotcha #5), and render one
episode (state/action trajectories + camera strips) before training — that
habit is what exposed both the dead `joint_targets` and the quaternion
hemisphere mess.

## 2. TrainConfigs

The standard config — `pi05_franka_double_cable_left_r6_rawrot`:

```python
TrainConfig(
    name="pi05_franka_double_cable_left_r6_rawrot",
    model=pi0_config.Pi0Config(pi05=True),          # action_horizon=50, action_dim=32
    data=LeRobotFrankaDataConfig(
        repo_id="local/double_cable_insert_left_r6_v21",
        default_prompt="Unplug the two cables from the right router, then insert them into the left router",
        use_delta_joint_actions=True,               # DeltaActions(make_bool_mask(9,-1))
        state_dim=10,
        action_representation="rot6d10",
        normalize_rot6d=False,                      # rot6d raw; xyz+gripper quantile
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=4_000,
    save_interval=1_000,
    keep_period=1_000,
    batch_size=64,
    fsdp_devices=4,
    num_workers=32,
    checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
    assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
)
```

`pi05_franka_double_cable_left_r6` (same but `normalize_rot6d=True`) exists for
the provenance of its already-trained checkpoint (v1, final loss 0.0068) —
don't use it for new runs, and never serve one variant's checkpoint under the
other's config name (their stats conventions differ).

Other knobs on `LeRobotFrankaDataConfig`:

- `wrist_camera_only=True` — zero + mask the side camera (`base_0_rgb`); the
  wrist stays in `left_wrist_0_rgb`. Camera ablations need no reconversion, and
  norm-stats are image-independent (copy, never recompute, for camera-only
  variants).
- `default_prompt` is what trains (`InjectDefaultPrompt`); the dataset's baked
  task string only matters under `prompt_from_task=True`. Changing the prompt
  needs no reconversion and no norm-stats redo — but the deploy client's prompt
  must match the training prompt exactly.

## 3. Compute norm-stats

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py --config-name pi05_franka_double_cable_left_r6_rawrot
```

Streams the dataset through repack → `FrankaInputs` → `DeltaActions(9,-1)` and
writes **true** stats to
`<assets_base_dir>/<config>/local/<repo>/norm_stats.json` (the rot6d identity
override is applied at load time, not baked into the file). Sanity-check the
result: all 9 pose action dims zero-centered (xyz ±~0.10 m, r6 ±~0.17 at
q01/q99), gripper absolute [0, 0.79]. Recompute only when the dataset content
or the state/action transform chain changes — never for camera or prompt
changes.

## 4. Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_franka_double_cable_left_r6_rawrot \
    --exp-name=v1 --no-resume
```

Checkpoints land at `<checkpoint_base_dir>/<config>/v1/{1000,2000,3000,3999}`,
each carrying the (identity-overridden) norm stats under `assets/`.

Running on another node where the storage root differs: no code edits — every
path is overridable:

```bash
HF_LEROBOT_HOME=<root>/lerobot_home OPENPI_DATA_HOME=<root>/openpi_cache \
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_franka_double_cable_left_r6_rawrot \
    --exp-name=v1 --no-resume \
    --checkpoint-base-dir=<root>/openpi-checkpoints \
    --assets-base-dir=<root>/openpi-assets
```

## Push checkpoints to huggingface

```bash
uv run python scripts/push_to_hub.py \
  --checkpoint=/mnt/localssd/Sichang/openpi-checkpoints/pi05_franka_double_cable_left_r6_rawrot/v1/3999 \
  --repo=Sichang0621/<name>
```

## 5. Serve

```bash
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_franka_double_cable_left_r6_rawrot \
  --policy.dir=<checkpoint_base_dir>/pi05_franka_double_cable_left_r6_rawrot/v1/3999
```

### Wire contract for the inference client

Request — the first server-side transform is `FrankaInputs`, so the client
sends *its* keys (no repack at serving time):

```python
# state (10,) from the live robot — NO quaternion canonicalization needed
R = ee_pose.orientation.as_matrix()                    # scipy Rotation -> (3,3)
rot6d = np.concatenate([R[:, 0], R[:, 1]])             # (6,) first two COLUMNS
grip_rad = (1.0 - gripper_value) * 0.7929              # avantbot 1=open -> rad 0=open
state = np.concatenate([ee_pose.position, rot6d, [grip_rad]]).astype(np.float32)

request = {
    "observation/state":       state,          # (10,)
    "observation/image":       side_rgb,       # side / third-person camera
    "observation/wrist_image": wrist_rgb,      # wrist camera
    "prompt": "<the training prompt, byte-identical>",
}
```

Images may be HWC or CHW at any resolution — the server resizes to 224×224.

Response — `{"actions": np.f32(50, 10)}`, **absolute** `[xyz, rot6d, gripper]`:
the server re-anchors every chunk to the request-time state (`AbsoluteActions`)
before replying, so there is **no anchor composition on the client**. Parse
each action:

```python
xyz, c0, c1, grip_rad = a[:3], a[3:6].copy(), a[6:9].copy(), float(a[9])
# Gram-Schmidt -> valid rotation (model output is only approximately orthonormal)
c0 /= np.linalg.norm(c0)
c1 -= c0 * (c0 @ c1); c1 /= np.linalg.norm(c1)
R = np.stack([c0, c1, np.cross(c0, c1)], axis=1)       # columns
quat_xyzw = Rotation.from_matrix(R).as_quat()          # -> target orientation
gripper_target = float(np.clip(1.0 - grip_rad / 0.7929, 0.0, 1.0))  # back to 1=open
```

Control rate 30 Hz (match the dataset), `action_horizon=50`.

⚠️ **Do not share a rot6d decoder with the UMI dual-arm pipeline**: UMI encodes
the first two **rows** of the rotation matrix; this pipeline uses the first two
**columns**. Crossing decoders silently yields transposed (inverse) rotations.

## Deploy on the robot

avantbot's stock Franka client (`droid_pi0`) is joint-space DROID — it does not
match this policy. The EE-space client `franka_pi05_ee`
(`avantbot/policies/pi0/franka_ee_client.py`) has the right shape (delta-pose →
runner IK `fr3_mink_droid` → Hybrid Joint Impedance controller) but currently
speaks the removed **quat8** contract; update its state build + action parse to
the rot6d contract above (10-D state, Gram-Schmidt parse, no
`quat_reference.json`) before rollouts. Clamp commanded poses into a workspace
box before IK (RLinf does; see appendix).

## Gotchas

1. **`joint_targets` arm columns are dead** (std=0). The arm signal is
   `target_pose`; only the gripper column of `joint_targets` is usable.
2. **Absolute in dataset, delta at train time.** All 9 pose dims are deltafied
   vs. the current state inside the pipeline; do not pre-delta. The 10-D state
   is always absolute.
3. **Gripper is raw knuckle radians** `[0, 0.7929]` (0 = open). Norm-stats
   handles scaling; the robot side maps back to its Robotiq convention. These
   datasets store the gripper in `joint_pos[:,7]` / `joint_targets[:,7]` (no
   separate `gripper_pos` key), which the converter handles.
4. **Rotation dims get ~35x less gradient** than normalized dims under
   `normalize_rot6d=False` (no per-dim loss weighting). Watch rotation accuracy
   at evaluation.
5. **Legacy artifacts are incompatible.** quat8 datasets (`*_s29_v21`,
   `lan_insertion_v21`), their norm-stats, and their checkpoints cannot be
   mixed with rot6d10 configs — and vice versa. `FrankaInputs` rejects
   mismatched state widths loudly.
6. **Disk.** `pi05_base` weights are 11.6 GB → symlink `~/.cache/openpi` to a
   big drive (or set `OPENPI_DATA_HOME`) before the first run.

## Removed legacy quat8 configs

These trained real checkpoints but used the quaternion representation
(absolute-quat actions, hemisphere canonicalization, `quat_reference.json`
client coupling) that rot6d10 obsoletes. Removed from `config.py`; last present
at commit `a7c3a4e`. Their checkpoints remain under
`/mnt/localssd/Sichang/openpi-checkpoints/<name>/` — to serve one, restore its
TrainConfig from git history (and for `pi05_franka_lan_insertion`, add the
formerly-default `state_dim=29` and `action_representation="quat8"`).

| Removed config | Dataset | Notes |
|---|---|---|
| `pi05_franka_lan_insertion` | `local/lan_insertion_s29_v21` | 29-D state |
| `pi05_franka_lan_insertion_s8` | same | 8-D state slice |
| `pi05_franka_lan_insertion_s8_wrist` | same | + wrist-only |
| `pi05_franka_double_cable_left_s8` | `local/double_cable_insert_left_s29_v21` | loss 0.0041 |
| `pi05_franka_double_cable_left_s8_wrist` | same | loss 0.0041 |

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

1. **Rotation representation.** Euler state + small Euler *increments* as
   action targets → no quaternion double-cover problem (their workspace clamps
   `rx` away from the ±π wrap). We use rot6d, which is also double-cover-free
   and needs no workspace trick, and regress elementwise rot6d deltas rather
   than composed rotation increments.
2. **Delta reference.** Their deltas are per-step (each w.r.t. the current
   pose, integrated at 10 Hz); ours are per-chunk (all 50 deltas w.r.t. the
   chunk-query state), so our late-chunk targets are cm-scale, not mm-scale.
3. **Proprio.** Their 19-D state carries velocity + force/torque (SERL,
   contact-rich heritage); our rot6d10 state is deliberately lean —
   `[pose, gripper]` only.
4. **Deploy safety.** They clamp every commanded pose into a workspace box
   before execution — our avantbot client should do the same before IK.

For another upstream reference: openpi's own `pi05_libero` uses an **8-D state**
`[eef_pos(3), axis-angle(3), gripper_qpos(2)]` with 7-D per-step relative
actions, `action_horizon=10`, and `discrete_state_input=False` (continuous
state only — our configs keep the pi0.5 default of discretized state tokens).
Low-dimensional EE state with a 3-D rotation parameterization + relative
actions is the common upstream pattern.

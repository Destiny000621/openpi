# Fine-tune pi0.5 on YAM single arm (left-arm only)

End-to-end recipe for fine-tuning pi0.5 (`pi05_base`) on a **single-arm** YAM
dataset (left arm only), pushing the checkpoint to HuggingFace, and serving it
for limb.

This is the single-arm counterpart to [`yam_finetune.md`](yam_finetune.md)
(which is bimanual / 14-D). The robot here is one YAM left arm teleoperated with
GELLO, so each frame is **7-D**: 6 joints + 1 gripper, with only `head_camera`
and `left_wrist_camera`.

Training target in this doc: a remote **8×H200** server.

## What the single-arm setup adds vs. the bimanual recipe

| File | Change |
|---|---|
| `src/openpi/policies/aloha_policy.py` | `AlohaOutputs` gains `action_dim: int = 14`; the output slice is `data["actions"][:, : self.action_dim]` (was hardcoded `:14`). |
| `src/openpi/training/config.py` | `LeRobotAlohaDataConfig` gains `action_dim: int = 14`; the delta-action mask is built as one `(6, -1)` block per arm (`action_dim // 7`), and `action_dim` is threaded into `AlohaOutputs`. |
| `src/openpi/training/config.py` | New `TrainConfig` **`pi05_yam_single_arm`** (`action_dim=7`). |
| `configs/yam_pi0_left_arm.yaml` (limb repo) | Single-arm deploy client (left arm, 2 cameras). |

All edits are backward compatible — defaults keep the bimanual 14-D path
unchanged. **These changes live on this branch; make sure the remote server has
them** (push/pull the branch before training).

### Why 7-D works without touching the model

- `AlohaInputs(adapt_to_pi=False)` is a pure passthrough, so a 7-D state/action
  flows through unchanged (YAM is *not* Trossen Aloha — `adapt_to_pi` must be
  `False`).
- The model's `action_dim` stays at the pi0.5 default **32**; `PadStatesAndActions`
  zero-pads 7→32 (just like it pads 14→32 for bimanual). `action_horizon=50`.
- The delta mask `(6, -1)` makes joints 0–5 deltas vs. state and keeps the
  gripper (index 6) absolute.

## Prerequisites

```bash
# on the 8×H200 server
git clone -b <your-branch> <your-openpi-remote> openpi
cd openpi
uv sync

# auth
gcloud auth application-default login    # for gs://openpi-assets/checkpoints/pi05_base
huggingface-cli login                    # for dataset pull + checkpoint push
```

**Disk**: `pi05_base` weights are ~11.6 GB and download to `~/.cache/openpi`. If
`~` is on a small partition, symlink the cache to scratch *before* the first run:

```bash
mkdir -p /mnt/localssd/<user>/openpi-cache
ln -s /mnt/localssd/<user>/openpi-cache ~/.cache/openpi
```

## Step 0 — collect single-arm demos (robot side, limb repo)

```bash
uv run limb record \
  --config-path configs/yam_gello_network_left_arm.yaml configs/collection_pedal.yaml
```

`yam_gello_network_left_arm.yaml` sets `bimanual: false` and only the `left`
arm, so episodes are 7-D with `head_camera` + `left_wrist_camera`.

## Step 1 — merge sessions & convert to LeRobot (robot side, limb repo)

Merge multiple recording sessions into one folder using **symlinks** (the
converter discovers `episode_*` one level deep; names are timestamped so they
don't collide):

```bash
cd <limb-repo>/recordings
mkdir -p vital_left_arm_combined
for S in grasp_..._165438 grasp_..._171105 grasp_..._172529 grasp_..._191859 grasp_..._141211; do
  ln -s "$PWD/$S"/episode_* vital_left_arm_combined/
done
```

Convert, **resampling to 30 fps** (raw recordings are ~100 Hz; 30 fps matches the
proven YAM recipe). `--nearest-action-dims 6` keeps the single-arm gripper
(index 6) on zero-order hold during resampling:

```bash
cd <limb-repo>
uv run limb convert-lerobot \
  --input-dir recordings/vital_left_arm_combined \
  --output-dir datasets/vital_left_arm \
  --target-fps 30 --nearest-action-dims 6
```

Verify `datasets/vital_left_arm/meta/info.json`: `fps: 30`,
`observation.state` shape `[7]`, `action` shape `[7]`, cameras
`head_camera` + `left_wrist_camera`. **Always verify fps against the source
recordings** — the bimanual v0 dataset was wrongly labeled and never performed
well.

> `limb convert-lerobot` writes **LeRobot v3.0**, despite older README comments
> saying v2.1.

### (optional) push the dataset to HuggingFace

```bash
uv run limb upload \
  --source datasets/vital_left_arm \
  --target hf://Sichang0621/yam-vital-left-hand    # hf:// scheme, NOT an https URL
```

## Step 2 — get the dataset onto the server as a v2.1 LeRobot repo

OpenPI's `lerobot` dep is pinned at 0.1.0, which reads **v2.1 only**. limb
exports **v3.0**, so convert.

```bash
# pull the v3.0 dataset from HF into the lerobot cache
huggingface-cli download --repo-type dataset Sichang0621/yam-vital-left-hand \
  --local-dir "$HOME/.cache/huggingface/lerobot/Sichang0621/yam-vital-left-hand"

# v3.0 -> v2.1 (writes the episode_NNNNNN.parquet layout openpi expects)
uv run python scripts/convert_v3_to_v21.py \
  --src="$HOME/.cache/huggingface/lerobot/Sichang0621/yam-vital-left-hand" \
  --dst="$HOME/.cache/huggingface/lerobot/local/yam_vital_left_v21"
```

This produces the v2.1 dataset at `local/yam_vital_left_v21` — the `repo_id` the
config below uses.

## Step 3 — the `TrainConfig`

`pi05_yam_single_arm` already exists in `src/openpi/training/config.py`. For the
8×H200 server, confirm these fields (set `batch_size`/`fsdp_devices`/base dirs to
your rig):

```python
TrainConfig(
    name="pi05_yam_single_arm",
    model=pi0_config.Pi0Config(pi05=True),            # action_dim=32, action_horizon=50
    data=LeRobotAlohaDataConfig(
        repo_id="local/yam_vital_left_v21",
        action_dim=7,                                  # << single left arm: delta mask (6,-1), output slice :7
        assets=AssetsConfig(),                         # << FRESH 7-D norm stats (NOT base-model trossen)
        adapt_to_pi=False,                             # YAM is not Trossen Aloha
        default_prompt="Grasp the vial, and insert it into the stand.",
        repack_transforms=_transforms.Group(inputs=[
            _transforms.RepackTransform({
                "images": {                            # NO cam_right_wrist (not recorded)
                    "cam_high":       "observation.images.head_camera",
                    "cam_left_wrist": "observation.images.left_wrist_camera",
                },
                "state":   "observation.state",
                "actions": "action",
            })
        ]),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=5_000,                             # pi0.5 transfers fast
    batch_size=64,                                     # 8 per device × 8 GPUs
    fsdp_devices=8,                                    # shard pi05 (~3B) across all 8 GPUs
    num_workers=8,                                     # default 2 starves multi-GPU loading
    checkpoint_base_dir="/mnt/localssd/<user>/openpi-checkpoints",
    assets_base_dir="/mnt/localssd/<user>/openpi-assets",
)
```

Key single-arm choices, in order of importance:

1. **`action_dim=7`** — drives the delta mask `(6, -1)` and the 7-D output slice.
   With the bimanual default (14) the mask is `(6,-1,6,-1)`, which crashes on
   7-D data (a 14-element mask broadcast against 7-D state).
2. **`assets=AssetsConfig()`** (no `asset_id="trossen"`) — the bimanual configs
   load the base model's 14-D Trossen norm stats; for single arm those are the
   wrong robot/ranges. The empty `AssetsConfig()` makes `asset_id` default to
   the `repo_id`, so training loads the **fresh 7-D stats** you compute in
   Step 4 (and the same stats get baked into the checkpoint for serving).
3. **`adapt_to_pi=False`** — YAM joint convention is not Trossen; `True` would
   flip joint signs and convert gripper units (both wrong).
4. **repack drops `cam_right_wrist`** — not recorded. `AlohaInputs` substitutes a
   black image with `image_mask=False` for the missing camera automatically.
5. **`fsdp_devices=8`** — pi05 is ~3B params; `fsdp_devices=1` replicates the
   full model per GPU and OOMs. H200 (141 GB) has headroom, but sharding is the
   proven setup; you can raise `batch_size` if you want.

## Step 4 — compute norm stats (one-time)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py pi05_yam_single_arm
```

Streams the dataset through `repack → AlohaInputs → DeltaActions(mask=(6,-1))`
and writes `norm_stats.json` to `<assets_base_dir>/pi05_yam_single_arm/local/yam_vital_left_v21/`.
Sanity check: the `state` and `actions` stats arrays should have length **7**.

## Step 5 — train (8×H200)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_yam_single_arm \
    --exp-name=v1 \
    --resume=false
```

Checkpoints land at `<checkpoint_base_dir>/pi05_yam_single_arm/v1/<step>/`, each
with `params/`, `assets/`, `train_state/`, `_CHECKPOINT_METADATA`. 5k steps is
enough for pi0.5; expect a few hours on 8×H200.

If a GPU is shared, set `CUDA_VISIBLE_DEVICES=...` and drop `batch_size`
accordingly (keep it a multiple of the visible-device count).

## Step 6 — push to HuggingFace (optional)

```bash
uv run python scripts/push_to_hub.py \
  --checkpoint=/mnt/localssd/<user>/openpi-checkpoints/pi05_yam_single_arm/v1/4999 \
  --repo=Sichang0621/yam-vital-left-pi05-v1
```

## Serving

```bash
# --port MUST come before the policy:checkpoint subcommand
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_single_arm \
  --policy.dir=/mnt/localssd/<user>/openpi-checkpoints/pi05_yam_single_arm/v1/4999

# (or from HF)
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_single_arm \
  --policy.dir=Sichang0621/yam-vital-left-pi05-v1
```

The served policy returns **7-D** actions (the `action_dim=7` config makes
`AlohaOutputs` slice to 7 and uses the 7-D norm stats baked into the checkpoint).
First request ~30 s (JAX compile), then ~160 ms. Health check:
`curl http://localhost:8111/healthz` → `OK`.

## limb client (single arm)

On the robot host, use `configs/yam_pi0_left_arm.yaml` (single arm, 2 cameras).
Key fields:

```yaml
robots:
  left: ["robot_configs/yam/left.yaml"]
agent:
  client:
    _target_: limb.agents.policy_learning.policy_client.OpenPIClient
    host: "<server-ip>"
    port: 8111
  obs_transform:
    state_keys: ["left-joint_pos", "left-gripper_pos"]   # 7-D
    image_keys:
      cam_high:        "head_camera-images-rgb"
      cam_left_wrist:  "left_wrist_camera-images-rgb"     # no cam_right_wrist
    prompt: "Grasp the vial, and insert it into the stand."
  action_transform:
    arm_names: ["left"]
    joints_per_arm: 7
    gripper_clip: [0.0, 2.4]
  action_horizon: 50
hz: 60.0
```

Run it:

```bash
uv run limb teleop --config-path configs/yam_pi0_left_arm.yaml
```

## Wire format (for diagnostics)

Send `{"state": np.f32(7,), "images": {"cam_high": np.uint8(3,H,W),
"cam_left_wrist": np.uint8(3,H,W)}, "prompt": "..."}`. Images are CHW,
padded-resized to 224×224 server-side. Use `openpi_client.WebsocketClientPolicy`
(OpenPI ships its own `msgpack_numpy` — don't mix with the PyPI `msgpack-numpy`).
`result["actions"].shape` should be `(50, 7)`.

## Gotchas

1. **FPS.** Raw recordings are ~100 Hz; always convert with `--target-fps 30`
   and verify `meta/info.json:fps`. High-fps data makes `action_horizon=50`
   cover too little time and the per-step deltas degenerate.
2. **v3.0 → v2.1.** limb writes v3.0; openpi reads v2.1. Run
   `scripts/convert_v3_to_v21.py` (Step 2).
3. **Norm stats are 7-D and per-dataset.** Don't reuse the bimanual `trossen`
   stats. `AssetsConfig()` + `compute_norm_stats.py` keeps them self-consistent
   between training and serving (the stats are copied into the checkpoint's
   `assets/`).
4. **Action mode is mixed-delta + Q01–Q99.** Internally pi0.5 applies
   `DeltaActions(make_bool_mask(6, -1))` then quantile-normalizes; joints 0–5
   become `action - state`, gripper (6) stays absolute. Nothing to do — just be
   aware when debugging at the wire.
5. **`adapt_to_pi=False`** is non-negotiable for YAM. `True` corrupts the
   checkpoint via Trossen joint-sign flips + gripper unit conversion.
6. **Don't deploy `pi05_base` to the arm un-fine-tuned.** It can't load with the
   single-arm config (no YAM stats in the base checkpoint) and, via the ALOHA
   preset, applies Trossen transforms to YAM — unsafe and meaningless. Use the
   fine-tuned checkpoint.

# Fine-tune pi0.5 on YAM bimanual

End-to-end recipe for fine-tuning pi0.5 (`pi05_base`) on a YAM bimanual
dataset, pushing the trained checkpoint to HuggingFace, and serving it for
limb.

This branch (`yam-vial-30fps-v1`) is meant to be **self-contained**: clone it
and you have everything you need. The only off-branch artifact is the
trained checkpoint, which lives on HF (or GCS for cold storage).

## What this branch adds vs. upstream openpi

| File | What it does |
|---|---|
| `src/openpi/training/config.py` | YAM `TrainConfig`s: `pi05_yam_place_vial` (v0), `pi05_yam_vial_30fps` (v1), plus ABC-130k configs `pi05_yam_abc_earbuds`, `pi05_yam_abc_fold_box` |
| `scripts/convert_v3_to_v21.py` | Convert a LeRobot v3.0 dataset to v2.1 layout (openpi's lerobot 0.1.0 only reads v2.1) |
| `scripts/convert_abc_mcap_to_lerobot_v21.py` | Convert ABC-130k MCAP episodes → LeRobot v2.1 (see [Fine-tuning from ABC-130k](#fine-tuning-from-abc-130k-mcap--v21)) |
| `scripts/push_to_hub.py` | Push a trained checkpoint dir to HF Hub |
| `docs/yam_finetune.md` | This doc |

## Prerequisites

```bash
git clone -b yam-vial-30fps-v1 https://github.com/Avant-US/openpi.git
cd openpi

conda activate openpi
export C_INCLUDE_PATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$C_INCLUDE_PATH"

uv sync

# auth
gcloud auth application-default login          # for gs://openpi-assets/checkpoints/pi05_base
huggingface-cli login
uv run huggingface-cli login                          # for dataset pull + checkpoint push
hf auth login
```

Set up conda env
```bash
cd /mnt/localssd/Sichang/openpi
source env.sh
```

OpenPI's lerobot dep is pinned at 0.1.0 which only reads **v2.1** datasets.
If your dataset was exported by limb as v3.0 (current default), convert first:

```bash
uv run python scripts/convert_v3_to_v21.py \
  --src=$HOME/.cache/huggingface/lerobot/local/<dataset_image> \
  --dst=$HOME/.cache/huggingface/lerobot/local/<dataset>_v21

uv run python scripts/convert_v3_to_v21.py \
  --src=/mnt/localssd/Sichang/lerobot_home/Sichang0621/vials_4_30fps_180 \
  --dst=/mnt/localssd/Sichang/lerobot_home/local/vials_4_30fps_180_v21

```

This symlinks the data/video files into the older `episode_NNNNNN.parquet`
layout and produces `meta/episodes.jsonl`, `meta/episodes_stats.jsonl`,
`meta/tasks.jsonl`. Idempotent.

## End-to-end pipeline

### 1. Add a `TrainConfig` to openpi

In `src/openpi/training/config.py`, add an entry like our `pi05_yam_vial_30fps`:

```python
TrainConfig(
    name="pi05_<your-task>",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotAlohaDataConfig(
        repo_id="local/<your_dataset>_v21",
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
            asset_id="trossen",            # YAM is ALOHA-shape (14-D) but DIFFERENT joint signs
        ),
        adapt_to_pi=False,                 # critical: YAM is NOT Trossen Aloha; pass raw 14-dim through
        default_prompt="<task instruction>",
        repack_transforms=_transforms.Group(inputs=[
            _transforms.RepackTransform({
                "images": {
                    "cam_high":         "observation.images.head_camera",
                    "cam_left_wrist":   "observation.images.left_wrist_camera",
                    "cam_right_wrist":  "observation.images.right_wrist_camera",
                },
                "state":   "observation.state",
                "actions": "action",
            })
        ]),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=5_000,                 # pi0.5 transfers FAST — 5k is enough on YAM
    batch_size=64,                         # 8 per device × 8 GPUs (use 56 if GPU 2 is in use)
    num_workers=8,                         # default 2 starves multi-GPU training
    checkpoint_base_dir="/mnt/localssd/<user>/openpi-checkpoints",
    assets_base_dir="/mnt/localssd/<user>/openpi-assets",
),
```

Key choices, in order of importance:

1. **`adapt_to_pi=False`** — YAM's joint convention is **not** Aloha-Trossen.
   If you set `True`, openpi flips joint signs and converts gripper units —
   both wrong. The trained checkpoint will be useless.
2. **`repack_transforms`** — your dataset uses YAM camera names but pi0.5's
   `AlohaInputs` expects `cam_high` / `cam_left_wrist` / `cam_right_wrist`.
   The repack maps them; this runs during dataset prep, not at inference.
3. **`num_train_steps=5_000`** — pi0.5 base is pretrained on ~10M robot
   frames so it transfers in 5k steps. Our v1 final loss at step 5000 was
   ~0.02.
4. **`num_workers=8`** — the default 2 makes data loading the bottleneck on
   a 7- or 8-GPU rig. Each worker holds the preprocessor in RAM
   (~hundreds of MB).
5. **`batch_size`** — 8 per device × num_gpus. The included v1 config uses
   64 (8 GPUs); drop to 56 if a GPU is shared with another user.

### 2. Compute norm-stats (one-time, ~25 min on H100)
num_workers=48 --> faster

```bash
export HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py pi05_yam_vial_30fps

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py --config-name pi05_yam_vial_4_30fps

cd /mnt/localssd/Sichang/openpi
export PATH="/mnt/localssd/Sichang:$PATH"
export HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/localssd/Sichang/miniconda3/envs/openpi/lib"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py --config-name pi05_yam_vial_4_30fps
```

This streams the dataset through the full transform chain
(repack → `AlohaInputs` → `DeltaActions(mask=(6,-1,6,-1))`) and writes
`norm_stats.json` to `<assets_base_dir>/<config_name>/<repo_id>/`. The
recorded stats are over the **post-DeltaActions distribution** (joints
become deltas vs state; gripper stays absolute). This is what makes the
joint stats tight enough for Q01-Q99 normalization to work well.

See `docs/norm_stats.md` for more.

### 3. Train (3h on 8×H100, batch 64, 5k steps)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_yam_vial_30fps \
    --exp-name=v1 \
    --resume=false

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_yam_vial_4_30fps --exp-name=v1 --no-resume
```

If a GPU is in use by another user, set `CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7`
and drop `batch_size` in the TrainConfig to 56.

Checkpoints land at `<checkpoint_base_dir>/pi05_yam_vial_30fps/v1/<step>/`.
Each contains `params/`, `assets/`, `train_state/`, and a
`_CHECKPOINT_METADATA` file. The final checkpoint dir is what you serve from.

### 4. Push to HuggingFace

```bash
uv run python scripts/push_to_hub.py \
  --checkpoint=/mnt/localssd/<user>/openpi-checkpoints/pi05_yam_vial_30fps/v1/4999 \
  --repo=ttotmoon/<task>-pi05-v1
```

### 5. Mirror to GCS (optional cold storage)

```bash
gsutil -m cp -r \
  /mnt/localssd/<user>/openpi-checkpoints/pi05_yam_vial_30fps \
  gs://<bucket>/<user>/checkpoints/
```

## Fine-tuning from ABC-130k (MCAP → v2.1)

[XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) is a large
open bimanual-YAM teleoperation dataset. It is **not** LeRobot — episodes ship
as **MCAP** files (`episode.mcap` + optional `annotation.mcap`), one directory
per episode under `data/{train,val}/<task_name>/episode_<uuid>/`. The robot,
14-D joint/gripper convention, and camera set are the **same YAM platform** this
doc already targets, so once converted to v2.1 an ABC task drops straight into
steps 1–5 above with the same `LeRobotAlohaDataConfig` recipe
(`adapt_to_pi=False`, `asset_id="trossen"`).

`scripts/convert_abc_mcap_to_lerobot_v21.py` does the MCAP→v2.1 conversion. Two
worked examples are already in `config.py`: `pi05_yam_abc_earbuds`
(`insert the wireless bluetooth earbuds into the charging case`) and
`pi05_yam_abc_fold_box` (`fold the paper box`), each trained on the first 500
episodes of its task.

### A0. Access + deps

ABC-130k is **gated** — request access on the dataset page and `hf auth login`
first. The converter shells out to **ffmpeg/ffprobe** (decode the per-frame
Annex-B video, letterbox, re-encode), so make sure they're on `PATH`
(`sudo apt-get install -y ffmpeg`).

### A1. Download N episodes of one task

Episodes are ~180 MB (RealSense) to ~600 MB (ZED-X) each — 500 episodes is
~140–435 GB, so pick N deliberately. **List explicitly, then fetch per file** —
do *not* rely on `snapshot_download(allow_patterns=["data/train/<task>/**"])`:
the `**` glob silently matched **0 files** under `huggingface_hub` 1.23.0 (works
on 0.32.x), so it can "succeed" while downloading nothing.

```python
# download_task.py  —  uv run --with huggingface_hub python download_task.py
import concurrent.futures as cf
from huggingface_hub import list_repo_files, hf_hub_download

REPO, TASK, N = "XDOF/ABC-130k", "fold_the_paper_box", 500
OUT = "/mnt/localssd/Sichang/abc_fold_box/raw"

files = sorted(
    f for f in list_repo_files(REPO, repo_type="dataset")
    if f.startswith(f"data/train/{TASK}/") and f.endswith("/episode.mcap")
)[:N]                                      # sorted by uuid → deterministic subset
print(f"{len(files)} episodes")            # assert files, else you got the glob bug

def get(p):
    return hf_hub_download(REPO, repo_type="dataset", filename=p, local_dir=OUT)

with cf.ThreadPoolExecutor(max_workers=12) as ex:
    list(ex.map(get, files))
```

`list_repo_files` enumerates the whole 170k-file repo (~1–2 min) before the
first byte downloads — that pause is normal, not a hang. To also pull subtask
labels for later use, drop the `episode.mcap` filter (grabs `annotation.mcap`
too; ~42k of ABC's episodes are annotated). For the **val split** used in A4,
repeat with `data/val/<task>/`.

### A2. Convert MCAP → LeRobot v2.1

```bash
cd /mnt/localssd/Sichang/openpi
uv run --python 3.11 \
  --with numpy --with mcap --with mcap-protobuf-support --with pyarrow --with tyro \
  python scripts/convert_abc_mcap_to_lerobot_v21.py \
    --root /mnt/localssd/Sichang/abc_fold_box/raw/data/train/fold_the_paper_box \
    --out  /mnt/localssd/Sichang/lerobot_home/local/abc_fold_box_v21 \
    --workers 48 \
    --max-episodes 500          # optional cap; omit to convert everything under --root
```

What it produces (byte-identical layout to `vials_4_30fps_180_v21`):

- `observation.state` / `action`: **14-D float32**
  `[left j0..5, left grip, right j0..5, right grip]`, joints in **radians**,
  gripper normalized **0=closed … 1=open**. `action` = commanded joints
  (`/{side}-arm-action` + `/{side}-ee-action`), `state` = measured. Absolute,
  not delta (pi0.5 applies `DeltaActions` internally — see Gotcha #4).
- Three cameras letterboxed to **640×480 h264**: `head_camera`,
  `left_wrist_camera`, `right_wrist_camera`.
- `meta/{info.json,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl,stats.json}`
  plus `meta/episode_ids.json` mapping each `episode_index` → original ABC uuid.
- `default_prompt` for your TrainConfig = the string the converter reads from
  each episode's `/instruction` topic and writes into `tasks.jsonl`.

**Fixed 30 Hz resampling (the important part).** In ABC every stream runs on its
own clock: the action stream at **~200 Hz**, state at **~265 Hz**, and cameras at
**30 Hz (ZED-X) or 50–60 Hz (RealSense)**. The converter builds a fixed 30 Hz
tick clock over the overlap window of all streams and does **causal floor
matching** (latest message at or before each tick) — actions are subsampled
~6.7:1, faster cameras are decimated, and no camera is below 30 Hz so frames are
never duplicated. This 30 Hz target matches the vial datasets and the official
ABC exporter, and is what keeps you clear of the FPS-mislabel trap in Gotcha #5.
Both station types are handled automatically: RealSense (mono top camera) and
ZED-X (stereo top — one eye picked deterministically per episode).

The conversion is CPU-bound (ffmpeg); 500 episodes ≈ 45–60 min on 48 workers.
Sanity-check that it loads through openpi's pinned lerobot:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/abc_fold_box_v21")     # HF_LEROBOT_HOME must be set
print(ds.num_episodes, ds.num_frames, ds.fps)     # 500, ~2.26M, 30
```

### A3. TrainConfig → norm-stats → train

Add a `TrainConfig` exactly like step 1 (copy `pi05_yam_abc_fold_box`), pointing
`repo_id` at your converted dataset and `default_prompt` at the task string.
Then run steps 2–3 unchanged:

```bash
export HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/mnt/localssd/Sichang/miniconda3/envs/openpi/lib"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/compute_norm_stats.py --config-name pi05_yam_abc_fold_box --max-frames 600000

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_yam_abc_fold_box --exp-name=v1 --no-resume
```

pi0.5 transfers fast but bigger single-task sets keep improving past 5k: our 500-
episode runs were still dropping held-out error at 15k steps (earbuds final train
loss ~0.011, fold-box ~0.019). Checkpoints land at 5000 / 10000 / 14999.

### A4. Pick a checkpoint on the held-out val split

Unlike the vial data (all in-train), ABC ships a real `data/val/<task>/` split.
Convert those episodes too (A1+A2 with `data/val/…`, output `<task>_val_v21`)
and rank checkpoints by open-loop action MSE on data the model never saw:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/eval_open_loop.py \
    --config-name pi05_yam_abc_fold_box \
    --repo-id local/abc_fold_box_val_v21 --episodes 0-29 \
    --stride 90 --max-per-ep 40 \
    --checkpoint <ckpt_dir>/5000 \
    --checkpoint <ckpt_dir>/10000 \
    --checkpoint <ckpt_dir>/14999
```

It reports overall / joints / gripper MSE (lower = better). For both example
tasks 14999 won cleanly, but treat this as a *proxy* to choose 1–2 checkpoints
for real robot demos: open-loop MSE penalizes valid alternative strategies on
multimodal tasks, and the gripper column matters most for grasp-timing tasks
(e.g. insertion) where MSE is least trustworthy.

## Serving

```bash
# from HF (simplest — pulls on first request)
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_vial_30fps \
  --policy.dir=ttotmoon/yam-vial-place-pi05-v1

# from local dir
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_vial_30fps \
  --policy.dir=/mnt/localssd/<user>/openpi-checkpoints/pi05_yam_vial_30fps/v1/4999

# from GCS
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_vial_30fps \
  --policy.dir=gs://<bucket>/<user>/checkpoints/pi05_yam_vial_30fps/v1/4999
```

**Important CLI quirk**: `--port` must come BEFORE the `policy:checkpoint`
subcommand. Putting it after gives `Unrecognized options: --port=8111`.

First request takes ~30s due to JAX compile. Subsequent ~160ms.

Verify it's listening:
```bash
curl http://localhost:8111/healthz   # → "OK"
```

## limb client

Use `limb/configs/yam_pi0_bimanual.yaml` on the robot host. The key fields:

```yaml
agent:
  client:
    _target_: limb.agents.policy_learning.policy_client.OpenPIClient
    host: "<gpu-machine-ip>"
    port: 8111
  obs_transform:
    _target_: limb.agents.policy_learning.transforms.OpenPIObsTransform
    state_keys: ["left-joint_pos", "left-gripper_pos",
                 "right-joint_pos", "right-gripper_pos"]
    # AlohaInputs camera names — must be cam_high/cam_left_wrist/cam_right_wrist
    image_keys:
      cam_high:        "head_camera-images-rgb"
      cam_left_wrist:  "left_wrist_camera-images-rgb"
      cam_right_wrist: "right_wrist_camera-images-rgb"
    image_size: [224, 224]
    prompt: "<task instruction>"
  action_horizon: 50       # pi0.5 default
  async_inference: false
hz: 60.0                   # match dataset fps (30 fps × 2)
```

## Gotchas — read before reproducing

These have cost real time:

1. **Wire-protocol detail.** OpenPI uses its OWN `msgpack_numpy` (in
   `packages/openpi-client/src/openpi_client/msgpack_numpy.py`) that
   serializes ndarrays with `__ndarray__` keys. The standard `msgpack-numpy`
   PyPI package uses a different format. If you write a diagnostic client,
   either use `openpi_client.WebsocketClientPolicy` directly or inline
   OpenPI's pack/unpack helpers — don't mix.

2. **Obs format at the wire.** Send
   `{"state": np.f32(14,), "images": {"cam_high": np.uint8(3,H,W), ...}, "prompt": "..."}`.
   Images are CHW (channels first), padded-resize to 224×224. The keys
   must already be `cam_high` / `cam_left_wrist` / `cam_right_wrist`.
   The `repack_transforms` in the TrainConfig runs during **dataset prep**,
   not at inference.

3. **Disk usage.** `pi05_base` weights are 11.6 GB and download to
   `~/.cache/openpi`. If `~` is on a small root partition (we hit a
   9 GB free disk and OOM'd), symlink the cache to a larger drive
   BEFORE the first training run:
   ```bash
   mkdir -p /mnt/localssd/<user>/openpi-cache
   ln -s /mnt/localssd/<user>/openpi-cache ~/.cache/openpi
   ```

4. **Action mode is mixed-delta + Q01-Q99.** pi0.5 internally applies
   `DeltaActions(mask=make_bool_mask(6, -1, 6, -1))` then quantile-normalizes
   using the stats from step 2. Joint dims (0-5, 7-12) become
   `action - state`; gripper dims (6, 13) stay absolute. This is intrinsic
   to the recipe — you don't need to do anything, just be aware of it when
   debugging at the wire.

5. **FPS labeling matters.** YAM dataset fps was wrongly labeled 30Hz when
   the source recordings were 50-58Hz; the v0 config (`pi05_yam_place_vial`)
   was trained on that broken dataset and never performed well. v1
   (`pi05_yam_vial_30fps`) uses the resampled 30fps dataset from
   `ttotmoon/8ml_vial_place_30fps`. **Always verify
   `meta/info.json:fps` against the source recordings** before training.

## What's different from upstream openpi

- YAM + ABC-130k `TrainConfig` entries in `src/openpi/training/config.py`
- Two dataset-converter scripts (`scripts/convert_v3_to_v21.py`,
  `scripts/convert_abc_mcap_to_lerobot_v21.py`)
- One HF-push script (`scripts/push_to_hub.py`)
- This doc

No upstream openpi code is modified. If you upgrade pi0.5 to a future
openpi release, rebase these four additions onto the new `main` and
re-push to `yam-vial-30fps-v1` (force-push is fine for this branch since
it's deployment-only).

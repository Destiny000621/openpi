# UMI dual-Franka cardboard-box fine-tuning and deployment

This guide supports two episode-construction choices from
[`byang11259/cardboard_box_tcp_curated`](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated),
then fine-tunes the JAX π0.5 policy and deploys it on a dual-Franka setup with
two UMI fisheye cameras. The recommended choice materializes one logical
episode per physical box. An explicit long-episode ablation trains directly on
the original source episodes. Each choice has a primary query-anchor-relative
representation and a controlled absolute-action baseline.

> [!WARNING]
> A source dataset episode can contain **multiple physical cardboard boxes**.
> A source episode is therefore a recording/session boundary, not necessarily a
> valid task-episode boundary. For boundary-safe training, split every physical
> box into a logical episode or use a sampler and validity mask that reject
> chunks crossing a logical boundary. The stock LeRobot loader and
> `compute_norm_stats.py` do not consume a sidecar boundary manifest by
> themselves. The `_long_episode` configs intentionally preserve this behavior
> as an ablation: their chunks stay inside a source episode but can cross from
> one physical box instance into the next.

## Registered configs

| Episode path | Purpose | Config | Dataset repo |
| --- | --- | --- | --- |
| Logical box (recommended) | relative primary | `pi05_umi_dual_franka_cardboard_box_relative` | `local/cardboard_box_tcp_curated_logical_train` |
| Logical box (recommended) | absolute baseline | `pi05_umi_dual_franka_cardboard_box_absolute` | `local/cardboard_box_tcp_curated_logical_train` |
| Original source episode (ablation) | relative primary | `pi05_umi_dual_franka_cardboard_box_relative_long_episode` | `local/cardboard_box_tcp_curated_x264` |
| Original source episode (ablation) | absolute baseline | `pi05_umi_dual_franka_cardboard_box_absolute_long_episode` | `local/cardboard_box_tcp_curated_x264` |

> [!NOTE]
> The long-episode configs point at `local/cardboard_box_tcp_curated_x264`, a
> local derived copy of the curated source dataset whose parquet/meta files are
> byte-identical and whose 36 videos are re-encoded near-losslessly
> (libx264, `-crf 14 -g 15 -bf 0`). The original HEVC exports use ~250-frame
> GOPs with B-frames; lerobot's torchcodec decode path
> (`seek_mode="approximate"`) returns wrong frames near GOP tails on such
> streams and fails its 1e-4 s timestamp-tolerance check. Dense keyframes and
> no B-frames make index-to-pts mapping exact and random access fast. See
> `REENCODE_PROVENANCE.md` inside the derived dataset directory.

All four configs:

- start from `gs://openpi-assets/checkpoints/pi05_base/params`;
- use the same two cameras, state representation, 6D rotation encoding,
  prompt plumbing, horizon, and training hyperparameters;
- consume the canonical post-repack keys `observation/state`,
  `observation/left_head`, `observation/right_head`, `actions`, and `prompt`;
- use an action horizon of 50 at the dataset's nominal 29.97 Hz, about
  1.67 seconds of predicted motion; and
- require their **own fresh normalization statistics**.

For either episode path, the relative and absolute configs use identical
absolute state20. The relative action is fixed-query-anchor true-SE(3)
action20; the baseline action is absolute action20. The logical and long paths
must not share norm stats or checkpoints even when their representation is the
same.

The portable defaults target one H200: full fine-tuning, batch size 32,
`fsdp_devices=1`, eight data workers, 5,000 steps, `./assets`, and
`./checkpoints`.

During training, the adapter maps each LeRobot row's `task` string to the model
`prompt`, so a reviewed logical-episode manifest can supply a fold-only label.
If no prompt is present at inference, the registered configs fall back to the
full-task prompt:

```text
Assemble the cardboard box and put it into the bin
```

Do not change only the prompt to create a fold-only dataset. See
[Task labeling and temporal crops](#task-labeling-and-temporal-crops).

## Raw dataset contract

### Raw state and action: 16 values

The dataset stores each state as:

```text
[
  left_x, left_y, left_z,
  left_qx, left_qy, left_qz, left_qw,
  left_gripper,
  right_x, right_y, right_z,
  right_qx, right_qy, right_qz, right_qw,
  right_gripper,
]
```

| Slice | Meaning |
| --- | --- |
| `0:3` | left chopstick-tip TCP position |
| `3:7` | left TCP quaternion in `xyzw` order |
| `7` | left gripper, absolute, in `[0, 1]`, with `1 = open` |
| `8:11` | right chopstick-tip TCP position |
| `11:15` | right TCP quaternion in `xyzw` order |
| `15` | right gripper, absolute, in `[0, 1]`, with `1 = open` |

The raw action stored at index `t` is the demonstrated absolute next TCP
target:

```text
raw_action[t] == raw_state[t + 1]
```

This one-step shift is part of the dataset contract. The recommended
logical-episode materialization prevents it from crossing a physical-box
boundary. The explicit long-episode ablation retains source-episode
transitions, including any transition between two physical boxes.

Additional invariants:

- arm order is always left, then right;
- poses are absolute in the episode ArUco marker/world frame;
- the dataset's `-90°` yaw convention is already baked into the poses—do not
  apply it again;
- the end effector is the chopstick grasp-tip TCP, not the Franka flange;
- gripper values remain absolute in all model configurations; and
- the custom dataset convention `1 = open` is intentionally preserved. Do not
  replace it with a generic robot convention.

### No measured joints

The dataset does **not** contain `joint_pos`, `joint_vel`, or `joint_target`.
None of the configs invent them. IK solutions are controller-side commands or
pseudo-labels, not measured demonstration state, and must not be introduced as
ground truth.

The robot runtime obtains a live Cartesian TCP pose from Franka FK plus
calibration and the fixed flange-to-tip transform. Joint states may be used by
the controller, collision checker, and FK implementation, but they are not
policy state dimensions.

### Units are a hardware gate

Confirm the dataset translation units empirically before commanding either
robot. Check representative displacements against a known physical dimension,
record the result in the experiment manifest, and test round trips in those
units. Do not infer metres versus millimetres from value magnitude alone.
Normalization can make an incorrect unit scale look numerically plausible
while producing dangerous commands.

## Choose the episode path

### Stock OpenPI/LeRobot long-source semantics

The two `_long_episode` configs deliberately point at the untouched
`byang11259/cardboard_box_tcp_curated` source repo and use the stock OpenPI
LeRobot action-chunk path
([OpenPI loader](https://github.com/Destiny000621/openpi/blob/59ce2725e887d44c36dd1a3d3106d00d8ad6cd5e/src/openpi/training/data_loader.py#L135-L147)):

```text
delta_timestamps["action"] =
    [0 / fps, 1 / fps, ..., 49 / fps]
```

For a query at source row `t`, LeRobot therefore requests the singular
`action` column at rows `t ... t + 49`. Because the dataset contract is
`action[t] = state[t + 1]`, those entries target states
`t + 1 ... t + 50`. At 29.97 Hz, the requested action samples span
`49 / 29.97 ≈ 1.64 s`, while the last physical target is approximately
`50 / 29.97 ≈ 1.67 s` after the query state.

LeRobot clips each requested index to the current **source episode** and marks
out-of-range values with `action_is_pad`
([pinned LeRobot implementation](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py#L665-L678)).
Thus:

- a chunk never enters a different source episode;
- near a source-episode end, the final stored source action is repeated;
- source-internal physical-box boundaries are invisible to the loader;
- a query before such a boundary can contain actions from the next box; and
- neither the current UMI transform nor `compute_norm_stats.py` consumes a
  physical-box sidecar mask or the padding flag.

For the long relative config, every one of those future absolute targets is
converted against the single state at query row `t`. A reset into the next
physical box can therefore become a large fixed-anchor relative target. This
is the intended long-episode ablation behavior, not a boundary-safe dataset.
Its norm stats and loss include those samples.

Use this path to test the episode-local, fixed-horizon hypothesis. Do not
describe it as one-box-per-episode training, leakage-free validation, or the
recommended hardware model. The logical path remains the default for
controlled train/validation/test splits and deployment claims.

### Relationship to the Hy-Embodied UMI loader

The long path is **inspired by**, but does not copy, the pinned Hy-Embodied UMI
target builder:

- Hy configures a 50-step horizon and downsamples the 30 Hz stream by three,
  producing a nominal 10 Hz, roughly five-second chunk
  ([config](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/config/dataset/umi_lance.yaml#L28-L50)).
- Hy limits sampling by dataset episode bounds
  ([episode bounds](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L157-L165))
  and clamps `c_id + 3k` to the last row, without an internal physical-box
  boundary check
  ([future index and clamp](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L520-L545),
  [assembled indices](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L590-L630)).
- Hy takes future `xyz + quaternion` from future `observation.state` rows and
  overwrites only gripper values from `action`
  ([target construction](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L525-L534)).
  The cardboard-box dataset already provides full absolute action16 targets,
  so OpenPI correctly chunks the singular `action` column instead of
  reproducing that builder.
- Hy's first future pose is the current state (`k = 0`), so its first relative
  pose is identity. OpenPI's first entry is `action[t] = state[t + 1]`, so its
  first relative pose is current-to-next. Hy also takes the prompt from the
  anchor row
  ([prompt behavior](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L580-L588)).

Both use absolute dual-TCP state20 and a fixed-anchor relative action20
concept, but the temporal sampling and target sources differ. See the pinned
[Hy transform implementation](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/utils/transform_utils.py#L16-L123).

## Recommended path: split multi-box recordings into logical episodes

Use these terms consistently:

- **collection session**: one acquisition session, potentially spanning
  several source files;
- **source episode**: the episode/file boundary in the published dataset;
- **box instance**: one physical cardboard box manipulated in that recording;
- **logical episode**: the contiguous frames for one box instance and one
  declared task variant.

All logical episodes derived from the same source episode or collection
session are correlated. Put the entire group into exactly one of train,
validation, or test. A random split by logical episode leaks the same camera
placement, operator, background, calibration, and adjacent motion into
validation.

### Annotation manifest

Copy the provided template and replace every illustrative boundary after
frame-by-frame review:

```bash
cp configs/cardboard_box_segments.example.json \
  configs/cardboard_box_segments.json
```

The splitter accepts this versioned JSON schema. `end_frame` is **exclusive**,
so every interval is `[start_frame, end_frame)`.

```json
{
  "version": 1,
  "source_repo_id": "byang11259/cardboard_box_tcp_curated",
  "segments": [
    {
      "source_episode": 0,
      "box_index": 0,
      "start_frame": 120,
      "end_frame": 2540,
      "split": "train",
      "task": "Assemble the cardboard box and put it into the bin"
    },
    {
      "source_episode": 0,
      "box_index": 1,
      "start_frame": 2710,
      "end_frame": 5160,
      "split": "train",
      "task": "Assemble the cardboard box and put it into the bin"
    },
    {
      "source_episode": 1,
      "box_index": 0,
      "start_frame": 90,
      "end_frame": 2480,
      "split": "validation",
      "task": "Assemble the cardboard box and put it into the bin"
    }
  ]
}
```

These numbers are examples, not reviewed annotations for the published data.
Keep richer annotation notes—collection-session ID, fold-complete frame,
bin-motion-start frame, annotator, calibration ID, camera serials, checksums,
and exclusions—in a versioned companion record. The materializer enforces one
split per `source_episode`. If one collection session spans several source
episodes, the reviewer must additionally assign all of those source episodes
to the same split; the current JSON schema cannot infer that relationship.

### Recommended preprocessing workflow

1. Inventory collection sessions and source episodes. Recover a stable
   `collection_session_id`; if none exists, conservatively group by source
   episode.
2. For every physical box, annotate its first task frame, first
   fully-folded frame, first bin-placement motion frame, and exclusive end.
3. Validate that intervals are ordered, non-overlapping, in range, and
   synchronized across both cameras and both arms.
4. Assign train/validation/test by collection-session group **before**
   materializing logical episodes. Every sibling box from a group inherits the
   same split.
5. Run a full-manifest dry run, then materialize train, validation, and test
   into separate local LeRobot repositories.
6. The splitter validates `action[t] = state[t + 1]` for every nonterminal
   logical frame and replaces the final action with an in-episode absolute
   no-op: `action[-1] = observation.state[-1]`.
7. It rewrites episode/frame/global indices and timestamps, trims both videos
   at the exact annotated frames, decodes them to verify frame count, and
   records source provenance in `meta/logical_segments.jsonl`.
   Numeric episode stats are recomputed. Cropped-video stats are omitted
   instead of incorrectly copying full-source-episode camera stats; this
   adapter's training normalization uses transformed state/actions only.
8. Point **both** registered configs to the same derived dataset revision.
   Record its immutable repository revision or content hash.
9. Only now compute the separate relative and absolute norm stats, then train.
10. Audit counts by split, source episode, session, task variant, and box
    instance. Manually inspect the first and last chunk of every logical
    episode.

Assuming the source LeRobot v2.1 dataset is under `HF_LEROBOT_HOME`, first
validate the **complete** manifest. The `--split train` argument selects a
nonempty output split, but `--dry-run` validates every segment and grouped
source-episode split in the manifest:

```bash
uv run python scripts/split_cardboard_box_lerobot_v21.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_train" \
  --manifest configs/cardboard_box_segments.json \
  --split train \
  --action-horizon 50 \
  --dry-run
```

Then materialize each declared split into an absent or empty destination:

```bash
uv run python scripts/split_cardboard_box_lerobot_v21.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_train" \
  --manifest configs/cardboard_box_segments.json \
  --split train \
  --action-horizon 50

uv run python scripts/split_cardboard_box_lerobot_v21.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_validation" \
  --manifest configs/cardboard_box_segments.json \
  --split validation \
  --action-horizon 50

uv run python scripts/split_cardboard_box_lerobot_v21.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_test" \
  --manifest configs/cardboard_box_segments.json \
  --split test \
  --action-horizon 50
```

Run only splits present in the reviewed manifest. Each logical segment must
contain at least `action_horizon + 1`, currently 51, frames. The registered
recommended training configs already point to
`local/cardboard_box_tcp_curated_logical_train`.

Materializing logical episodes is preferred because the normal LeRobot
episode-aware sampler can then enforce boundaries. If data must remain in the
original source episodes **and boundary-safe training is still the goal**, a
manifest alone is insufficient: the dataset sampler, chunk builder,
normalization pass, and training loss must all consume the same
logical-boundary validity mask. In particular:

- reject any query/action pair crossing from one box instance to another;
- reject any 50-step chunk that reaches across a boundary;
- mask any deliberately padded terminal waypoint in both norm stats and loss;
- keep image, state, action, timestamp, and mask indices synchronized; and
- test that the online loader and offline preprocessing select identical
  frames.

The materialized route does not create a separate terminal-loss mask. LeRobot
may clamp a terminal chunk to the final in-episode no-op, but it can no longer
cross into the next physical box. If an experiment requires excluding repeated
terminal no-ops from norm stats or loss, implement and test that mask in both
paths; do not assume the sidecar manifest is consumed automatically.

The non-suffixed logical-config commands later in this guide assume one of
these boundary-safe preparations. The explicitly labeled `_long_episode`
commands intentionally skip them and must be reported as an unsegmented-source
ablation.

### Task labeling and temporal crops

For the full task, retain the folding and bin-placement frames and use the
dataset prompt exactly:

```text
Assemble the cardboard box and put it into the bin
```

For a fold-only experiment:

1. make a separate logical-episode view;
2. end it at the completed-fold boundary and before any bin-placement frame
   (normally the manifest uses
   `end_frame = fold_complete_frame + 1`, because `end_frame` is exclusive);
3. let the splitter validate the shifted actions and install the final no-op;
   and
4. relabel it consistently, for example:

```text
Fold the cardboard box
```

Merely replacing the full-task string while retaining motions that put the box
in the bin creates contradictory supervision. Conversely, cropping the motion
without changing the prompt removes the prompt's stated outcome. Keep one
prompt and one temporal definition per task variant, and do not mix variants
silently in the same evaluation.

The splitter writes this per-episode task label and the data adapter maps it to
the training prompt. At deployment, send that exact same fold-only prompt on
every request. For protection against a client accidentally omitting it,
register a dedicated fold-only config whose `default_prompt` is also
`Fold the cardboard box`.

The registered `_long_episode` configs intentionally retain the original
full-task source episodes and task labels. They are not fold-only configs.
Changing only a long config's prompt is contradictory.

## Model representations

Define `T^A_B` as the pose of frame `B` expressed in frame `A`. Both configs
first convert `xyzw` quaternions to a paired 6D rotation representation.

### Absolute policy state: 20 values

The state is identical for the primary and baseline configs:

```text
[
  left_abs_xyz[3], left_abs_rot6d[6], left_current_gripper[1],
  right_abs_xyz[3], right_abs_rot6d[6], right_current_gripper[1],
]
```

| Slice | Meaning |
| --- | --- |
| `0:3` | left TCP absolute translation in marker/world |
| `3:9` | left TCP absolute rotation, 6D |
| `9` | left current absolute gripper, `1 = open` |
| `10:13` | right TCP absolute translation in marker/world |
| `13:19` | right TCP absolute rotation, 6D |
| `19` | right current absolute gripper, `1 = open` |

### Primary relative action: 20 values per waypoint

For query time `t`, arm `r`, and future waypoint `k`, the raw dataset has
already shifted the first action target to `t + 1`. The primary config computes

```text
T_rel[r, t, k] = inverse(T_world_tcp[r, t]) @ T_world_tcp[r, t + k + 1]
```

and encodes:

```text
[
  left_rel_xyz[3], left_rel_rot6d[6], left_future_gripper_abs[1],
  right_rel_xyz[3], right_rel_rot6d[6], right_future_gripper_abs[1],
]
```

The translation is expressed in the current TCP frame because this is a true
SE(3) relative transform, not component-wise position/quaternion subtraction.
Each arm has its own synchronized query-time anchor. Every waypoint in a
predicted chunk uses the **same** saved anchor:

```text
T_world_tcp_pred[r, k] = T_world_tcp_query[r] @ T_rel_pred[r, k]
```

Do not compose waypoint `k` against the robot pose reached at waypoint
`k - 1`, and do not replace the saved anchor with a moving live pose during
open-loop prefix execution. Capture new anchors only when the policy is
queried again.

### Absolute-action baseline: 20 values per waypoint

The baseline encodes:

```text
[
  left_abs_xyz[3], left_abs_rot6d[6], left_future_gripper_abs[1],
  right_abs_xyz[3], right_abs_rot6d[6], right_future_gripper_abs[1],
]
```

Its outputs are already world-frame TCP targets. Within either the logical
pair or the long pair, it uses the same state, rotation representation,
cameras, prompt, episode construction, horizon, and hyperparameters as the
relative primary so that pairwise comparison isolates action relativity.

### Rotation and internal padding

The implemented 6D convention is the first two **rows** of the rotation matrix,
flattened row-major. Its paired decoder orthonormalizes those rows and projects
the result to SO(3). Use the repository implementation in both offline
preprocessing and online serving; do not mix it with a column-based decoder or
transpose the decoded matrix. Tests must cover identity, random rotations,
near-degenerate 6D vectors, and quaternion sign equivalence (`q` and `-q`).

The physical state and action dimensions are 20. π0.5 zero-pads them to its
native 32-dimensional model tensors internally and slices predictions back to
20 before the robot-specific output transform. The last 12 values have no
robot meaning and must never reach a controller or enter 20D norm stats.

## Cameras and visual preprocessing

The two dataset streams are:

| Dataset key | Policy slot | Source |
| --- | --- | --- |
| `left_head` | `left_wrist_0_rgb` | left Insta360 X5 `front_equi` |
| `right_head` | `right_wrist_0_rgb` | right Insta360 X5 `front_equi` |

The two source cameras are the same model and projection. Their images are
384 × 384 `front_equi` fisheye views at nominal 29.97 Hz. The π0.5
base/exterior slot is filled with a placeholder and marked absent in the image
mask; it is not a third real camera.

Deployment must use the same camera model and projection as the demonstrations:

```text
raw fisheye -> same fixed crop/resize/pad -> π0.5 image tensor
```

Do **not** rectify the deployment fisheye images to pinhole unless the entire
training dataset is regenerated with that exact rectification. Do not rectify
one side only. Before training and again before robot execution, compare:

- camera model, serial/configuration, `front_equi` projection, and distortion;
- native resolution, aspect ratio, crop, resize, interpolation, and padding;
- left/right ordering and timestamp synchronization;
- mount extrinsics, roll/pitch/yaw, tool visibility, and occlusions; and
- field of view and approximate pixels per known workspace distance.

Log a transformed training pair and a transformed live pair side by side. The
same object should occupy comparable pixels with matching handedness and tool
placement.

## World, base, flange, and TCP transforms

For each arm `r`, FK supplies the flange pose in its own base,
`T^B_r_F_r`. Calibration supplies `T^W_B_r`, and the measured fixed tool
transform supplies `T^F_r_TCP_r`. The live state sent to the policy is:

```text
T_world_tcp[r] =
    T_world_base[r] @ T_base_flange[r] @ T_flange_chopstick_tip[r]
```

The complete command path is:

```text
Franka joints
  -> FK in each robot base
  -> fixed flange-to-chopstick-tip transform
  -> calibrated common marker/world TCP poses
  -> absolute state20
  -> π0.5 action20
  -> (relative config only) policy output transform composes the saved
     query-time TCP anchors
  -> absolute world TCP targets
  -> transform into each Franka base
  -> convert TCP target to flange target if required by the controller
  -> Cartesian controller or online IK
```

For a flange-target controller:

```text
T_world_flange_des =
    T_world_tcp_des @ inverse(T_flange_tcp)

T_base_flange_des =
    inverse(T_world_base) @ T_world_flange_des
```

For a controller that natively accepts the calibrated TCP:

```text
T_base_tcp_des = inverse(T_world_base) @ T_world_tcp_des
```

Never feed a chopstick-tip target to a flange controller without the fixed tool
transform. Keep separate calibration records for the left and right robot
bases, and stop if calibration, FK, timestamps, or camera frames are stale.

## Blank H200 machine setup

The upstream openpi project is tested on Ubuntu 22.04. Full JAX fine-tuning
requires more than 70 GB of GPU memory; the portable defaults target one H200
with `fsdp_devices=1` and batch size 32. Expose one H200 with
`CUDA_VISIBLE_DEVICES=0` in the commands below. The current JAX script supports
multi-device meshes on one node, but not multi-node training.

Start with a working NVIDIA driver:

```bash
nvidia-smi
```

System CUDA libraries are not required by openpi; the locked `jax[cuda12]`
environment installs user-space CUDA dependencies. A compatible NVIDIA driver
is still required.

On a blank Ubuntu machine:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl ffmpeg
git lfs install

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone --recurse-submodules https://github.com/Destiny000621/openpi.git
cd openpi
git checkout umi-dual-franka
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Clone or move the repository onto local SSD with enough space for the dataset,
π0.5 base checkpoint, two independent norm-stat trees, and experiment
checkpoints. The configs use `./assets` and `./checkpoints`, resolved from the
repository working directory. Always run commands from the repository root.

The upstream checkpoint downloader caches in `~/.cache/openpi` by default. A
large local SSD cache can be selected before any download:

```bash
export OPENPI_DATA_HOME=/mnt/localssd/openpi-cache
export HF_HOME=/mnt/localssd/huggingface
export HF_LEROBOT_HOME=/mnt/localssd/lerobot_home

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME"
```

If access requires authentication, run `uv run huggingface-cli login`.
Download the pinned source revision into the layout expected by the splitter:

```bash
uv run huggingface-cli download byang11259/cardboard_box_tcp_curated \
  --repo-type dataset \
  --revision a366d2b92723d795b6b93e6303f57708e87b63ea \
  --local-dir "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated"
```

For the logical path, annotate next, then run the dry-run and materialization
commands above and record content hashes for the local
train/validation/test datasets. The long path points directly at this pinned
source download and does not run the splitter.

Verify JAX sees the intended devices:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

The expected output is the GPU backend and one H200. Do not continue if the
device count is incompatible with `fsdp_devices=1` or if either
camera/calibration check fails. For the recommended logical path, also stop if
materialization and grouped-split review are incomplete. The long path is the
only intentional unsegmented exception.

## CPU preflight

The focused policy tests do not require a GPU. Force CPU mode so this step is
safe on a login node:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run pytest -q src/openpi/policies/umi_dual_franka_policy_test.py
```

Run the config and logical-episode splitter tests as well:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run pytest -q \
    src/openpi/training/config_test.py \
    scripts/split_cardboard_box_lerobot_v21_test.py \
    scripts/inspect_umi_dual_franka_dataset_test.py \
    scripts/eval_umi_dual_franka_open_loop_test.py
```

Inspect full, unnormalized 50-step chunks and transform round trips before
computing stats. Recommended logical relative and absolute:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative \
  --repo-id local/cardboard_box_tcp_curated_logical_train \
  --num-samples 32

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute \
  --repo-id local/cardboard_box_tcp_curated_logical_train \
  --num-samples 32
```

Original long-source relative and absolute:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_long_episode \
  --repo-id byang11259/cardboard_box_tcp_curated \
  --num-samples 32

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute_long_episode \
  --repo-id byang11259/cardboard_box_tcp_curated \
  --num-samples 32
```

The inspector verifies raw16 shapes, `action[t] = state[t + 1]`, transformed
state20/action20 shapes, image masks, and raw16 round trips. On the long repo,
its reported episode count means **source** episodes. It samples chunks that
fit within those source bounds; it does not infer or mask physical-box
boundaries. Review known boundary-adjacent chunks separately.

These are schema/transform preflights, not a substitute for a small
dataloader inspection, GPU compilation test, camera dry run, or
controller-disabled hardware rehearsal.

## Compute fresh normalization statistics

The published dataset's raw absolute-16 statistics (for example,
`meta/stats.json`) are invalid for the transformed absolute-state-20 and
relative-action-20 distributions. The absolute baseline also changes
quaternions to 6D rotations and therefore needs new statistics.

Every config needs stats computed after its final action transform. Never share
stats across relative versus absolute or logical versus long episode paths.

### Logical-episode stats (recommended)

Compute these after logical-episode construction and grouped split selection.
The configured repo ID selects the logical **train** dataset, so
validation/test episodes are not included. The splitter's terminal target is
an explicit in-episode no-op rather than an action from the next box. If a
separate padding mask is introduced, the norm-stat pass and training loader
must apply the same mask.

Relative config:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative
```

Absolute baseline:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute
```

The two files are written separately:

```text
assets/pi05_umi_dual_franka_cardboard_box_relative/local/cardboard_box_tcp_curated_logical_train/norm_stats.json
assets/pi05_umi_dual_franka_cardboard_box_absolute/local/cardboard_box_tcp_curated_logical_train/norm_stats.json
```

Do not copy, symlink, or share one with the other. Before training, inspect
`mean`, `std`, `q01`, and `q99` for all 20 physical dimensions and confirm that
the asset ID is `local/cardboard_box_tcp_curated_logical_train`. Tiny spreads,
unexpected translation magnitudes, gripper polarity changes, or padded
dimensions in the physical stats are stop signs.

### Original long-source stats (ablation)

These commands intentionally traverse the unsegmented source episodes through
the stock 50-step action-chunk path:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_long_episode

uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute_long_episode
```

They write two additional, independent files:

```text
assets/pi05_umi_dual_franka_cardboard_box_relative_long_episode/byang11259/cardboard_box_tcp_curated/norm_stats.json
assets/pi05_umi_dual_franka_cardboard_box_absolute_long_episode/byang11259/cardboard_box_tcp_curated/norm_stats.json
```

The long relative stats include any fixed-anchor jumps across physical-box
boundaries. Both long stats include source-end clamping/repetition. Treat
those distributions as part of the ablation; do not replace them with logical
stats to make the run appear better behaved. Inspect all four files and record
their asset IDs with the experiments.

## JAX training on H200

### Logical-episode training (recommended)

Run the two experiments separately with the same dataset revision, grouped
split, seed, batch construction, and evaluation protocol. First compile the
complete data/model path and execute two steps with a small batch:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_umi_dual_franka_cardboard_box_relative \
  --exp-name=fold_box_relative_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_umi_dual_franka_cardboard_box_absolute \
  --exp-name=fold_box_absolute_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled
```

Inspect the loaded norm-stat paths, first camera batch, transformed tensor
shapes/ranges, finite losses, and checkpoints before starting full runs.

Primary relative representation:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_umi_dual_franka_cardboard_box_relative \
  --exp-name=fold_box_relative_v1
```

Absolute-action baseline:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_umi_dual_franka_cardboard_box_absolute \
  --exp-name=fold_box_absolute_v1
```

These use the registered one-H200 defaults: batch 32, one FSDP device, eight
workers, and 5,000 steps. Add `--no-wandb-enabled` if the machine should not
contact Weights & Biases.

Use `--resume` to continue an existing experiment. Use `--overwrite` only when
deliberately deleting an experiment directory. Watch the first logged camera
batch, transformed state/action ranges, loss, gradient norm, and device memory.
Stop on NaNs, wrong camera ordering, unmasked base images, cross-box chunks, or
implausible Cartesian magnitudes.

### Original long-source training (ablation)

After computing the two long-source norm-stat files, compile and execute two
steps for each long config:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_relative_long_episode \
  --exp-name=fold_box_relative_long_episode_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_absolute_long_episode \
  --exp-name=fold_box_absolute_long_episode_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled
```

Confirm that each smoke run loads its own
`byang11259/cardboard_box_tcp_curated` asset tree. Then run the full
long-source experiments:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_relative_long_episode \
  --exp-name=fold_box_relative_long_episode_v1

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_absolute_long_episode \
  --exp-name=fold_box_absolute_long_episode_v1
```

The same one-H200 defaults apply. Cross-box chunks are expected here; stop
only if behavior differs from the documented stock semantics, values are
non-finite, or transforms/cameras are wrong. Compare long relative versus long
absolute to isolate the action representation within the unsegmented source
path. Comparing a long config with a logical config also changes episode
construction, sample distribution, terminal targets, and train/validation
availability, so it is not a pure action-representation ablation.

## Rank checkpoints on held-out logical episodes

This section applies to the recommended logical configs. The long configs train
on the full source repo, including source episodes from which the logical
validation set is derived; evaluating them there is not held out and must not
be labeled validation.

Use only `local/cardboard_box_tcp_curated_logical_validation`, whose source
episodes and collection sessions are absent from training. The evaluation
script reconstructs the training transforms, selects only complete 50-step
chunks, and reports normalized 20D MSE overall and for translation, 6D
rotation, and gripper dimensions. It loads normalization stats from each
checkpoint's embedded `assets/<asset_id>` tree, matching `serve_policy.py`;
stale or missing local `./assets` stats do not define the score.

Rank relative checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/eval_umi_dual_franka_open_loop.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative \
  --repo-id local/cardboard_box_tcp_curated_logical_validation \
  --episodes all \
  --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_relative/fold_box_relative_v1/1000 \
  --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_relative/fold_box_relative_v1/4999
```

Rank absolute checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/eval_umi_dual_franka_open_loop.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute \
  --repo-id local/cardboard_box_tcp_curated_logical_validation \
  --episodes all \
  --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_absolute/fold_box_absolute_v1/1000 \
  --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_absolute/fold_box_absolute_v1/4999
```

Pass only checkpoint steps that exist. Lower held-out MSE is useful for ranking
checkpoints **within a representation**, but it is not a hardware success rate
and does not replace calibration replay, safety checks, or controlled robot
evaluation. Because the two logical configs have distinct normalization
statistics, report each component and physical-space rollout metrics when
comparing the representations.

## Serve a matching config with JAX

Replace `<step>` with a saved checkpoint step. The config and checkpoint must
match exactly; this is also how the server locates the correct norm stats.

Recommended logical relative policy:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_relative \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_relative/fold_box_relative_v1/<step>
```

Recommended logical absolute baseline:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_absolute \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_absolute/fold_box_absolute_v1/<step>
```

Long-source relative ablation:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_relative_long_episode \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_relative_long_episode/fold_box_relative_long_episode_v1/<step>
```

Long-source absolute ablation:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_absolute_long_episode \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_absolute_long_episode/fold_box_absolute_long_episode_v1/<step>
```

Serving support does not make a long-source checkpoint deployment-qualified.
Keep it in offline/controller-disabled evaluation until it passes the same
held-out, calibration, and safety gates as the recommended path.

The WebSocket server listens on all interfaces and defaults to port 8000.
Restrict access with the machine firewall or a private robot network; the
server does not replace a robot safety layer.

## Remote WebSocket robot deployment

Install the lightweight client in the robot environment:

```bash
cd "$OPENPI_ROOT/packages/openpi-client"
pip install -e .
```

Create one `WebsocketClientPolicy` outside the control loop and connect it to
the H200 server:

```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(
    host="POLICY_SERVER_IP",
    port=8000,
)
```

At each policy query, send synchronized, unnormalized observations:

```python
observation = {
    "observation/state": raw_absolute_state16,
    "observation/left_head": left_front_equi_uint8_hwc,
    "observation/right_head": right_front_equi_uint8_hwc,
    "prompt": "Assemble the cardboard box and put it into the bin",
}
action_chunk = policy.infer(observation)["actions"]
```

The state must describe both live chopstick-tip TCPs in the calibrated common
world frame, have a floating dtype, and use `xyzw`, left-then-right, and
`1 = open`. Prefer native 384 × 384 `uint8` HWC images; the adapter also accepts
CHW and validated floating image ranges before the shared resize/pad. Images
must be timestamp-matched to the state. The server applies checkpoint
normalization; the client must not pre-normalize state or actions.

For **all four** configs the WebSocket response is an absolute raw action chunk
with shape `[50, 16]` and the same left/right, `xyz + xyzw + gripper` layout as
the raw state. For the relative config, the server-side output transform uses
the unnormalized state20 derived from that query as one immutable anchor for
every waypoint, then returns absolute world-frame targets. Do not compose
these returned targets a second time on the client. Either absolute baseline
also returns absolute world-frame targets, directly decoded from action20.

Log the state16 sent with each request as the relative policy's anchor record,
then apply the world/base/flange command chain to the returned absolute
targets. Regardless of representation, reject malformed, late, non-finite,
out-of-workspace, or discontinuous responses before the controller sees them.

## Action horizon and receding-horizon execution

Fifty waypoints at 29.97 Hz span approximately:

```text
50 / 29.97 ≈ 1.67 seconds
```

That is a prediction horizon, not permission to execute 1.67 seconds open
loop. Start with a short, conservative prefix under controller-disabled replay
and low-speed testing. Select the executed prefix from measured:

- network and inference latency;
- camera/state age and synchronization;
- Cartesian controller tracking error;
- contact dynamics and force/torque margins;
- inter-arm coordination and collision clearance; and
- how quickly model predictions become stale.

After the prefix, acquire a new synchronized observation, query again, and
re-anchor both arms. Log query timestamp, state/image timestamps, inference
latency, chosen prefix length, anchors, predicted chunk, filtered commands, and
executed commands.

If control frequency is changed, resample images, state, actions, timestamps,
episode boundaries, and masks together; then choose a horizon representing the
intended physical duration and recompute every affected config's norm stats.
Changing only the nominal frequency or horizon silently changes supervision.

## Safety gates

Use a controller-independent safety layer for both arms. Before any powered
run, require:

- verified translation units, quaternion order, gripper polarity, arm order,
  and tool transform;
- current world-to-base calibration for both robots;
- workspace and table/fixture exclusion volumes;
- Cartesian translation/rotation velocity and acceleration limits;
- force/torque and contact limits appropriate for cardboard interaction;
- Franka joint position, velocity, torque, and singularity margins;
- self-collision, inter-arm collision, tool collision, and camera collision
  checks;
- IK convergence and continuity checks when IK is used;
- finite-value, jump, timestamp-age, and network-timeout checks;
- an accessible hardware emergency stop and a tested software stop; and
- stop-on-stale behavior for either camera, FK, calibration, policy response,
  or controller state.

Commission in this order:

1. offline transform and chunk replay;
2. policy server with recorded observations;
3. controller disabled, commands logged only;
4. single waypoint at very low speed with no box;
5. one arm at a time in a clear workspace;
6. both arms at low speed with collision monitoring;
7. compliant contact with an expendable box; and
8. short receding-horizon prefixes, increased only after log review.

Do not enable both robots merely because model inference and IK succeed.

## Required validation and tests

The preflight suite and experiment checklist must cover:

- raw-state length 16, left/right ordering, quaternion `xyzw`, and
  `action[t] = state[t + 1]`;
- absence of invented joint-position, joint-velocity, or joint-target policy
  dimensions;
- measured translation units and the no-double-`-90°` rule;
- raw absolute16 → state/action20 → raw absolute16 round trips;
- quaternion sign equivalence (`q` and `-q`);
- pure tool-frame translations and rotations with known signs;
- identity and random-rotation 6D encode/decode round trips;
- identical fixed anchor use for every waypoint in a relative chunk;
- independent, synchronized left and right anchors;
- grippers remaining future absolute values with `1 = open`;
- physical 20D stats and correct zero-padding/slicing to/from model 32D;
- four separate norm-stat trees across representation and episode path;
- online WebSocket transforms matching offline training transforms;
- exactly two real camera streams, correct ordering, and an absent/masked base
  slot;
- identical fisheye projection and preprocessing between data and deployment;
- for logical configs, no sampled query, shifted action, or 50-step chunk
  crossing a physical-box boundary;
- for long configs, exact `0..49 / fps` action deltas, source-episode clamping,
  ignored padding/boundary masks, and intentionally possible cross-box chunks;
- the materializer's final absolute no-op, frame-accurate two-video trims, and
  rewritten LeRobot indices/timestamps;
- when the masked alternative is used, identical terminal/boundary masks in
  sampling, norm stats, and loss;
- session/source-grouped train/validation/test splits with no sibling leakage;
- fold-only crops containing no bin-placement motion;
- full-task crops and prompts retaining bin placement;
- held-out logical-episode checkpoint ranking with no train-session leakage;
- world/base/flange/TCP calibration-chain round trips for both arms;
- stale-data, non-finite, workspace, IK, and collision rejection; and
- controller-disabled and low-speed hardware rehearsals.

Archive the selected episode path, source revision, logical manifest and
derived-data hash when applicable, norm stats, config name, checkpoint step,
git commit, calibration IDs, camera settings, and safety-limit version with
every reported result.

## Repository entry points

- [Logical-episode splitter](../scripts/split_cardboard_box_lerobot_v21.py)
- [Reviewed-manifest template](../configs/cardboard_box_segments.example.json)
- [Dual-Franka UMI policy transforms](../src/openpi/policies/umi_dual_franka_policy.py)
- [Training data and model configs](../src/openpi/training/config.py)
- [Dataset/chunk transform inspector](../scripts/inspect_umi_dual_franka_dataset.py)
- [Held-out open-loop checkpoint ranking](../scripts/eval_umi_dual_franka_open_loop.py)
- [Policy transform tests](../src/openpi/policies/umi_dual_franka_policy_test.py)
- [Config tests](../src/openpi/training/config_test.py)
- [Splitter tests](../scripts/split_cardboard_box_lerobot_v21_test.py)
- [Inspector tests](../scripts/inspect_umi_dual_franka_dataset_test.py)
- [Checkpoint-ranking tests](../scripts/eval_umi_dual_franka_open_loop_test.py)

## Official and source references

- [Cardboard-box dataset card and schema](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated#schema)
- [Dataset coordinate conventions](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated#coordinate-conventions-read-before-using)
- [Dataset relative-representation notes](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated#using-the-hy-vla--relative-representation)
- [Dataset normalization notes](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated#normalization)
- [Pinned dataset revision](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated/tree/a366d2b92723d795b6b93e6303f57708e87b63ea)
- [Hy-VLA SE(3) transform implementation, pinned revision](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/utils/transform_utils.py#L16-L123)
- [Hy UMI dataset config, pinned revision](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/config/dataset/umi_lance.yaml#L28-L50)
- [Hy UMI dataset loader, pinned revision](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L520-L630)
- [Physical Intelligence π0.5 overview](https://www.physicalintelligence.company/blog/pi05)
- [Official openpi repository](https://github.com/Physical-Intelligence/openpi)
- [Pinned project-base OpenPI LeRobot chunk construction](https://github.com/Destiny000621/openpi/blob/59ce2725e887d44c36dd1a3d3106d00d8ad6cd5e/src/openpi/training/data_loader.py#L135-L147)
- [Pinned LeRobot episode clamp/padding behavior](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py#L665-L678)
- [Local openpi normalization guide](norm_stats.md)
- [Local openpi remote-inference guide](remote_inference.md)

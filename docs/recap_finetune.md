# RECAP fine-tune of pi0.5/pi0.6 on YAM bimanual (PiStar path)

End-to-end recipe for **RECAP**, the offline RL algorithm in **pi0.6**, using
[ybpy/pistar](https://github.com/ybpy/pistar) — the direct implementation of
pi0.6 RECAP — on top of a pi0.5/pi0.6 checkpoint, starting from the dataset
produced by `limb/docs/recap_collection.md`.

> **Which implementation this doc follows.**
> [**ybpy/pistar**](https://github.com/ybpy/pistar) — PiStar (`π★ ≈ pi0.6`) is
> a **fork of openpi** (JAX/flax.nnx) where RECAP is the native training
> algorithm. The advantage is a tokenized `adv_ind ∈ {"positive","negative",
> "none"}` consumed by openpi's standard tokenizer transform — so
> **`serve_policy.py` runs vanilla** with `adv_ind_input="positive"` in the
> model config; **limb's `OpenPIClient` and prompt are unchanged**. PiStar
> also ships `control_your_robot/` for real-robot deployment.
>
> The JAX backend matches `docs/yam_finetune.md`'s existing SFT path
> (`scripts/train.py`), so the SFT bootstrap, the PiStar fine-tune, and
> serving are all in the same stack.
>
> **Secondary references:**
>
> - [RLinf `examples/recap/`](https://github.com/RLinf/RLinf) — a **PyTorch**
>   re-implementation of pi0.6 RECAP. Same value-model architecture (SigLIP
>   + Gemma3 + 201-atom C51 head) and same tokenized-advantage conditioning,
>   but different labeling (Critic-Expert + N-step lookahead → top-30%
>   quantile binarization). Sim-validated only (LIBERO). The vendored
>   standalone `openpi/scripts/recap/compute_returns.py` belongs to this
>   alternative path; not used here.
> - [MINT-SJTU/Evo-RL](https://github.com/MINT-SJTU/Evo-RL) — real-robot
>   RECAP-variant with a *different* conditioning (advantage tag appended to
>   task text). Useful only for collection-protocol intuition.

---

## How PiStar conditions on advantage (from the source)

`src/openpi/transforms.py:264-276`:

```python
if self.adv_ind_input:
    if (adv_ind := data.pop("adv_ind", None)) is None:
        raise ...
else:
    data.pop("adv_ind", None)
    adv_ind = None
...
tokens, token_masks = self.tokenizer.tokenize(
    prompt, state, adv_ind, adv_ind_dropout=self.adv_ind_dropout)
```

The advantage is a **separate input** to the tokenizer (`prompt`, `state`,
`adv_ind`), **not** appended to the task string. The tokenizer maps the
string `"positive"`/`"negative"`/`"none"` to its own tokens. During training,
`adv_ind_dropout=True` randomly drops the advantage token (the pistar
analogue of CFG `unconditional_prob`). At inference, the model config sets
`adv_ind_input="positive"` and the same tokenizer path inserts the positive
token.

Practical consequence: **no CFG sampler shim, no two-pass guidance machinery
in the server**. `scripts/serve_policy.py` runs the standard openpi forward,
and the conditioning rides through the standard transform. limb's prompt
stays plain.

The `pistar` flag on `Pi0Config` enables `adv_ind_input` end-to-end
(`src/openpi/training/config.py:139`: `adv_ind_input=model_config.pistar`).

---

## Pipeline (PiStar's data closed loop, for YAM)

```
[limb]   collect demos (gello)               ─┐
[limb]   convert-lerobot --pistar             │  Stage 1
[pistar] train.py  (initial PiStar)           │
                                              │
[limb]   serve initial checkpoint via limb    │
[limb]   collect rollouts (DAgger)            │  Stages 2-3
[limb]   convert-lerobot --pistar             │
[pistar] merge_datasets.py (demo + rollout)   │
                                              │
[pistar] train_value.py  (VLM value model)    │  Stage 4
[pistar] label_advantage_from_vlm.py          │  Stage 5
         (rewrites adv_ind on intervention=0  │
          rollout frames)                     │
                                              │
[pistar] train.py  (continue PiStar with      │  Stage 6
         adv_ind-conditioned tokens)          │
                                              │
[openpi] serve_policy.py + limb's vanilla     │
         OpenPIClient (adv_ind_input=positive)─┘
```

The dataset is reused round after round; only `adv_ind` (on
`intervention=0` rollout frames) is rewritten by the VLM each iteration.

---

## Implementation checklist

**limb** (`/home/ssc/Desktop/research/limb`):
- [ ] Add the **five pistar columns** to `limb/data/convert_lerobot.py`
      behind `--pistar` (helpers in `episode_utils.py`). See
      `limb/docs/recap_collection.md` § What's left to do.

**openpi (PiStar fork)** — installed alongside this repo:
- [ ] Clone PiStar: `git clone https://github.com/ybpy/pistar` and follow
      its `README.md` install instructions (it is itself an openpi fork —
      keep it in a separate venv from your existing openpi).
- [ ] **YAM policy file** in `src/openpi/policies/` — mirror
      `libero_policy.py` / `piper_policy.py` for YAM bimanual. Reuse the
      repack from `docs/yam_finetune.md`
      (`head_camera → cam_high`, `left_wrist_camera → cam_left_wrist`,
      `right_wrist_camera → cam_right_wrist`), AlohaInputs with
      `adapt_to_pi=False`. The file must pass `adv_ind` through (like
      `libero_policy.py:83-84`):
      ```python
      if "adv_ind" in data:
          inputs["adv_ind"] = data["adv_ind"]
      ```
- [ ] **YAM `TrainConfig`** — clone `pi05_yam_vial_30fps` from your existing
      openpi `config.py` into PiStar's `config.py`, with `Pi0Config(pi05=True,
      pistar=True)` to enable `adv_ind` tokenization, and
      `adv_ind_dropout: bool = True`.
- [ ] (`v3.0 → v2.1`) Run `openpi/scripts/convert_v3_to_v21.py` on the
      `--pistar` dataset; verify all five columns (including the **string**
      `adv_ind`) carry through.

**Serving** (`scripts/serve_policy.py`): no change. The PiStar checkpoint
serves through the standard openpi path; limb's `OpenPIClient` is unchanged
(see [§ Serving](#serving-back-to-limb)).

---

## Prerequisites

```bash
# PiStar (the openpi fork). Use its install path.
git clone https://github.com/ybpy/pistar
cd pistar
git submodule update --init --recursive
uv venv --python 3.11.9 /path/to/pistar-venv
source /path/to/pistar-venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --active
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install -r pistar_requirements.txt

# auth
gcloud auth application-default login          # pi05_base weights
huggingface-cli login                          # dataset pull + checkpoint push
```

You also need:

- A pi0.5/pi0.6 checkpoint to start from (`pi05_base` works). PiStar's first
  train.py call lifts it; the second train.py call (after VLM relabeling)
  refines it with `adv_ind` conditioning.
- A `--pistar` dataset from `limb convert-lerobot` (demo set + at least one
  rollout set), each converted v3.0→v2.1.

---

## Stage 1 — initial PiStar checkpoint (`scripts/train.py`)

Train an SFT-quality initial policy from your demo data alone (intervention=1
frames, adv_ind="positive" throughout). Use **your YAM TrainConfig** with
`pistar=True`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_yam_pistar \
    --exp-name=v0 \
    --resume=false
```

This is the same `scripts/train.py` your `docs/yam_finetune.md` SFT uses —
just a different config (`pistar=True`). Output: a PiStar checkpoint that
already conditions on `adv_ind`, trained only on `"positive"` so far.

---

## Stage 2 — collect rollouts (limb DAgger)

Serve the v0 checkpoint to limb (see [§ Serving](#serving-back-to-limb)),
then collect rollouts with the DAgger config:

```bash
uv run limb record \
  --config-path configs/yam_dagger_pi0_bimanual.yaml \
                configs/dagger_collection.yaml
```

`s`/SPACE per episode for success/failure. Aim for ~50 rollouts with mixed
outcomes (see `recap_collection.md` hygiene).

Convert with `--pistar` (Stage 0 → pistar schema):

```bash
uv run limb convert-lerobot \
  --input-dir recordings/<session> \
  --output-dir datasets/<task>_rollout_v1 --target-fps 30 --pistar

uv run python openpi/scripts/convert_v3_to_v21.py \
  --src=datasets/<task>_rollout_v1 \
  --dst=datasets/<task>_rollout_v1_v21
```

---

## Stage 3 — merge demo + rollout (`scripts/merge_datasets.py`)

```bash
uv run python scripts/merge_datasets.py \
  --inputs datasets/<task>_demo_v21 datasets/<task>_rollout_v1_v21 \
  --output datasets/<task>_merged_v1
```

PiStar's merger keeps only `image`, `wrist_image`, `state`, `actions`,
`intervention`, `value_label`, `reward`, `reward_label`, `adv_ind` plus
`timestamp`, `frame_index`, `episode_index`, `index`, `task_index`. It is
**pure merging** — no field-filling, no recomputation — so the source
datasets must each carry all five RECAP columns. limb's `--pistar` flag
guarantees that.

---

## Stage 4 — train the VLM value model (`scripts/train_value.py`)

Value model = SigLIP (400M) + Gemma3 (270M) + **201-atom C51 head over
[-1, 0]** (see `src/openpi/models/value_model.py:21-23` —
`NUM_ATOMS = 201`, `V_MIN = -1.0`, `V_MAX = 0.0`). Trained on `value_label`.

```bash
uv run python scripts/train_value.py pi05_yam_pistar_value \
  --exp-name=v1 \
  --dataset.repo_id=<user>/<task>_merged_v1
```

`value_label` is filled per-frame by `limb convert-lerobot --pistar`:
linear ramp `-(T-1-t)/T` for SUCCESS episodes, constant `-1.0` for FAILURE.
The VLM learns to predict these from `(image, wrist_image, state, task)`.

**Verify** with `scripts/check_value_data.py` (pistar ships this) before
training to confirm the data shape and value distribution.

---

## Stage 5 — relabel rollout `adv_ind` (`scripts/label_advantage_from_vlm.py`)

Walks the merged dataset, runs VLM value inference, and computes N-step
advantage for **rollout** rows (intervention=0). Demo rows (intervention=1)
keep `adv_ind="positive"` untouched.

```bash
uv run python scripts/label_advantage_from_vlm.py \
  --data_dir datasets/<task>_merged_v1 \
  --checkpoint_dir <path to Stage 4 value checkpoint>
```

From `label_advantage_from_vlm.py`'s header docstring (verbatim):

```
1) Classify each episode by `intervention`: all-1 episodes are demos and are skipped;
   episodes with any 0 are rollouts and are fully relabeled.
2) Run VLM value inference for rollout rows and the lookahead endpoint rows
   needed to compute their N-step advantage.
3) Convert 201-dim logits -> softmax -> expectation over supports in [-1.0, 0.0].
4) Compute N-step Advantage per rollout time step:
   A_t = sum_{k=0}^{N-1} r_{t+k} + V_{t+N} - V_t
5) Compute the percentile threshold over rollout advantages of non-intervention steps.
6) For rollout rows only:
   - if `intervention = 1`, set `adv_ind = positive`
   - if `intervention = 0`, mark the configured top percentage as `positive`,
     otherwise `negative`
   Existing labels on rollout rows are overwritten; demo rows are preserved.
```

After this, every frame's `adv_ind` is in `{"positive","negative","none"}`
and the dataset is ready for PiStar continuation.

---

## Stage 6 — continue PiStar fine-tune (`scripts/train.py`)

Same `train.py`, same YAM TrainConfig, but now the dataset is relabeled and
`adv_ind_dropout` ensures the policy learns under both conditioned and
dropped tags:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/train.py pi05_yam_pistar \
    --exp-name=recap_v1 \
    --policy.pretrained_dir=<Stage 1 checkpoint> \
    --dataset.repo_id=<user>/<task>_merged_v1   # after Stage 5 relabel
```

Output: a PiStar checkpoint conditioned on `adv_ind`, served the standard way.

---

## Serving back to limb

The Stage-6 output is a standard openpi JAX checkpoint. The advantage
conditioning is in the **tokenizer** (`adv_ind_input="positive"` in the
model config), so `scripts/serve_policy.py` runs the standard forward — no
CFG sampler, no guidance scale flag.

```bash
uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \
  --policy.config=pi05_yam_pistar \
  --policy.dir=<...>/pi05_yam_pistar/recap_v1/<step>
```

limb's `OpenPIClient` and `inner_policy.obs_transform.prompt` are
**unchanged** — same task instruction as your SFT serving. The
positive-advantage token is inserted automatically by the tokenizer because
`adv_ind_input="positive"` is set in the model config (`train_value.py` /
the `pistar` flag).

The advantage-conditioning channel does not appear in limb's wire protocol
at all; it lives entirely inside the openpi tokenizer.

---

## Iterate (closed loop)

```
Stage 1 v0 → serve to limb → DAgger rollouts v1 → merge → Stages 4-6 → v1
v1         → serve to limb → DAgger rollouts v2 → merge → Stages 4-6 → v2
...
```

Each round, the VLM relabels `adv_ind` on the latest rollout frames; demo
frames are preserved. Stop when held-out success plateaus, intervention rate
in the latest round drops below ~10%, or Stage-6 loss flattens.

---

## RLinf alternative (PyTorch, sim-validated)

A different re-implementation of the same pi0.6 RECAP algorithm. Pipeline
shape:
`compute_returns.py` (vendored at `scripts/recap/compute_returns.py` —
needs `is_success` per-frame, *not* the pistar five columns) →
`train_value.py` → `compute_advantages.py` → `train_cfg.py` with an
`openpi_cfg_action_model` wrapping the openpi policy and a
`cfgrl_guidance_scale` at inference.

Costs vs PiStar for this stack:
- Requires a CFG-sampler shim around `serve_policy.py` (extra serving work).
- PyTorch, not JAX — diverges from `docs/yam_finetune.md`'s SFT path.
- A different dataset schema (one `is_success` bool column vs PiStar's five).

Reach for it only if you specifically need RLinf's quantile-based offline
labeling (no VLM) or want to reproduce LIBERO results.

---

## Evo-RL reference (text-tag conditioning)

The only RECAP **validated on real hardware** (SO-101, AgileX PiPER) but
with a *different conditioning*: an `Advantage: positive`/`negative` tag
appended to the **task text** (`src/lerobot/rl/acp_tags.py`). LeRobot-native,
PyTorch end-to-end. Worth reading for collection-protocol intuition
(`s`/`f` hotkeys, mixed-success hygiene); not for the conditioning
mechanism, which we explicitly don't use here.

---

## Gotchas

1. **`pistar=True` in `Pi0Config`** — without it `adv_ind_input` is `False`
   in `transforms.py:264`, the tokenizer pops and discards `adv_ind`, and
   the conditioning never reaches the model.
2. **`adv_ind` is a string column.** `convert_v3_to_v21.py` must carry it
   through; some dtype guards drop unknown non-numeric columns. Verify per
   `recap_collection.md` § Format/version.
3. **VLM value model is trained on `value_label`, not on returns.** If your
   `--pistar` converter wrote the wrong formula (e.g. success-shaped ramp
   for failure episodes), the VLM learns a broken value function and Stage
   5's advantages are noise.
4. **`adv_ind_input="positive"` at serving.** Confirm it's set in the
   served model's config. If it's `False` or `"none"`, you lose the RECAP
   conditioning silently.
5. **Don't pre-mix tasks.** The value model and percentile labeling are
   per-task; one task per dataset, or be deliberate about multi-task.
6. **Keep both success and failure episodes** — a pure-success or
   pure-failure dataset collapses the value distribution. Enforce at
   collection.

---

## See also

- `limb/docs/recap_collection.md` — data-collection + `--pistar` converter
- `docs/yam_finetune.md` — SFT pi0.5 on YAM (Stage 0 prerequisite path)
- [ybpy/pistar](https://github.com/ybpy/pistar) — the implementation this doc follows
- [ybpy/pistar/control_your_robot](https://github.com/ybpy/pistar/tree/main/control_your_robot)
  — pistar's real-robot deployment toolkit
- [RLinf `examples/recap/`](https://github.com/RLinf/RLinf) — PyTorch RECAP alternative
- [MINT-SJTU/Evo-RL](https://github.com/MINT-SJTU/Evo-RL) — real-robot reference (text-tag)

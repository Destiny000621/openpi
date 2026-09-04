from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # SubRL: also return pi0.5's pooled image embedding (RLT z_rl) from infer()
            # so the online-RL critic/actor can condition on what the VLA sees. Off by
            # default; enable with SUBRL_RETURN_EMBED=1 at serve time. SUBRL_RLTOKEN=<npz>
            # swaps the mean-pool readout for a trained RL-token encoder (same 2048-d
            # output, so learner/replay/agent are untouched); it fails loudly on a
            # missing npz rather than silently falling back to the mean-pool.
            self._return_embed = (
                os.environ.get("SUBRL_RETURN_EMBED") == "1" and hasattr(model, "extract_image_embedding")
            )
            if self._return_embed:
                rltoken_path = os.environ.get("SUBRL_RLTOKEN", "").strip()
                if rltoken_path:
                    from openpi.models import rl_token as _rl_token  # noqa: PLC0415

                    encoder = _rl_token.load_encoder(rltoken_path)  # raises if missing/invalid
                    extract_tokens = nnx_utils.module_jit(model.extract_image_tokens)

                    def _embed_with_token(rng_key, observation):  # noqa: ANN001, ANN202
                        embs, mask = extract_tokens(rng_key, observation)
                        return encoder(np.asarray(embs), np.asarray(mask))

                    self._extract_embed = _embed_with_token
                    logging.info("SubRL: serving RL-token z_rl from %s", rltoken_path)
                else:
                    # JIT the embedding extraction like sample_actions — an eager
                    # version runs a full un-jitted prefix forward per infer and
                    # stalls the robot at every replan.
                    self._extract_embed = nnx_utils.module_jit(model.extract_image_embedding)
                    logging.info("SubRL: serving mean-pool z_rl (SUBRL_RETURN_EMBED=1)")
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if not self._is_pytorch_model and getattr(self, "_return_embed", False):
            # jitted pooled image-token embedding -> [emb] (RLT z_rl); first call compiles
            outputs["image_embedding"] = np.asarray(self._extract_embed(self._rng, observation)[0], np.float32)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def get_prefix_rep(self, obs: dict) -> dict:
        """DSRL state feature: `{"prefix_rep": float32 [1, emb]}` for one observation.

        Runs the SAME prefix forward `infer` would, but stops before the flow-matching
        denoise loop, so it is strictly cheaper than a full inference. Exposed as its
        own serve method (`method="get_prefix_rep"`) because DSRL needs z_rl BEFORE it
        picks the noise it will hand to `infer` — the two calls cannot be merged.

        Deliberately independent of `SUBRL_RETURN_EMBED` / `SUBRL_RLTOKEN`: the SubRL
        hooks swap `image_embedding` for a trained RL-token readout, and DSRL's
        baseline must keep the upstream last-prefix-slot feature no matter how the
        serve is launched for SubRL. Compiled lazily on first call so serves that
        never do DSRL pay nothing.
        """
        if self._is_pytorch_model:
            raise NotImplementedError("prefix_rep is JAX-only (DSRL serves a JAX pi0.5).")
        if not hasattr(self._model, "extract_prefix_last_rep"):
            raise NotImplementedError(
                f"{type(self._model).__name__} has no extract_prefix_last_rep; "
                "get_prefix_rep needs a pi0/pi0.5 model."
            )
        if getattr(self, "_dsrl_prefix_embed", None) is None:
            self._dsrl_prefix_embed = nnx_utils.module_jit(self._model.extract_prefix_last_rep)

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        rep = np.asarray(self._dsrl_prefix_embed(self._rng, observation), np.float32)
        return {
            "prefix_rep": rep,  # [1, emb] float32 — msgpack cannot pack bfloat16
            "policy_timing": {"infer_ms": (time.monotonic() - start_time) * 1000},
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs) if noise is None else self._policy.infer(obs, noise=noise)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def get_prefix_rep(self, obs: dict) -> dict:
        """Pass-through so `--record` serves still answer DSRL's get_prefix_rep."""
        return self._policy.get_prefix_rep(obs)

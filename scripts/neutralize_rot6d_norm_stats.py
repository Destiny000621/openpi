"""Neutralize the rot6d dimensions of a config's norm-stats FILE.

Rewrites ``assets/<config>/<repo_id>/norm_stats.json`` so the rot6d dimensions
carry neutral parameters (mean 0 / std 1 / q01 -1 / q99 +1): the stock openpi
normalizer then passes rot6d through unchanged while translation and gripper
dimensions keep their data-driven statistics. A one-time backup of the original
file is written alongside as ``norm_stats.json.pre_rot6d_neutral.bak`` (never
overwritten).

Action rot6d is always neutralized. The 20-D relative-history state shares the
same ``[xyz(3), rot6d(6), gripper(1)]`` arm layout, so its rot6d dimensions are
neutralized too; the 2-D gripper-only state has no rot6d and is left alone.

This edits the stats file rather than the load path on purpose: checkpoints
embed their stats, so runs trained before/after the neutralization each keep a
self-consistent serving pipeline under the same config name.
"""

import pathlib
import shutil

import numpy as np
import tyro

import openpi.policies.umi_dual_franka_policy as umi_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config


def main(config_name: str, *, include_state: bool | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    stats_dir = pathlib.Path(config.assets_dirs) / data_config.repo_id
    stats_file = stats_dir / "norm_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"{stats_file} does not exist; compute norm stats first.")

    if include_state is None:
        state_mode = getattr(config.data, "state_mode", "full")
        include_state = state_mode == "relative_history"
    keys = ("actions", "state") if include_state else ("actions",)

    backup = stats_dir / "norm_stats.json.pre_rot6d_neutral.bak"
    if not backup.exists():
        shutil.copy2(stats_file, backup)
        print(f"backup written: {backup}")

    stats = normalize.load(stats_dir)
    patched = umi_policy.neutralize_rot6d_norm_stats(stats, keys=keys)
    normalize.save(stats_dir, patched)

    for key in keys:
        entry = patched.get(key)
        dim = None if entry is None else np.asarray(entry.mean).reshape(-1).shape[-1]
        status = "neutralized" if dim is not None and dim >= umi_policy.MODEL_STATE_DIM else "skipped (no rot6d dims)"
        print(f"  {key}: dim={dim} -> {status}")
    print(f"wrote: {stats_file}")


if __name__ == "__main__":
    tyro.cli(main)

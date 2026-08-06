"""Neutralize the rot6d action dims of a config's norm-stats FILE.

Rewrites ``assets/<config>/<repo_id>/norm_stats.json`` so the 12 rot6d action
dimensions carry neutral parameters (mean 0 / std 1 / q01 -1 / q99 +1): the
stock openpi quantile normalizer then passes rot6d through unchanged while
translation and gripper dims keep their data-driven statistics. A one-time
backup of the original file is written alongside as
``norm_stats.json.pre_rot6d_neutral.bak`` (never overwritten).

This edits the stats file rather than the load path on purpose: checkpoints
embed their stats, so runs trained before/after the neutralization each keep
a self-consistent serving pipeline under the same config name.
"""

import pathlib
import shutil

import tyro

import openpi.policies.umi_dual_franka_policy as umi_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config


def main(config_name: str):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    stats_dir = pathlib.Path(config.assets_dirs) / data_config.repo_id
    stats_file = stats_dir / "norm_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"{stats_file} does not exist; compute norm stats first.")

    backup = stats_dir / "norm_stats.json.pre_rot6d_neutral.bak"
    if not backup.exists():
        shutil.copy2(stats_file, backup)
        print(f"backup written: {backup}")

    stats = normalize.load(stats_dir)
    patched = umi_policy.neutralize_rot6d_action_norm_stats(stats)
    normalize.save(stats_dir, patched)
    print(f"neutralized rot6d action dims in: {stats_file}")


if __name__ == "__main__":
    tyro.cli(main)

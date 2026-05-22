"""Push an openpi checkpoint directory to the HuggingFace Hub.

Wraps `huggingface_hub.upload_folder`. The checkpoint dir should contain
`params/`, `assets/`, `train_state/`, and a `_CHECKPOINT_METADATA` file —
i.e. what openpi's train.py writes to `<checkpoint_base_dir>/<config>/<exp>/<step>/`.

Once uploaded, serve from anywhere with:

    uv run python scripts/serve_policy.py --port=8111 policy:checkpoint \\
      --policy.config=<train_config_name> --policy.dir=<repo_id>

Usage:
    uv run python scripts/push_to_hub.py \\
        --checkpoint=/mnt/localssd/<user>/openpi-checkpoints/pi05_yam_vial_30fps/v1/4999 \\
        --repo=ttotmoon/yam-vial-place-pi05-v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import create_repo, upload_folder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Absolute path to the openpi checkpoint dir (the one with params/, assets/, _CHECKPOINT_METADATA).")
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. 'ttotmoon/my-pi05-v1'.")
    parser.add_argument("--private", action="store_true", help="Create as a private repo.")
    parser.add_argument("--commit-message", default=None, help="Override commit message.")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not (ckpt / "_CHECKPOINT_METADATA").exists():
        print(f"ERROR: {ckpt} does not look like an openpi checkpoint (no _CHECKPOINT_METADATA file).", file=sys.stderr)
        return 1

    print(f"Pushing {ckpt} → {args.repo}")
    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.commit_message or f"Upload openpi checkpoint from {ckpt.name}",
    )
    print(f"pushed: {args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

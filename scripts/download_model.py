#!/usr/bin/env python3
"""
Download HunyuanVideo model to local path for offline use.

Usage:
    python scripts/download_model.py --output_dir /path/to/models/HunyuanVideo
"""

import argparse
import os


def download_hunyuanvideo(output_dir: str, revision: str = "refs/pr/18"):
    """Download HunyuanVideo model to local directory."""
    from huggingface_hub import snapshot_download

    print(f"Downloading HunyuanVideo to: {output_dir}")
    print(f"Revision: {revision}")
    print("This may take a while (~25GB)...")

    # Download the full model
    snapshot_download(
        repo_id="tencent/HunyuanVideo",
        revision=revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,  # Copy files instead of symlinks
        resume_download=True,  # Resume if interrupted
    )

    print(f"\nModel downloaded successfully to: {output_dir}")
    print(f"\nUsage:")
    print(f"  python hyvideo_t2v_inference.py --model_id {output_dir} ...")
    print(f"  python experiments/run_experiments.py --model_id {output_dir} --suite quick_validation")


def main():
    parser = argparse.ArgumentParser(description="Download HunyuanVideo model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/HunyuanVideo",
        help="Local directory to save the model"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="refs/pr/18",
        help="Model revision to download"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    download_hunyuanvideo(args.output_dir, args.revision)


if __name__ == "__main__":
    main()

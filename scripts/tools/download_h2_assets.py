# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility script to download official Unitree H2 USD assets into the local Isaac Lab asset cache."""

import argparse
import os
import shutil
import subprocess
import urllib.request

REPO_ID = "unitreerobotics/unitree_model"
HF_BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/H2"

# Files composing the H2 USD asset
H2_FILES = [
    "H2_dae.usd",
    "configuration/H2_dae_base.usd",
    "configuration/H2_dae_physics.usd",
    "configuration/H2_dae_robot.usd",
    "configuration/H2_dae_sensor.usd",
]


def get_default_target_dir() -> str:
    """Get default asset target directory inside isaaclab_assets/data/Robots/Unitree/H2."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "../../"))
    return os.path.join(repo_root, "source/isaaclab_assets/data/Robots/Unitree/H2")


def download_via_huggingface_hub(target_dir: str) -> bool:
    """Attempt download using huggingface_hub Python package."""
    try:
        from huggingface_hub import hf_hub_download

        print("[INFO] Attempting download using huggingface_hub...")
        for rel_path in H2_FILES:
            dest = os.path.join(target_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            hf_path = f"H2/{rel_path}"
            print(f"[INFO] Downloading {hf_path} -> {dest}")
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=hf_path,
                repo_type="dataset",
                local_dir_use_symlinks=False,
            )
            shutil.copy2(downloaded, dest)
        return True
    except Exception as e:
        print(f"[WARNING] huggingface_hub download failed: {e}")
        return False


def download_via_git_sparse_checkout(target_dir: str) -> bool:
    """Attempt download using git sparse-checkout."""
    try:
        print("[INFO] Attempting download using git clone (sparse-checkout)...")
        temp_dir = os.path.join(target_dir, "_temp_git_clone")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        clone_cmd = [
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--sparse",
            f"https://huggingface.co/datasets/{REPO_ID}",
            temp_dir,
        ]
        subprocess.run(clone_cmd, check=True)
        subprocess.run(["git", "-C", temp_dir, "sparse-checkout", "set", "H2"], check=True)
        subprocess.run(["git", "-C", temp_dir, "lfs", "pull"], check=False)

        src_h2 = os.path.join(temp_dir, "H2")
        if os.path.exists(src_h2):
            for item in os.listdir(src_h2):
                s = os.path.join(src_h2, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            shutil.rmtree(temp_dir)
            return True
    except Exception as e:
        print(f"[WARNING] Git sparse-checkout failed: {e}")
        temp_dir = os.path.join(target_dir, "_temp_git_clone")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    return False


def download_via_urllib(target_dir: str) -> bool:
    """Fallback direct HTTP download using urllib."""
    print("[INFO] Attempting direct download via urllib...")
    try:
        for rel_path in H2_FILES:
            dest = os.path.join(target_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            url = f"{HF_BASE_URL}/{rel_path}"
            print(f"[INFO] Downloading {url} -> {dest} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "IsaacLab-H2-Downloader"})
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as out_file:
                shutil.copyfileobj(resp, out_file)
        return True
    except Exception as e:
        print(f"[ERROR] Direct urllib download failed: {e}")
        return False


def ensure_h2_assets(target_dir: str | None = None) -> str:
    """Ensure H2 USD assets are present in the target directory, downloading if missing."""
    if target_dir is None:
        target_dir = get_default_target_dir()

    main_usd = os.path.join(target_dir, "H2_dae.usd")
    base_usd = os.path.join(target_dir, "configuration/H2_dae_base.usd")

    if os.path.exists(main_usd) and os.path.exists(base_usd) and os.path.getsize(base_usd) > 1000000:
        print(f"[INFO] H2 assets already present at: {target_dir}")
        return main_usd

    os.makedirs(target_dir, exist_ok=True)
    print(f"[INFO] Downloading Unitree H2 assets into: {target_dir}")

    success = (
        download_via_huggingface_hub(target_dir)
        or download_via_git_sparse_checkout(target_dir)
        or download_via_urllib(target_dir)
    )

    if not success or not os.path.exists(main_usd):
        raise RuntimeError(
            f"Failed to download Unitree H2 assets into '{target_dir}'. "
            f"Please download manually from https://huggingface.co/datasets/{REPO_ID}."
        )

    print(f"[INFO] Successfully downloaded H2 assets to: {target_dir}")
    return main_usd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Unitree H2 USD assets.")
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Target directory for downloaded assets (default: source/isaaclab_assets/data/Robots/Unitree/H2).",
    )
    args = parser.parse_args()
    ensure_h2_assets(args.target_dir)

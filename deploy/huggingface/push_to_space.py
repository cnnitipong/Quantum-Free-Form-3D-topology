"""Upload this repository to a Hugging Face Docker Space.

Used by .github/workflows/deploy-hf-space.yml and for a manual deployment:

    pip install huggingface_hub
    export HF_TOKEN=hf_...            # a token with "write" access
    python deploy/huggingface/push_to_space.py --space cnnitipong/qff3d

What is uploaded: the files git tracks at HEAD (``git ls-files``), minus
what the container does not need (tests, screenshots, CI files), with
README.md replaced by deploy/huggingface/SPACE_README.md (the Space's YAML
front matter: sdk docker, app_port 7860).  The upload goes through the Hub API
(``HfApi.upload_folder``), which stores binary files (STL, PNG) the way the Hub
requires; a plain ``git push`` of binary files to a Space is rejected.
Files on the Space that are no longer in the repository are deleted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPACE_README = Path(__file__).resolve().parent / "SPACE_README.md"

#: path prefixes that are not uploaded to the Space
EXCLUDE = (".github/", "tests/", "webapp/screenshots/", "deploy/vercel/",
           "Start QFF-3D (Windows).bat", "Start QFF-3D (macOS).command", "run_app.sh")


def tracked_files() -> list:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=True,
                         capture_output=True).stdout.decode("utf-8")
    return [f for f in out.split("\0") if f and not f.startswith(EXCLUDE)]


def stage(dest: Path) -> int:
    files = tracked_files()
    for rel in files:
        src = REPO / rel
        if not src.is_file():
            continue
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)
    shutil.copy2(SPACE_README, dest / "README.md")
    return len(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--space", default=os.environ.get("HF_SPACE") or "cnnitipong/qff3d",
                    help="Space id <user>/<name> (default: $HF_SPACE or cnnitipong/qff3d)")
    ap.add_argument("--message", default=None, help="commit message on the Space")
    ap.add_argument("--dry-run", action="store_true", help="stage the files and list them only")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="qff3d_space_") as td:
        dest = Path(td)
        n = stage(dest)
        print(f"staged {n} files for {args.space}")
        if args.dry_run:
            for p in sorted(dest.rglob("*")):
                if p.is_file():
                    print("  ", p.relative_to(dest).as_posix())
            return 0
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("HF_TOKEN is not set: nothing uploaded.", file=sys.stderr)
            return 2
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        sha = os.environ.get("GITHUB_SHA") or subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        info = api.upload_folder(
            repo_id=args.space, repo_type="space", folder_path=str(dest),
            commit_message=args.message or f"Sync from GitHub ({sha[:12]})",
            delete_patterns=["*"],  # drop files that left the repository
        )
        print(f"uploaded: {info}")
        print(f"Space: https://huggingface.co/spaces/{args.space}  "
              f"(app: https://{args.space.replace('/', '-').replace('_', '-').lower()}.hf.space/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

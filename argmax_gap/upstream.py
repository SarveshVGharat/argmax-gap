"""Immutable upstream sources used for the paper's two frozen policies."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

REPOSITORIES = {
    "maia3": ("https://github.com/CSSLab/maia3.git", "1e13597c42d4858b7cfd7cfdae01e297263364b2"),
    "allie": ("https://github.com/ippolito-cmu/allie.git", "a50f2d86618798cec2195e37e3484da579631328"),
}
# repo_id, repo_type, revision, filename, SHA-256
ASSETS = {
    "maia3-79m.pt": ("UofTCSSLab/Maia3-79M", "model", "a107d6ceb7b298cb04ae1da4edffe2939858b894", "maia3-79m.pt", "3fc6181d5db789b45a15305732148757ae74efa3e0028e81ba335b462dac45c2"),
    "allie-medium.pt": ("yimingzhang/allie-models", "dataset", "29ae29c84a1587e2a64257406cc194f1bd329e89", "medium/best.pt", "e64e8862ca630dd1b8cc0b3ff9d6dfa78014ea2c93247192b8f6e835ecf4f523"),
    "allie-test.jsonl": ("yimingzhang/allie-data", "dataset", "990f79105b1f08a8fd367fcb10c5e18c69436e30", "lichess-2022-blitz-test/2022-test-annotated.jsonl", "a6014de8ef861b2ee23a84d7ce824e644390577a000350fad06bd995bcddab0d"),
}
HELDOUT_SHA256 = "9a00544f74150ef8a5da156464becbccf99ea85a2af4b8f89fef1dc30fef1978"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def verify_repository(path: Path, name: str) -> None:
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != REPOSITORIES[name][1]:
        raise ValueError(f"{name}: expected commit {REPOSITORIES[name][1]}, got {revision}")
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise ValueError(f"{name}: tracked upstream files have local modifications")


def fetch_assets(destination: Path, heldout_file: Path | None = None) -> None:
    """Fetch frozen source and weights; the exact development split is a release asset."""
    from huggingface_hub import hf_hub_download

    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in REPOSITORIES.items():
        target = destination / "upstream" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", url, str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
        verify_repository(target, name)
    for name, (repo_id, repo_type, revision, filename, digest) in ASSETS.items():
        target = destination / name
        if not target.exists():
            cached = hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision)
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(cached, temporary)
            verify_hash(temporary, digest)
            temporary.replace(target)
        verify_hash(target, digest)
    if heldout_file is not None:
        verify_hash(heldout_file, HELDOUT_SHA256)
        target = destination / "heldout.parquet"
        if heldout_file.resolve() != target.resolve():
            if target.exists():
                verify_hash(target, HELDOUT_SHA256)
            else:
                shutil.copyfile(heldout_file, target)

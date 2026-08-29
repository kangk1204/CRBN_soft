#!/usr/bin/env python3
"""Release preflight checks that do not read credential contents."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

EXACT_CREDENTIAL_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".netrc",
    "credentials",
    "credentials.json",
    "github.txt",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
    "passwd",
    "password.txt",
    "pwd.txt",
    "secret.txt",
    "secrets.json",
    "token.txt",
}

CREDENTIAL_SUFFIXES = {
    ".key",
    ".pat",
    ".p12",
    ".pem",
    ".pfx",
}

CREDENTIAL_NAME_PARTS = (
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "client_secret",
    "credential",
    "github_pat",
    "password",
    "private_key",
    "secret",
)

SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}


def is_credential_like_path(path: Path) -> bool:
    """Classify credential-like files by path metadata only."""
    name = path.name.lower()
    stem = path.stem.lower()
    return (
        name in EXACT_CREDENTIAL_NAMES
        or path.suffix.lower() in CREDENTIAL_SUFFIXES
        or any(part in name or part in stem for part in CREDENTIAL_NAME_PARTS)
    )


def iter_files(root: Path):
    for child in sorted(root.iterdir(), key=lambda p: str(p)):
        if child.is_symlink():
            # Inspect the link's own path metadata without following its target.
            yield child
            continue
        if child.is_dir():
            if child.name in SKIP_DIR_NAMES:
                continue
            yield from iter_files(child)
        elif child.is_file():
            yield child


def credential_like_files(repo_root: Path, include_parent: bool = True) -> list[Path]:
    repo_root = repo_root.resolve()
    scope_root = repo_root.parent if include_parent else repo_root
    matches = []
    for path in iter_files(scope_root):
        if is_credential_like_path(path):
            matches.append(path)
    return matches


def format_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def run(repo_root: Path, include_parent: bool = True) -> int:
    scope_root = repo_root.parent if include_parent else repo_root
    matches = credential_like_files(repo_root, include_parent=include_parent)
    if matches:
        print("Release audit failed: credential-like files are present.", file=sys.stderr)
        print("The audit reports paths only and does not read file contents.", file=sys.stderr)
        for path in matches:
            print(f"- {format_path(path, scope_root)}", file=sys.stderr)
        return 1
    print("Release audit passed: no credential-like files found in release scope.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail release if credential-like files exist in the repo or parent release scope."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to audit. Defaults to this script's repository.",
    )
    parser.add_argument(
        "--repo-only",
        action="store_true",
        help="Audit only the repository root, not its parent release scope.",
    )
    args = parser.parse_args(argv)
    return run(args.repo_root, include_parent=not args.repo_only)


if __name__ == "__main__":
    raise SystemExit(main())

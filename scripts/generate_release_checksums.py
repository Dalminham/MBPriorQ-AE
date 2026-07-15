#!/usr/bin/env python3
"""Generate SHA-256 checksums for files included in the public archive."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "metadata" / "release_files.sha256"
SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "checkpoints",
    "datasets",
    "local_runs",
    "models",
    "outputs",
    "simWorkspace",
    "target",
    "tmp",
}
SKIP_NAMES = {
    "ae.aux",
    "ae.fdb_latexmk",
    "ae.fls",
    "ae.log",
    "ae.out",
    "ae.pdf",
    "release_files.sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or any(part in SKIP_PARTS for part in path.parts):
            continue
        rows.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} checksums to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

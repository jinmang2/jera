"""Corpus fetch helper — reads manifest.json, verifies sha256 of locally-present PDFs.

Does NOT auto-download in bulk.  If a PDF is missing, prints download instructions
so the user can obtain it manually from the institution's website.

Usage::

    uv run python scripts/fetch_corpus.py [--manifest data/corpus/manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    entries: list[dict[str, str]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_dir = manifest_path.parent
    ok = 0
    missing = 0
    bad_hash = 0

    for entry in entries:
        filename: str = entry.get("filename", "")
        if not filename:
            print(f"  [SKIP] entry has no filename: {entry.get('title', '?')!r}")
            continue

        pdf_path = corpus_dir / filename
        expected_sha256: str = entry.get("sha256", "")

        if not pdf_path.exists():
            missing += 1
            print(f"  [MISSING] {filename}")
            print(f"    Institution : {entry.get('inst', '?')}")
            print(f"    Title       : {entry.get('title', '?')}")
            print(f"    URL         : {entry.get('url', '?')}")
            print(f"    License     : {entry.get('license', '?')}")
            print(f"    Download manually and save to: {pdf_path}")
            print()
        elif expected_sha256 and not expected_sha256.startswith("<"):
            actual = _sha256_of(pdf_path)
            if actual != expected_sha256:
                bad_hash += 1
                print(f"  [HASH MISMATCH] {filename}")
                print(f"    expected : {expected_sha256}")
                print(f"    actual   : {actual}")
            else:
                ok += 1
                print(f"  [OK] {filename}")
        else:
            # sha256 not yet recorded in manifest — just note presence
            ok += 1
            actual = _sha256_of(pdf_path)
            print(f"  [PRESENT, sha256 not recorded] {filename}")
            print(f"    actual sha256: {actual}")
            print(f'    Add to manifest: "sha256": "{actual}"')

    print(f"\nSummary: {ok} ok, {missing} missing, {bad_hash} hash mismatch(es).")
    return 0 if (missing == 0 and bad_hash == 0) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify corpus PDFs against manifest sha256.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/corpus/manifest.json"),
        help="Path to manifest.json (default: data/corpus/manifest.json)",
    )
    args = parser.parse_args()
    sys.exit(main(args.manifest))

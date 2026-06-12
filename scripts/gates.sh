#!/usr/bin/env bash
# Jera M1 acceptance gates. Run from the repo root: `bash scripts/gates.sh`.
set -euo pipefail

echo "==> 1/4 ruff check"
uv run ruff check .

echo "==> 2/4 ruff format --check"
uv run ruff format --check .

echo "==> 3/4 mypy (strict)"
uv run mypy -p jera -p app

echo "==> 4/4 pytest (unit + integration + gates + e2e)"
uv run pytest

echo "All gates passed."

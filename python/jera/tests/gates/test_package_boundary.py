"""Package-boundary gate (Gate 1): no domain ownership under app.*; src-layout install proof."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

_FORBIDDEN = re.compile(
    r"^\s*(from|import)\s+app\.(ingestion|retrieval|evaluation)\b", re.MULTILINE
)


def _foreign_interpreter() -> str | None:
    """A python interpreter that is NOT this workspace venv (so it lacks the editable install)."""
    for name in ("/usr/bin/python3", "/usr/local/bin/python3", shutil.which("python3") or ""):
        if name and Path(name).exists() and Path(name).resolve() != Path(sys.executable).resolve():
            return name
    return None


def test_no_domain_imports_from_app_namespace() -> None:
    # No reusable domain (ingestion/retrieval/evaluation) is imported from the API package.
    # Pure-Python scan so the gate does not depend on ripgrep being on PATH.
    offenders: list[str] = []
    for root in ("apps", "python/jera/src"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if _FORBIDDEN.match(line):
                    offenders.append(f"{path}:{i}: {line.strip()}")
    assert not offenders, "found forbidden app.* domain imports:\n" + "\n".join(offenders)


def test_jera_rag_imports_from_installed_package() -> None:
    import jera.rag  # noqa: F401  # importable because the workspace package is installed

    assert hasattr(jera.rag, "build_system")


def test_jera_not_importable_without_install_from_foreign_cwd() -> None:
    # Negative cwd-trap proof: an interpreter WITHOUT the editable install, run from /tmp,
    # must NOT find `jera` (src-layout means it cannot be imported from the working directory).
    interpreter = _foreign_interpreter()
    if interpreter is None:
        pytest.skip("no non-venv interpreter available to prove the cwd-import trap")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [interpreter, "-c", "import jera"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode == 0:
        pytest.skip(
            "foreign interpreter already has `jera` installed globally; trap not testable here"
        )
    assert "ModuleNotFoundError" in result.stderr

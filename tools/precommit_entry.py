"""Launch repository checks with the project development environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    candidates = (
        ROOT / "ml/.venv/Scripts/python.exe",
        ROOT / "ml/.venv/bin/python",
    )
    python = next((candidate for candidate in candidates if candidate.is_file()), None)
    if python is None:
        python = Path(sys.executable)
    return subprocess.call([str(python), str(ROOT / "tools/precommit.py")], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

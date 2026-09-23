"""Generate PlatformIO's compilation database for VS Code IntelliSense."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware"


def platformio_command() -> list[str]:
    """Find PlatformIO in the active environment or common local installs."""

    if importlib.util.find_spec("platformio") is not None:
        return [sys.executable, "-m", "platformio"]

    executable = shutil.which("pio") or shutil.which("platformio")
    if executable:
        return [executable]

    venv_python = (
        ROOT / "ml" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if venv_python.is_file():
        return [str(venv_python), "-m", "platformio"]

    platformio_home = Path.home() / ".platformio" / "penv"
    names = ("Scripts/pio.exe", "Scripts/platformio.exe") if os.name == "nt" else ("bin/pio",)
    for name in names:
        candidate = platformio_home / name
        if candidate.is_file():
            return [str(candidate)]

    raise SystemExit(
        "PlatformIO was not found. Install the recommended VS Code extension "
        "or run: python -m pip install -r requirements-dev.txt"
    )


def main() -> None:
    environment = os.environ.copy()
    environment.setdefault("WIFI_SSID", "intellisense-placeholder")
    environment.setdefault("WIFI_PASSWORD", "intellisense-placeholder")
    command = [
        *platformio_command(),
        "run",
        "--project-dir",
        str(FIRMWARE),
        "--target",
        "compiledb",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)

    database = FIRMWARE / "compile_commands.json"
    if not database.is_file():
        raise SystemExit(f"PlatformIO did not create {database}")
    print(f"\nVS Code compilation database refreshed: {database}")


if __name__ == "__main__":
    main()

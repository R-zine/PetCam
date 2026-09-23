"""Run the repository checks used by the pre-commit hook."""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRECTORIES = {".git", ".pio", ".pytest_cache", ".venv", "__pycache__"}
IGNORED_FILES = {"secrets.generated.h"}
TEXT_SUFFIXES = {
    ".cpp",
    ".example",
    ".h",
    ".ini",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {".gitattributes", ".gitignore", "README"}


def run(label: str, command: Sequence[str], *, environment: dict[str, str] | None = None) -> None:
    """Run one check and stop immediately with its exit status on failure."""

    print(f"\n==> {label}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def executable(name: str) -> str:
    """Require a development tool with an actionable installation message."""

    sibling_name = f"{name}.exe" if os.name == "nt" else name
    sibling = Path(sys.executable).with_name(sibling_name)
    path = str(sibling) if sibling.is_file() else shutil.which(name)
    if path is None:
        raise SystemExit(
            f"Required tool {name!r} is unavailable. "
            "Install the hook dependencies with: "
            f"{sys.executable} -m pip install -r requirements-dev.txt"
        )
    return path


def repository_text_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.name in IGNORED_FILES
            or any(part in IGNORED_DIRECTORIES for part in path.parts)
        ):
            continue
        if path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


def validate_text_files() -> None:
    """Check inexpensive repository-wide text and configuration invariants."""

    problems: list[str] = []
    retired_name = "rabbit" + "cam"
    for path in repository_text_files():
        relative = path.relative_to(ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"{relative}: not valid UTF-8 ({exc})")
            continue
        if text and not text.endswith("\n"):
            problems.append(f"{relative}: missing final newline")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                problems.append(f"{relative}:{number}: trailing whitespace")
        if retired_name in text.lower():
            problems.append(f"{relative}: contains the retired project name")

    try:
        yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
        yaml.safe_load((ROOT / "ml/configs/default.yaml").read_text(encoding="utf-8"))
        tomllib.loads((ROOT / "ml/pyproject.toml").read_text(encoding="utf-8"))
        parser = configparser.ConfigParser()
        with (ROOT / "firmware/platformio.ini").open(encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, ValueError, configparser.Error, yaml.YAMLError) as exc:
        problems.append(f"configuration parsing failed: {exc}")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    print("==> Repository text and configuration")
    validate_text_files()

    python = sys.executable
    run(
        "Ruff lint",
        [
            python,
            "-m",
            "ruff",
            "check",
            "--config",
            "ml/pyproject.toml",
            "ml/src",
            "ml/tests",
            "tools",
        ],
    )
    run(
        "Ruff format",
        [
            python,
            "-m",
            "ruff",
            "format",
            "--check",
            "--config",
            "ml/pyproject.toml",
            "ml/src",
            "ml/tests",
            "tools",
        ],
    )
    cpp_files = sorted(
        str(path.relative_to(ROOT))
        for directory in (ROOT / "firmware/include", ROOT / "firmware/src")
        for path in directory.rglob("*")
        if path.suffix in {".cpp", ".h"} and path.name not in IGNORED_FILES
    )
    run(
        "C++ format",
        [executable("clang-format"), "--dry-run", "--Werror", *cpp_files],
    )
    run("ML tests", [python, "-m", "pytest", "-q", "ml/tests"])

    firmware_environment = os.environ.copy()
    firmware_environment["WIFI_SSID"] = "precommit-placeholder"
    firmware_environment["WIFI_PASSWORD"] = "precommit-placeholder"
    run(
        "Firmware build",
        [python, "-m", "platformio", "run", "--project-dir", "firmware"],
        environment=firmware_environment,
    )

    print("\nAll pre-commit checks passed.")


if __name__ == "__main__":
    main()

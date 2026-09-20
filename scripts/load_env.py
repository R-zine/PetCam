from pathlib import Path

Import("env")

project_dir = Path(env["PROJECT_DIR"])
env_file = project_dir / ".env"
output_file = project_dir / "include" / "secrets.generated.h"

if not env_file.exists():
    raise RuntimeError(
        "Missing .env file. Create one containing WIFI_SSID and WIFI_PASSWORD."
    )


def parse_env(path):
    values = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]

        values[key] = value

    return values


def cpp_string(value):
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    return f'"{value}"'


values = parse_env(env_file)

required = ["WIFI_SSID", "WIFI_PASSWORD"]

for key in required:
    if key not in values:
        raise RuntimeError(f"Missing {key} in .env")

output_file.parent.mkdir(parents=True, exist_ok=True)

output_file.write_text(
    "#pragma once\n\n"
    f"#define WIFI_SSID {cpp_string(values['WIFI_SSID'])}\n"
    f"#define WIFI_PASSWORD {cpp_string(values['WIFI_PASSWORD'])}\n",
    encoding="utf-8",
)

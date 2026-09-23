# PetCam

PetCam is a self-hosted ESP32-S3 camera for monitoring a pet enclosure. The
firmware provides a browser view, MJPEG stream, and battery status. An optional
local Python package collects labeled frames and trains a three-class posture
model: `standing`, `lateral`, or `unknown`.

## Repository

- [`firmware/`](firmware/) - PlatformIO firmware for the ESP32-S3 camera.
- [`ml/`](ml/) - PyTorch training, evaluation, capture, and inference tools.

The ML model classifies individual frames only. Persistent-state decisions,
health interpretation, and alerts belong in a separate service.

## Firmware quick start

Requirements: PlatformIO and the supported ESP32-S3 camera hardware.

```powershell
cd firmware
Copy-Item .env.example .env
```

Set `WIFI_SSID` and `WIFI_PASSWORD` in `.env`, then build and upload:

```powershell
pio run
pio run --target upload
pio device monitor
```

The default upload and monitor port is `COM4`; change it in
[`firmware/platformio.ini`](firmware/platformio.ini) when needed. After the
device connects, the serial monitor prints its IP address.

- Web interface: `http://DEVICE_IP/`
- Status API: `http://DEVICE_IP/api/status`
- MJPEG stream: `http://DEVICE_IP:81/stream`

See [`firmware/README.md`](firmware/README.md) for firmware-specific notes.

## ML quick start

Python 3.11 or newer is required.

```powershell
cd ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
petcam-ml --help
```

Typical workflow:

```powershell
petcam-ml capture --stream-url http://DEVICE_IP:81/stream `
  --label standing --lighting visible --session standing-visible-001 `
  --interval 1.0 --duration 60

petcam-ml prepare --config configs/default.yaml
petcam-ml train --config configs/default.yaml
```

Collect multiple independent sessions for every class under both visible and
infrared lighting. Session-aware splitting prevents adjacent frames from
leaking between training and evaluation sets.

See [`ml/README.md`](ml/README.md) for the complete capture, training, resume,
evaluation, and inference workflow.

## Generated files

Wi-Fi credentials, generated firmware secrets, captured images, checkpoints,
and training artifacts are excluded from Git. Do not commit them.

## Development checks

Install and enable the repository hook from the project root:

```powershell
python -m pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

Every commit checks repository text/configuration, runs Ruff and clang-format,
runs the ML test suite, and compiles the firmware with placeholder credentials.

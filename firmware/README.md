# PetCam firmware

This directory is the self-contained PlatformIO project for the ESP32-S3
camera. The Python posture-classification subsystem lives separately in
`../ml/`.

Create the local firmware environment file before building:

```powershell
Copy-Item .env.example .env
```

Set `WIFI_SSID` and `WIFI_PASSWORD`, then run PlatformIO from this directory:

```powershell
pio run
pio run --target upload
pio device monitor
```

The pre-build script generates `include/secrets.generated.h`; both `.env` and
the generated header are ignored by Git.

## VS Code C++ type checking

Install the recommended C/C++ and PlatformIO extensions, then run the
`PetCam: Refresh C++ IntelliSense` task once. Run it again after changing
PlatformIO dependencies or build flags. The task generates the local
`compile_commands.json` that VS Code uses for the ESP32 compiler flags,
defines, and framework headers.

If diagnostics remain stale after refreshing, run `C/C++: Reset IntelliSense
Database` from the VS Code command palette.

## Source layout

- `main.cpp` coordinates startup and the main loop.
- `battery_monitor.cpp` owns the MAX17048 state and polling interval.
- `camera_controller.cpp` owns camera pins and initialization.
- `network.cpp` owns Wi-Fi connection details.
- `web_server.cpp` owns the UI, status API, and MJPEG routes.
- `web_page.h` contains the embedded browser page.

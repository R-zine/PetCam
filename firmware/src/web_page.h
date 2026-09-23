#pragma once

namespace petcam {
namespace web {
namespace detail {

static const char PAGE[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PetCam</title>
    <style>
        body {
            background: #111;
            color: #eee;
            font-family: sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        h1 { margin-top: 0; }
        #status { margin-bottom: 16px; line-height: 1.8; }
        #battery { font-size: 1.2rem; font-weight: bold; }
        #video { max-width: 100%; height: auto; border-radius: 8px; }
        .secondary { opacity: 0.75; font-size: 0.9rem; }
    </style>
</head>
<body>
    <h1>PetCam</h1>
    <div id="status">
        <div id="battery">Battery: loading...</div>
        <div id="system" class="secondary">Loading...</div>
    </div>
    <img id="video" alt="PetCam live stream">
    <script>
        const video = document.getElementById("video");
        const batteryElement = document.getElementById("battery");
        const systemElement = document.getElementById("system");

        video.src = "http://" + window.location.hostname + ":81/stream";

        async function updateStatus() {
            try {
                const response = await fetch("/api/status");
                const status = await response.json();

                if (status.battery_percent !== null && status.battery_voltage !== null) {
                    batteryElement.textContent =
                        "Battery: " + status.battery_percent.toFixed(1) + "% (" +
                        status.battery_voltage.toFixed(3) + " V)";
                } else {
                    batteryElement.textContent = "Battery: unavailable";
                }

                systemElement.textContent =
                    "Wi-Fi: " + status.rssi + " dBm | Uptime: " + status.uptime_seconds +
                    " s | PSRAM: " + (status.psram ? "yes" : "no");
            } catch (error) {
                batteryElement.textContent = "Battery: unavailable";
                systemElement.textContent = "Status unavailable";
            }
        }

        updateStatus();
        setInterval(updateStatus, 3000);
    </script>
</body>
</html>
)rawliteral";

} // namespace detail
} // namespace web
} // namespace petcam

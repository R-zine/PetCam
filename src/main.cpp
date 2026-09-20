#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"

#include "secrets.generated.h"

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

// GOOUUU ESP32-S3-CAM / ESP32S3_EYE compatible mapping

#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM  4
#define SIOC_GPIO_NUM  5

#define Y2_GPIO_NUM    11
#define Y3_GPIO_NUM    9
#define Y4_GPIO_NUM    8
#define Y5_GPIO_NUM    10
#define Y6_GPIO_NUM    12
#define Y7_GPIO_NUM    18
#define Y8_GPIO_NUM    17
#define Y9_GPIO_NUM    16

#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM  7
#define PCLK_GPIO_NUM  13


static httpd_handle_t web_server = nullptr;
static httpd_handle_t stream_server = nullptr;


static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=frame";

static const char* STREAM_BOUNDARY =
    "\r\n--frame\r\n";

static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n"
    "\r\n";


static esp_err_t index_handler(httpd_req_t* req)
{
    static const char html[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RabbitCam</title>

    <style>
        body {
            background: #111;
            color: #eee;
            font-family: sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }

        #video {
            max-width: 100%;
            height: auto;
        }

        #status {
            margin-bottom: 10px;
        }
    </style>
</head>

<body>

<h1>RabbitCam</h1>

<div id="status">
    Loading...
</div>

<img id="video">

<script>
    const video = document.getElementById("video");

    video.src =
        "http://" +
        window.location.hostname +
        ":81/stream";

    async function updateStatus() {
        try {
            const response = await fetch("/api/status");
            const status = await response.json();

            document.getElementById("status").textContent =
                "Wi-Fi RSSI: " + status.rssi + " dBm | " +
                "Uptime: " + status.uptime_seconds + " s";
        }
        catch (e) {
            document.getElementById("status").textContent =
                "Status unavailable";
        }
    }

    updateStatus();
    setInterval(updateStatus, 3000);
</script>

</body>
</html>
)rawliteral";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}


static esp_err_t status_handler(httpd_req_t* req)
{
    char json[256];

    snprintf(
        json,
        sizeof(json),
        "{"
        "\"rssi\":%d,"
        "\"uptime_seconds\":%lu,"
        "\"psram\":%s"
        "}",
        WiFi.RSSI(),
        millis() / 1000UL,
        psramFound() ? "true" : "false"
    );

    httpd_resp_set_type(req, "application/json");
    return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}


static esp_err_t stream_handler(httpd_req_t* req)
{
    esp_err_t result =
        httpd_resp_set_type(req, STREAM_CONTENT_TYPE);

    if (result != ESP_OK) {
        return result;
    }

    while (true) {

        camera_fb_t* fb = esp_camera_fb_get();

        if (!fb) {
            Serial.println("Camera capture failed");
            return ESP_FAIL;
        }

        char header[64];

        size_t header_length = snprintf(
            header,
            sizeof(header),
            STREAM_PART,
            fb->len
        );

        result = httpd_resp_send_chunk(
            req,
            STREAM_BOUNDARY,
            strlen(STREAM_BOUNDARY)
        );

        if (result == ESP_OK) {
            result = httpd_resp_send_chunk(
                req,
                header,
                header_length
            );
        }

        if (result == ESP_OK) {
            result = httpd_resp_send_chunk(
                req,
                reinterpret_cast<const char*>(fb->buf),
                fb->len
            );
        }

        esp_camera_fb_return(fb);

        if (result != ESP_OK) {
            break;
        }
    }

    return result;
}


void startServers()
{
    // Main HTTP/API server

    httpd_config_t web_config = HTTPD_DEFAULT_CONFIG();
    web_config.server_port = 80;

    if (httpd_start(&web_server, &web_config) == ESP_OK) {

        httpd_uri_t index_uri = {};
        index_uri.uri = "/";
        index_uri.method = HTTP_GET;
        index_uri.handler = index_handler;

        httpd_register_uri_handler(
            web_server,
            &index_uri
        );

        httpd_uri_t status_uri = {};
        status_uri.uri = "/api/status";
        status_uri.method = HTTP_GET;
        status_uri.handler = status_handler;

        httpd_register_uri_handler(
            web_server,
            &status_uri
        );
    }


    // MJPEG server

    httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();

    stream_config.server_port = 81;

    // Must differ from the first server
    stream_config.ctrl_port =
        web_config.ctrl_port + 1;

    if (httpd_start(
            &stream_server,
            &stream_config
        ) == ESP_OK) {

        httpd_uri_t stream_uri = {};
        stream_uri.uri = "/stream";
        stream_uri.method = HTTP_GET;
        stream_uri.handler = stream_handler;

        httpd_register_uri_handler(
            stream_server,
            &stream_uri
        );
    }
}


void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("RabbitCam booting...");

    Serial.printf(
        "PSRAM found: %s\n",
        psramFound() ? "yes" : "no"
    );


    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;

    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;

    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;

    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;

    config.xclk_freq_hz = 20000000;

    config.pixel_format = PIXFORMAT_JPEG;

    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;

    config.fb_count =
        psramFound() ? 2 : 1;

    config.fb_location =
        psramFound()
            ? CAMERA_FB_IN_PSRAM
            : CAMERA_FB_IN_DRAM;


    esp_err_t err =
        esp_camera_init(&config);

    if (err != ESP_OK) {
        Serial.printf(
            "Camera init failed: 0x%x\n",
            err
        );
        return;
    }

    Serial.println(
        "Camera initialized successfully!"
    );


    WiFi.begin(ssid, password);

    Serial.print("Connecting to Wi-Fi");

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("Wi-Fi connected.");

    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());


    startServers();

    Serial.println();
    Serial.println("RabbitCam is running.");

    Serial.printf(
        "UI:     http://%s/\n",
        WiFi.localIP().toString().c_str()
    );

    Serial.printf(
        "Stream: http://%s:81/stream\n",
        WiFi.localIP().toString().c_str()
    );
}


void loop()
{
    delay(1000);
}
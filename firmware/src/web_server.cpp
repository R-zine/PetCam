#include "web_server.h"

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "battery_monitor.h"
#include "esp_camera.h"
#include "esp_http_server.h"
#include "network.h"
#include "web_page.h"

namespace {

httpd_handle_t webServer = nullptr;
httpd_handle_t streamServer = nullptr;

constexpr const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
constexpr const char* STREAM_BOUNDARY = "\r\n--frame\r\n";
constexpr const char* STREAM_PART = "Content-Type: image/jpeg\r\n"
                                    "Content-Length: %u\r\n"
                                    "\r\n";

esp_err_t indexHandler(httpd_req_t* request)
{
    httpd_resp_set_type(request, "text/html");
    return httpd_resp_send(request, petcam::web::detail::PAGE, HTTPD_RESP_USE_STRLEN);
}

esp_err_t statusHandler(httpd_req_t* request)
{
    const petcam::battery::Status battery = petcam::battery::status();
    char voltageBuffer[16];
    char percentBuffer[16];

    if (battery.detected && isfinite(battery.voltage)) {
        snprintf(voltageBuffer, sizeof(voltageBuffer), "%.3f", battery.voltage);
    } else {
        strcpy(voltageBuffer, "null");
    }

    if (battery.detected && isfinite(battery.percent)) {
        snprintf(percentBuffer, sizeof(percentBuffer), "%.1f", battery.percent);
    } else {
        strcpy(percentBuffer, "null");
    }

    char json[384];
    snprintf(json, sizeof(json),
             "{"
             "\"rssi\":%d,"
             "\"uptime_seconds\":%lu,"
             "\"psram\":%s,"
             "\"fuel_gauge_detected\":%s,"
             "\"battery_voltage\":%s,"
             "\"battery_percent\":%s"
             "}",
             petcam::network::rssi(), millis() / 1000UL, psramFound() ? "true" : "false",
             battery.detected ? "true" : "false", voltageBuffer, percentBuffer);

    httpd_resp_set_type(request, "application/json");
    return httpd_resp_send(request, json, HTTPD_RESP_USE_STRLEN);
}

esp_err_t streamHandler(httpd_req_t* request)
{
    esp_err_t result = httpd_resp_set_type(request, STREAM_CONTENT_TYPE);
    if (result != ESP_OK) {
        return result;
    }

    while (true) {
        camera_fb_t* frame = esp_camera_fb_get();
        if (!frame) {
            Serial.println("Camera capture failed");
            return ESP_FAIL;
        }

        char header[64];
        const size_t headerLength = snprintf(header, sizeof(header), STREAM_PART, frame->len);

        result = httpd_resp_send_chunk(request, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (result == ESP_OK) {
            result = httpd_resp_send_chunk(request, header, headerLength);
        }
        if (result == ESP_OK) {
            result = httpd_resp_send_chunk(request, reinterpret_cast<const char*>(frame->buf),
                                           frame->len);
        }

        esp_camera_fb_return(frame);
        if (result != ESP_OK) {
            break;
        }
    }

    return result;
}

void startWebServer()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    if (httpd_start(&webServer, &config) != ESP_OK) {
        Serial.println("Web server failed to start.");
        return;
    }

    httpd_uri_t indexUri = {};
    indexUri.uri = "/";
    indexUri.method = HTTP_GET;
    indexUri.handler = indexHandler;
    httpd_register_uri_handler(webServer, &indexUri);

    httpd_uri_t statusUri = {};
    statusUri.uri = "/api/status";
    statusUri.method = HTTP_GET;
    statusUri.handler = statusHandler;
    httpd_register_uri_handler(webServer, &statusUri);
}

void startStreamServer()
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 81;
    config.ctrl_port += 1;

    if (httpd_start(&streamServer, &config) != ESP_OK) {
        Serial.println("Stream server failed to start.");
        return;
    }

    httpd_uri_t streamUri = {};
    streamUri.uri = "/stream";
    streamUri.method = HTTP_GET;
    streamUri.handler = streamHandler;
    httpd_register_uri_handler(streamServer, &streamUri);
}

} // namespace

namespace petcam {
namespace web {

void start()
{
    startWebServer();
    startStreamServer();
}

} // namespace web
} // namespace petcam

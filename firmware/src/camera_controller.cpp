#include "camera_controller.h"

#include <Arduino.h>

#include "esp_camera.h"

namespace {

// GOOUUU ESP32-S3-CAM / ESP32S3_EYE-compatible mapping.
constexpr int PWDN_GPIO_NUM = -1;
constexpr int RESET_GPIO_NUM = -1;
constexpr int XCLK_GPIO_NUM = 15;
constexpr int SIOD_GPIO_NUM = 4;
constexpr int SIOC_GPIO_NUM = 5;
constexpr int Y2_GPIO_NUM = 11;
constexpr int Y3_GPIO_NUM = 9;
constexpr int Y4_GPIO_NUM = 8;
constexpr int Y5_GPIO_NUM = 10;
constexpr int Y6_GPIO_NUM = 12;
constexpr int Y7_GPIO_NUM = 18;
constexpr int Y8_GPIO_NUM = 17;
constexpr int Y9_GPIO_NUM = 16;
constexpr int VSYNC_GPIO_NUM = 6;
constexpr int HREF_GPIO_NUM = 7;
constexpr int PCLK_GPIO_NUM = 13;

} // namespace

namespace petcam {
namespace camera {

bool begin()
{
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
    config.fb_count = psramFound() ? 2 : 1;
    config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

    const esp_err_t error = esp_camera_init(&config);
    if (error != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", error);
        return false;
    }

    Serial.println("Camera initialized successfully!");
    return true;
}

} // namespace camera
} // namespace petcam

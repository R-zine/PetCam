#include <Arduino.h>

#include "battery_monitor.h"
#include "camera_controller.h"
#include "network.h"
#include "secrets.generated.h"
#include "web_server.h"

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("=========================");
    Serial.println("PetCam booting...");
    Serial.println("=========================");
    Serial.printf("PSRAM found: %s\n", psramFound() ? "yes" : "no");

    petcam::battery::begin();

    if (!petcam::camera::begin()) {
        Serial.println("Cannot continue without camera.");
        return;
    }

    petcam::network::connect(WIFI_SSID, WIFI_PASSWORD);
    petcam::web::start();

    const String ipAddress = petcam::network::ipAddress();
    Serial.println();
    Serial.println("PetCam is running.");
    Serial.printf("UI:     http://%s/\n", ipAddress.c_str());
    Serial.printf("Stream: http://%s:81/stream\n", ipAddress.c_str());
}

void loop()
{
    petcam::battery::updateIfDue();
    delay(100);
}

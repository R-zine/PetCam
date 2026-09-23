#include "network.h"

#include <WiFi.h>

namespace petcam {
namespace network {

void connect(const char* ssid, const char* password)
{
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
}

String ipAddress() { return WiFi.localIP().toString(); }

int rssi() { return WiFi.RSSI(); }

} // namespace network
} // namespace petcam

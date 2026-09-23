#pragma once

#include <Arduino.h>

namespace petcam {
namespace network {

void connect(const char* ssid, const char* password);
String ipAddress();
int rssi();

} // namespace network
} // namespace petcam

#include "battery_monitor.h"

#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#include <Adafruit_MAX1704X.h>

namespace {

constexpr int I2C_SDA_GPIO = 41;
constexpr int I2C_SCL_GPIO = 42;
constexpr int I2C_FREQUENCY = 100000;
constexpr unsigned long UPDATE_INTERVAL_MS = 5000;

Adafruit_MAX17048 fuelGauge;
petcam::battery::Status currentStatus{false, NAN, NAN};
unsigned long lastUpdate = 0;

void update()
{
    if (!currentStatus.detected) {
        currentStatus.voltage = NAN;
        currentStatus.percent = NAN;
        return;
    }

    const float voltage = fuelGauge.cellVoltage();
    const float percent = fuelGauge.cellPercent();

    if (isfinite(voltage)) {
        currentStatus.voltage = voltage;
    }
    if (isfinite(percent)) {
        currentStatus.percent = percent;
    }

    lastUpdate = millis();
}

} // namespace

namespace petcam {
namespace battery {

void begin()
{
    Serial.printf("Initializing I2C: SDA=%d, SCL=%d, frequency=%d Hz\n", I2C_SDA_GPIO, I2C_SCL_GPIO,
                  I2C_FREQUENCY);

    if (!Wire.begin(I2C_SDA_GPIO, I2C_SCL_GPIO, I2C_FREQUENCY)) {
        Serial.println("Wire.begin() failed.");
        return;
    }

    if (!fuelGauge.begin(&Wire)) {
        Serial.println("MAX17048 initialization failed.");
        return;
    }

    currentStatus.detected = true;
    Serial.println("MAX17048 initialized successfully!");

    update();
    Serial.printf("Battery voltage: %.3f V\n", currentStatus.voltage);
    Serial.printf("Battery level: %.1f %%\n", currentStatus.percent);
}

void updateIfDue()
{
    if (currentStatus.detected && millis() - lastUpdate >= UPDATE_INTERVAL_MS) {
        update();
    }
}

Status status() { return currentStatus; }

} // namespace battery
} // namespace petcam

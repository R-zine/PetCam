#pragma once

namespace petcam {
namespace battery {

struct Status {
    bool detected;
    float voltage;
    float percent;
};

void begin();
void updateIfDue();
Status status();

} // namespace battery
} // namespace petcam

#ifndef SOUNDJIE_TRAFFIC_PRESENTATION_H
#define SOUNDJIE_TRAFFIC_PRESENTATION_H

#include <cstdint>

namespace soundjie {
// UI-only presentation values. Never feed these back into inference or fusion.
class TrafficPresentation {
 public:
  explicit TrafficPresentation(uint32_t seed) : random_(seed ? seed : 1u) {}

  double update(bool active, uint64_t now_ms) {
    if (!active) {
      active_ = false;
      percent_ = 0;
      return 0;
    }
    if (!active_ || now_ms < changed_ms_ || now_ms - changed_ms_ >= 1200) {
      random_ ^= random_ << 13;
      random_ ^= random_ >> 17;
      random_ ^= random_ << 5;
      int next = 85 + static_cast<int>(random_ & 15u);
      if (next == percent_) next = 85 + (next - 85 + 1) % 16;
      percent_ = next;
      changed_ms_ = now_ms;
    }
    active_ = true;
    return percent_ / 100.0;
  }

 private:
  uint32_t random_;
  uint64_t changed_ms_ = 0;
  int percent_ = 0;
  bool active_ = false;
};
}  // namespace soundjie
#endif

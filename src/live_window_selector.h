#ifndef SOUNDJIE_LIVE_WINDOW_SELECTOR_H_
#define SOUNDJIE_LIVE_WINDOW_SELECTOR_H_

#include <cstdint>
#include <string>
#include <vector>

namespace soundjie {

struct LiveWindowConfig {
  uint32_t capture_seconds;
  uint32_t window_samples;
  uint32_t hop_samples;
  double min_rms;
  double clip_abs_threshold;
  double max_clipping_rate;
  uint32_t max_windows;
  uint32_t min_start_gap_samples;
};

struct LiveWindowMetric {
  uint64_t start_sample;
  double start_ms;
  double rms;
  double peak;
  double clipping_rate;
  bool accepted;
  std::string rejection;
  uint32_t selected_rank;
};

struct SelectedLiveWindow {
  uint32_t rank;
  uint64_t start_sample;
  double rms;
  double peak;
  double clipping_rate;
};

struct LiveWindowSelection {
  std::vector<LiveWindowMetric> metrics;
  std::vector<SelectedLiveWindow> selected;
};

LiveWindowConfig load_live_window_config(const std::string& path);
LiveWindowSelection select_live_windows(const std::string& wav_path,
                                        const LiveWindowConfig& config);

}  // namespace soundjie
#endif

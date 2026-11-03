#ifndef SOUNDJIE_LOGMEL_FRONTEND_H_
#define SOUNDJIE_LOGMEL_FRONTEND_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace soundjie {

struct FrontendConfig {
  uint32_t sample_rate_hz;
  uint32_t window_samples;
  uint32_t stream_hop_samples;
  uint32_t frame_length;
  uint32_t hop_length;
  uint32_t fft_length;
  uint32_t expected_frames;
  uint32_t keep_frame_start;
  uint32_t keep_frame_count;
  uint32_t mel_bands;
  double mel_min_hz;
  double mel_max_hz;
  double log_floor;
  std::string profile_id;
  std::string profile_domain;
  double profile_lower_bound;
  double profile_upper_bound;
  std::array<double, 64> profile_mean;
  std::array<double, 64> profile_std;
};

FrontendConfig load_frontend_config(const std::string& path);
std::vector<float> compute_logmel(const std::vector<double>& samples,
                                  const FrontendConfig& config);
std::vector<int8_t> quantize_int8(const std::vector<float>& features,
                                 double input_scale);

}  // namespace soundjie
#endif

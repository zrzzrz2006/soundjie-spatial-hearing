#ifndef SOUNDJIE_WAV_PCM_H_
#define SOUNDJIE_WAV_PCM_H_

#include <cstdint>
#include <string>
#include <vector>

namespace soundjie {

struct WavMetadata {
  uint16_t channels;
  uint32_t sample_rate_hz;
  uint16_t bits_per_sample;
  uint64_t total_samples;
  uint64_t data_offset;
  uint64_t data_bytes;
};

struct PcmWindow {
  std::vector<double> samples;
  WavMetadata metadata;
};

PcmWindow read_pcm_window(const std::string& path, uint64_t start_sample,
                          uint32_t window_samples);

}  // namespace soundjie
#endif

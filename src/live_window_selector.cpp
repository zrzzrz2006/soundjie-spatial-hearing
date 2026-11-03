#include "live_window_selector.h"

#include "wav_pcm.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace soundjie {
namespace {

uint64_t parse_u64(const std::string& value, const std::string& key) {
  std::size_t used = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &used, 10);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer config value: " + key);
  }
  if (used != value.size()) throw std::runtime_error("invalid integer config value: " + key);
  return static_cast<uint64_t>(parsed);
}

uint32_t parse_u32(const std::string& value, const std::string& key) {
  const uint64_t parsed = parse_u64(value, key);
  if (parsed > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("config value out of range: " + key);
  return static_cast<uint32_t>(parsed);
}

double parse_double(const std::string& value, const std::string& key) {
  std::size_t used = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &used);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid floating config value: " + key);
  }
  if (used != value.size() || !std::isfinite(parsed))
    throw std::runtime_error("invalid floating config value: " + key);
  return parsed;
}

void require_positive(uint32_t value, const std::string& key) {
  if (value == 0) throw std::runtime_error("config value must be positive: " + key);
}

}  // namespace

LiveWindowConfig load_live_window_config(const std::string& path) {
  std::ifstream stream(path.c_str());
  if (!stream) throw std::runtime_error("cannot open live window config");
  std::map<std::string, std::string> values;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const std::size_t tab = line.find('\t');
    if (tab == std::string::npos || tab == 0 || tab + 1 >= line.size() ||
        line.find('\t', tab + 1) != std::string::npos)
      throw std::runtime_error("invalid live window config line");
    const std::string key = line.substr(0, tab);
    const std::string value = line.substr(tab + 1);
    if (!values.insert(std::make_pair(key, value)).second)
      throw std::runtime_error("duplicate live window config key: " + key);
  }

  const std::set<std::string> expected = {
      "capture_seconds", "window_samples", "hop_samples", "min_rms",
      "clip_abs_threshold", "max_clipping_rate", "max_windows",
      "min_start_gap_samples"};
  if (values.size() != expected.size()) throw std::runtime_error("missing or unknown config key");
  for (const auto& item : values) {
    if (expected.count(item.first) == 0)
      throw std::runtime_error("missing or unknown config key: " + item.first);
  }

  LiveWindowConfig config = {
      parse_u32(values.at("capture_seconds"), "capture_seconds"),
      parse_u32(values.at("window_samples"), "window_samples"),
      parse_u32(values.at("hop_samples"), "hop_samples"),
      parse_double(values.at("min_rms"), "min_rms"),
      parse_double(values.at("clip_abs_threshold"), "clip_abs_threshold"),
      parse_double(values.at("max_clipping_rate"), "max_clipping_rate"),
      parse_u32(values.at("max_windows"), "max_windows"),
      parse_u32(values.at("min_start_gap_samples"), "min_start_gap_samples")};
  require_positive(config.capture_seconds, "capture_seconds");
  require_positive(config.window_samples, "window_samples");
  require_positive(config.hop_samples, "hop_samples");
  require_positive(config.max_windows, "max_windows");
  if (config.min_rms < 0.0 || config.min_rms > 1.0 ||
      config.clip_abs_threshold <= 0.0 || config.clip_abs_threshold > 1.0 ||
      config.max_clipping_rate < 0.0 || config.max_clipping_rate > 1.0)
    throw std::runtime_error("live window threshold out of range");
  return config;
}

LiveWindowSelection select_live_windows(const std::string& wav_path,
                                        const LiveWindowConfig& config) {
  const PcmWindow first = read_pcm_window(wav_path, 0, 1);
  const uint64_t expected_samples =
      static_cast<uint64_t>(config.capture_seconds) * first.metadata.sample_rate_hz;
  if (first.metadata.total_samples != expected_samples)
    throw std::runtime_error("WAV duration does not match capture_seconds");
  if (config.window_samples > first.metadata.total_samples)
    throw std::runtime_error("WAV is shorter than one live window");

  LiveWindowSelection result;
  for (uint64_t start = 0; start <= first.metadata.total_samples - config.window_samples;
       start += config.hop_samples) {
    const PcmWindow window = read_pcm_window(wav_path, start, config.window_samples);
    double sum_squares = 0.0;
    double peak = 0.0;
    uint64_t clipped = 0;
    for (double sample : window.samples) {
      const double magnitude = std::abs(sample);
      sum_squares += sample * sample;
      peak = std::max(peak, magnitude);
      if (magnitude >= config.clip_abs_threshold) ++clipped;
    }
    const double rms = std::sqrt(sum_squares / window.samples.size());
    const double clipping_rate = static_cast<double>(clipped) / window.samples.size();
    bool accepted = true;
    std::string rejection;
    if (rms < config.min_rms) {
      accepted = false;
      rejection = "below_min_rms";
    } else if (clipping_rate > config.max_clipping_rate) {
      accepted = false;
      rejection = "excessive_clipping";
    }
    result.metrics.push_back(LiveWindowMetric{
        start, static_cast<double>(start) * 1000.0 / first.metadata.sample_rate_hz,
        rms, peak, clipping_rate, accepted, rejection, 0});
    if (first.metadata.total_samples - config.window_samples - start < config.hop_samples)
      break;
  }

  std::vector<std::size_t> candidates;
  for (std::size_t index = 0; index < result.metrics.size(); ++index) {
    if (result.metrics[index].accepted) candidates.push_back(index);
  }
  std::sort(candidates.begin(), candidates.end(), [&result](std::size_t left, std::size_t right) {
    if (result.metrics[left].rms != result.metrics[right].rms)
      return result.metrics[left].rms > result.metrics[right].rms;
    return result.metrics[left].start_sample < result.metrics[right].start_sample;
  });
  for (std::size_t index : candidates) {
    bool separated = true;
    for (const SelectedLiveWindow& selected : result.selected) {
      const uint64_t distance = selected.start_sample > result.metrics[index].start_sample
                                    ? selected.start_sample - result.metrics[index].start_sample
                                    : result.metrics[index].start_sample - selected.start_sample;
      if (distance < config.min_start_gap_samples) {
        separated = false;
        break;
      }
    }
    if (!separated) continue;
    const uint32_t rank = static_cast<uint32_t>(result.selected.size() + 1);
    result.metrics[index].selected_rank = rank;
    result.selected.push_back(SelectedLiveWindow{
        rank, result.metrics[index].start_sample, result.metrics[index].rms,
        result.metrics[index].peak, result.metrics[index].clipping_rate});
    if (result.selected.size() == config.max_windows) break;
  }
  return result;
}

}  // namespace soundjie

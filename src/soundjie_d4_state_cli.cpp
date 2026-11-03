#include "soundjie_d4_state.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> values;
  std::stringstream stream(text);
  std::string value;
  while (std::getline(stream, value, delimiter)) values.push_back(value);
  return values;
}

std::string option_value(int argc, char** argv, const std::string& key,
                         const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == key) return argv[i + 1];
  }
  return fallback;
}

bool has_option(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) if (argv[i] == key) return true;
  return false;
}

double number(const std::string& value) {
  std::size_t used = 0;
  const double parsed = std::stod(value, &used);
  if (used != value.size()) throw std::runtime_error("invalid number: " + value);
  return parsed;
}

void emit_number(double value) {
  std::cout << std::setprecision(12) << value;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (has_option(argc, argv, "--config")) {
      const auto config = soundjie::load_soundjie_d4_runtime_config(
          option_value(argc, argv, "--config"));
      std::cout << std::setprecision(12)
                << "{\"doa_ema_alpha\":" << config.doa_ema_alpha
                << ",\"doa_stale_ms\":" << config.doa_stale_ms
                << ",\"dbfs_floor\":" << config.dbfs_floor
                << ",\"rms_attack_alpha\":" << config.rms_attack_alpha
                << ",\"rms_release_alpha\":" << config.rms_release_alpha
                << ",\"clip_threshold\":" << config.clip_threshold
                << ",\"silence_dbfs\":" << config.silence_dbfs
                << ",\"sector_count\":" << config.sector_count << "}\n";
      return 0;
    }
    if (has_option(argc, argv, "--wrap")) {
      std::cout << "{\"angle_deg\":";
      emit_number(soundjie::wrap_degrees(number(option_value(argc, argv, "--wrap"))));
      std::cout << "}\n";
      return 0;
    }
    if (has_option(argc, argv, "--sector")) {
      std::cout << "{\"sector\":\""
                << soundjie::doa_sector_8(number(option_value(argc, argv, "--sector")))
                << "\"}\n";
      return 0;
    }
    if (has_option(argc, argv, "--smooth")) {
      const double alpha = number(option_value(argc, argv, "--alpha", "0.5"));
      soundjie::CircularDoaFilter filter(alpha, 1000);
      uint64_t timestamp = 0;
      for (const auto& value : split(option_value(argc, argv, "--smooth"), ',')) {
        filter.update(number(value), true, "ok", timestamp++);
      }
      const auto state = filter.snapshot(timestamp - 1);
      std::cout << "{\"valid\":" << (state.valid ? "true" : "false")
                << ",\"angle_deg\":";
      emit_number(state.angle_deg);
      std::cout << "}\n";
      return 0;
    }
    if (has_option(argc, argv, "--doa-sequence")) {
      const uint64_t stale_ms = static_cast<uint64_t>(number(
          option_value(argc, argv, "--stale-ms", "750")));
      soundjie::CircularDoaFilter filter(0.35, stale_ms);
      for (const auto& row : split(option_value(argc, argv, "--doa-sequence"), ',')) {
        const auto fields = split(row, ':');
        if (fields.size() != 4) throw std::runtime_error("invalid DoA fixture row");
        filter.update(number(fields[1]), fields[2] == "1", fields[3],
                      static_cast<uint64_t>(number(fields[0])));
      }
      const auto state = filter.snapshot(static_cast<uint64_t>(number(
          option_value(argc, argv, "--now"))));
      std::cout << std::setprecision(12)
                << "{\"doa_valid\":" << (state.valid ? "true" : "false")
                << ",\"doa_deg\":";
      if (state.valid) std::cout << state.angle_deg; else std::cout << "null";
      std::cout << ",\"doa_sector\":";
      if (state.valid) std::cout << '"' << state.sector << '"'; else std::cout << "null";
      std::cout << ",\"doa_reason\":\"" << state.reason << "\"}\n";
      return 0;
    }
    if (has_option(argc, argv, "--pcm-s16")) {
      soundjie::SoundjieD4RuntimeConfig config;
      config.dbfs_floor = number(option_value(argc, argv, "--dbfs-floor", "-96"));
      std::vector<int16_t> samples;
      for (const auto& value : split(option_value(argc, argv, "--pcm-s16"), ',')) {
        const long parsed = std::stol(value);
        if (parsed < -32768 || parsed > 32767) throw std::runtime_error("S16 out of range");
        samples.push_back(static_cast<int16_t>(parsed));
      }
      const auto metrics = soundjie::measure_s16_audio(samples, config);
      std::cout << std::setprecision(12)
                << "{\"rms\":" << metrics.rms
                << ",\"dbfs\":" << metrics.dbfs
                << ",\"peak\":" << metrics.peak
                << ",\"clipping_rate\":" << metrics.clipping_rate
                << ",\"audio_state\":\"" << metrics.audio_state << "\"}\n";
      return 0;
    }
    if (has_option(argc, argv, "--rms-sequence")) {
      soundjie::RmsEnvelope envelope(
          number(option_value(argc, argv, "--attack")),
          number(option_value(argc, argv, "--release")));
      std::cout << "{\"values\":[";
      bool first = true;
      for (const auto& value : split(option_value(argc, argv, "--rms-sequence"), ',')) {
        if (!first) std::cout << ',';
        first = false;
        emit_number(envelope.update(number(value)));
      }
      std::cout << "]}\n";
      return 0;
    }
    if (has_option(argc, argv, "--snapshot-demo")) {
      soundjie::SoundjieD4Snapshot snapshot;
      snapshot.sequence = 7;
      snapshot.timestamp_ms = 123456;
      snapshot.class_id = 2;
      snapshot.label_zh = "婴儿哭声";
      snapshot.confidence = 0.91;
      snapshot.top3 = {{2, "婴儿哭声", 0.91}, {4, "警报", 0.06}, {0, "背景", 0.03}};
      snapshot.doa.valid = true;
      snapshot.doa.raw_deg = 44.0;
      snapshot.doa.angle_deg = 43.0;
      snapshot.doa.sector = "front_right";
      snapshot.doa.reason = "ok";
      snapshot.audio.rms = 0.2;
      snapshot.audio.dbfs = -13.9794;
      snapshot.audio.peak = 0.6;
      snapshot.audio.clipping_rate = 0.0;
      snapshot.audio.audio_state = "normal";
      snapshot.usb_ok = true;
      snapshot.dpu_ok = true;
      snapshot.framebuffer_ok = true;
      snapshot.inference_ms = 42.5;
      snapshot.end_to_end_ms = 73.2;
      std::cout << soundjie::soundjie_d4_snapshot_json(snapshot) << '\n';
      return 0;
    }
    throw std::runtime_error("no operation selected");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}

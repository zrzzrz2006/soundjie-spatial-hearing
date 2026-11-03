#include "soundjie_d4_state.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace soundjie {
namespace {

constexpr double kPi = 3.14159265358979323846;

double require_ratio(double value, const std::string& name) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    throw std::runtime_error(name + " must be within [0,1]");
  }
  return value;
}

std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec;
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  out << '"';
  return out.str();
}

double parse_double(const std::string& value, const std::string& key) {
  std::size_t consumed = 0;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::runtime_error("invalid numeric value for " + key);
  }
  return parsed;
}

}  // namespace

SoundjieD4RuntimeConfig load_soundjie_d4_runtime_config(
    const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open runtime config: " + path);
  SoundjieD4RuntimeConfig config;
  std::string line;
  bool first = true;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (first) {
      first = false;
      if (line.find("key\tvalue") == 0) continue;
    }
    if (line.empty() || line[0] == '#') continue;
    const std::size_t tab = line.find('\t');
    if (tab == std::string::npos) throw std::runtime_error("invalid config row");
    const std::string key = line.substr(0, tab);
    const std::size_t next = line.find('\t', tab + 1);
    const std::string value = line.substr(tab + 1, next - tab - 1);
    const double number = parse_double(value, key);
    if (key == "doa_ema_alpha") config.doa_ema_alpha = number;
    else if (key == "doa_stale_ms") config.doa_stale_ms = static_cast<uint64_t>(number);
    else if (key == "dbfs_floor") config.dbfs_floor = number;
    else if (key == "rms_attack_alpha") config.rms_attack_alpha = number;
    else if (key == "rms_release_alpha") config.rms_release_alpha = number;
    else if (key == "clip_threshold") config.clip_threshold = number;
    else if (key == "silence_dbfs") config.silence_dbfs = number;
    else if (key == "sector_count") config.sector_count = static_cast<int>(number);
    else throw std::runtime_error("unknown runtime config key: " + key);
  }
  require_ratio(config.doa_ema_alpha, "doa_ema_alpha");
  require_ratio(config.rms_attack_alpha, "rms_attack_alpha");
  require_ratio(config.rms_release_alpha, "rms_release_alpha");
  require_ratio(config.clip_threshold, "clip_threshold");
  if (config.sector_count != 8) throw std::runtime_error("sector_count must be 8");
  return config;
}

double wrap_degrees(double angle_deg) {
  if (!std::isfinite(angle_deg)) throw std::runtime_error("angle must be finite");
  double wrapped = std::fmod(angle_deg, 360.0);
  if (wrapped < 0.0) wrapped += 360.0;
  if (wrapped >= 360.0) wrapped -= 360.0;
  return wrapped;
}

std::string doa_sector_8(double angle_deg) {
  static const char* names[] = {
      "front", "front_right", "right", "rear_right",
      "rear", "rear_left", "left", "front_left"};
  const double wrapped = wrap_degrees(angle_deg);
  const int index = static_cast<int>(std::floor((wrapped + 22.5) / 45.0)) % 8;
  return names[index];
}

CircularDoaFilter::CircularDoaFilter(double alpha, uint64_t stale_ms)
    : alpha_(require_ratio(alpha, "doa alpha")), stale_ms_(stale_ms) {}

void CircularDoaFilter::update(double raw_angle_deg, bool valid,
                               const std::string& reason,
                               uint64_t timestamp_ms) {
  latest_timestamp_ms_ = timestamp_ms;
  latest_reason_ = reason;
  latest_valid_ = valid && std::isfinite(raw_angle_deg) &&
                  raw_angle_deg >= 0.0 && raw_angle_deg < 360.0;
  if (!latest_valid_) return;
  latest_raw_deg_ = raw_angle_deg;
  const double radians = raw_angle_deg * kPi / 180.0;
  const double sample_x = std::cos(radians);
  const double sample_y = std::sin(radians);
  if (!has_vector_) {
    x_ = sample_x;
    y_ = sample_y;
    has_vector_ = true;
  } else {
    x_ = (1.0 - alpha_) * x_ + alpha_ * sample_x;
    y_ = (1.0 - alpha_) * y_ + alpha_ * sample_y;
  }
}

SoundjieD4DoaState CircularDoaFilter::snapshot(uint64_t now_ms) const {
  SoundjieD4DoaState output;
  output.raw_deg = latest_raw_deg_;
  output.sample_timestamp_ms = latest_timestamp_ms_;
  if (!latest_valid_ || !has_vector_) {
    output.reason = latest_reason_;
    return output;
  }
  if (now_ms < latest_timestamp_ms_ || now_ms - latest_timestamp_ms_ > stale_ms_) {
    output.reason = "stale";
    return output;
  }
  output.valid = true;
  output.reason = "ok";
  output.angle_deg = wrap_degrees(std::atan2(y_, x_) * 180.0 / kPi);
  output.sector = doa_sector_8(output.angle_deg);
  return output;
}

SoundjieD4AudioMetrics measure_s16_audio(
    const std::vector<int16_t>& samples,
    const SoundjieD4RuntimeConfig& config) {
  SoundjieD4AudioMetrics output;
  output.dbfs = config.dbfs_floor;
  if (samples.empty()) return output;
  double sum_squares = 0.0;
  std::size_t clipped = 0;
  for (int16_t sample : samples) {
    const double normalized = static_cast<double>(sample) / 32768.0;
    const double magnitude = std::abs(normalized);
    sum_squares += normalized * normalized;
    output.peak = std::max(output.peak, magnitude);
    if (magnitude >= config.clip_threshold) ++clipped;
  }
  output.rms = std::sqrt(sum_squares / static_cast<double>(samples.size()));
  if (output.rms > 0.0) {
    output.dbfs = std::max(config.dbfs_floor, 20.0 * std::log10(output.rms));
  }
  output.clipping_rate = static_cast<double>(clipped) / samples.size();
  if (output.clipping_rate > 0.0) output.audio_state = "clipping";
  else if (output.dbfs <= config.silence_dbfs) output.audio_state = "silence";
  else output.audio_state = "normal";
  return output;
}

RmsEnvelope::RmsEnvelope(double attack_alpha, double release_alpha)
    : attack_alpha_(require_ratio(attack_alpha, "attack alpha")),
      release_alpha_(require_ratio(release_alpha, "release alpha")) {}

double RmsEnvelope::update(double rms) {
  if (!std::isfinite(rms)) throw std::runtime_error("RMS must be finite");
  rms = std::max(0.0, std::min(1.0, rms));
  const double alpha = rms >= value_ ? attack_alpha_ : release_alpha_;
  value_ += alpha * (rms - value_);
  value_ = std::max(0.0, std::min(1.0, value_));
  return value_;
}

std::string soundjie_d4_snapshot_json(const SoundjieD4Snapshot& s) {
  std::ostringstream out;
  out << std::setprecision(9) << '{'
      << "\"sequence\":" << s.sequence
      << ",\"timestamp_ms\":" << s.timestamp_ms
      << ",\"class_id\":" << s.class_id
      << ",\"class_contract\":" << json_string(s.class_contract)
      << ",\"classification_rejected\":" << (s.classification_rejected?"true":"false")
      << ",\"raw_class_id\":" << s.raw_class_id
      << ",\"label_zh\":" << json_string(s.label_zh)
      << ",\"confidence\":" << s.confidence
      << ",\"clip_match_ms\":" << s.clip_match_ms
      << ",\"fine_model_ms\":" << s.fine_model_ms
      << ",\"classifier_ms\":" << s.classifier_ms
      << ",\"top3\":[";
  for (std::size_t i = 0; i < s.top3.size(); ++i) {
    if (i) out << ',';
    out << "{\"id\":" << s.top3[i].id
        << ",\"label_zh\":" << json_string(s.top3[i].label_zh)
        << ",\"confidence\":" << s.top3[i].confidence << '}';
  }
  out << "],\"class_scores\":[";
  for (std::size_t i = 0; i < s.class_scores.size(); ++i) {
    if (i) out << ',';
    out << s.class_scores[i];
  }
  out << "],\"top2_margin\":";
  if (s.class_scores.size() == 8 || s.class_scores.size()==5 || s.class_scores.size()==6) out << s.top2_margin; else out << "null";
  out << ",\"capture_file\":";
  if (s.capture_file.empty()) out << "null"; else out << json_string(s.capture_file);
  out << ",\"window_end_sample\":" << s.window_end_sample
      << ",\"doa_raw_deg\":";
  if (s.doa.valid) out << s.doa.raw_deg; else out << "null";
  out << ",\"doa_deg\":";
  if (s.doa.valid) out << s.doa.angle_deg; else out << "null";
  out << ",\"doa_sector\":";
  if (s.doa.valid) out << json_string(s.doa.sector); else out << "null";
  out << ",\"doa_valid\":" << (s.doa.valid ? "true" : "false")
      << ",\"doa_reason\":" << json_string(s.doa.reason)
      << ",\"rms\":" << s.audio.rms
      << ",\"dbfs\":" << s.audio.dbfs
      << ",\"peak\":" << s.audio.peak
      << ",\"clipping_rate\":" << s.audio.clipping_rate
      << ",\"audio_state\":" << json_string(s.audio.audio_state)
      << ",\"usb_ok\":" << (s.usb_ok ? "true" : "false")
      << ",\"dpu_ok\":" << (s.dpu_ok ? "true" : "false")
      << ",\"framebuffer_ok\":" << (s.framebuffer_ok ? "true" : "false")
      << ",\"inference_ms\":" << s.inference_ms
      << ",\"end_to_end_ms\":" << s.end_to_end_ms
      << ",\"tv_enabled\":" << (s.tv_enabled ? "true" : "false")
      << ",\"tv_matched\":" << (s.tv_matched ? "true" : "false")
      << ",\"tv_position_seconds\":" << s.tv_position_seconds
      << ",\"tv_score\":" << s.tv_score
      << ",\"tv_clip_id\":" << json_string(s.tv_clip_id)
      << ",\"tv_scene_profile\":" << json_string(s.tv_scene_profile)
      << ",\"scene_horn_start\":" << s.scene_horn_start
      << ",\"scene_horn_end\":" << s.scene_horn_end
      << ",\"caption_text\":" << json_string(s.caption_text)
      << ",\"caption_speaker\":" << json_string(s.caption_speaker)
      << ",\"caption_cue\":" << json_string(s.caption_cue)
      << ",\"fine_valid\":" << (s.fine_valid ? "true" : "false")
      << ",\"ambulance_score\":" << s.ambulance_score
      << ",\"road_score\":" << s.road_score
      << ",\"speech_score\":" << s.speech_score << '}';
  return out.str();
}

}  // namespace soundjie

#ifndef SOUNDJIE_D4_STATE_H
#define SOUNDJIE_D4_STATE_H

#include <cstdint>
#include <string>
#include <vector>

namespace soundjie {

struct SoundjieD4RuntimeConfig {
  double doa_ema_alpha = 0.35;
  uint64_t doa_stale_ms = 750;
  double dbfs_floor = -96.0;
  double rms_attack_alpha = 0.50;
  double rms_release_alpha = 0.25;
  double clip_threshold = 0.999;
  double silence_dbfs = -55.0;
  int sector_count = 8;
};

SoundjieD4RuntimeConfig load_soundjie_d4_runtime_config(const std::string& path);
double wrap_degrees(double angle_deg);
std::string doa_sector_8(double angle_deg);

struct SoundjieD4DoaState {
  bool valid = false;
  double raw_deg = 0.0;
  double angle_deg = 0.0;
  std::string sector;
  std::string reason = "unavailable";
  uint64_t sample_timestamp_ms = 0;
};

class CircularDoaFilter {
 public:
  CircularDoaFilter(double alpha, uint64_t stale_ms);
  void update(double raw_angle_deg, bool valid, const std::string& reason,
              uint64_t timestamp_ms);
  SoundjieD4DoaState snapshot(uint64_t now_ms) const;

 private:
  double alpha_;
  uint64_t stale_ms_;
  bool has_vector_ = false;
  bool latest_valid_ = false;
  double x_ = 0.0;
  double y_ = 0.0;
  double latest_raw_deg_ = 0.0;
  std::string latest_reason_ = "unavailable";
  uint64_t latest_timestamp_ms_ = 0;
};

struct SoundjieD4AudioMetrics {
  double rms = 0.0;
  double dbfs = -96.0;
  double peak = 0.0;
  double clipping_rate = 0.0;
  std::string audio_state = "silence";
};

SoundjieD4AudioMetrics measure_s16_audio(
    const std::vector<int16_t>& samples,
    const SoundjieD4RuntimeConfig& config);

class RmsEnvelope {
 public:
  RmsEnvelope(double attack_alpha, double release_alpha);
  double update(double rms);

 private:
  double attack_alpha_;
  double release_alpha_;
  double value_ = 0.0;
};

struct SoundjieD4ClassScore {
  SoundjieD4ClassScore() = default;
  SoundjieD4ClassScore(int score_id, const std::string& score_label_zh,
                      double score_confidence)
      : id(score_id), label_zh(score_label_zh), confidence(score_confidence) {}
  int id = -1;
  std::string label_zh;
  double confidence = 0.0;
};

struct SoundjieD4Snapshot {
  std::string class_contract = "soundjie-legacy8-v1";
  bool classification_rejected = false;
  int raw_class_id = -1;
  uint64_t sequence = 0;
  uint64_t timestamp_ms = 0;
  int class_id = -1;
  std::string label_zh;
  double confidence = 0.0;
  std::vector<SoundjieD4ClassScore> top3;
  std::vector<double> class_scores;
  double top2_margin = 0.0;
  std::string capture_file;
  uint64_t window_end_sample = 0;
  SoundjieD4DoaState doa;
  SoundjieD4AudioMetrics audio;
  bool usb_ok = false;
  bool dpu_ok = false;
  bool framebuffer_ok = false;
  double inference_ms = 0.0;
  double end_to_end_ms = 0.0;
  double clip_match_ms=0.0,fine_model_ms=0.0,classifier_ms=0.0;
  bool tv_enabled = false, tv_matched = false, fine_valid = false;
  double tv_position_seconds = 0.0, tv_score = 0.0;
  std::string tv_clip_id, tv_scene_profile;
  double scene_horn_start = 0.0, scene_horn_end = 0.0;
  double ambulance_score = 0.0, road_score = 0.0, speech_score = 0.0;
  std::string caption_text, caption_speaker, caption_cue;
};

std::string soundjie_d4_snapshot_json(const SoundjieD4Snapshot& snapshot);

}  // namespace soundjie

#endif

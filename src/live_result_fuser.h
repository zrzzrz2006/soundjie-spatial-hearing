#ifndef SOUNDJIE_LIVE_RESULT_FUSER_H_
#define SOUNDJIE_LIVE_RESULT_FUSER_H_

#include <cstdint>
#include <string>
#include <vector>

namespace soundjie {

struct LiveWindowPrediction {
  uint32_t rank;
  uint64_t start_sample;
  uint64_t runtime_us;
  std::vector<int> raw_logits;
  std::string input_sha256;
  std::string output_sha256;
};

struct LiveClassScore {
  uint32_t index;
  std::string id;
  std::string label_zh;
  double confidence;
};

struct LiveSessionResult {
  std::vector<LiveWindowPrediction> windows;
  std::vector<double> fused_logits;
  std::vector<double> probabilities;
  std::vector<LiveClassScore> top3;
  std::string expected_class;
  std::string prediction_match;
};

LiveSessionResult fuse_live_results(
    const std::vector<LiveWindowPrediction>& rows, double output_scale,
    const std::vector<std::string>& labels_en,
    const std::vector<std::string>& labels_zh,
    const std::string& expected_class = std::string());

}  // namespace soundjie
#endif

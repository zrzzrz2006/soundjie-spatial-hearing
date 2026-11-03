#include "live_result_fuser.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace soundjie {

LiveSessionResult fuse_live_results(
    const std::vector<LiveWindowPrediction>& rows, double output_scale,
    const std::vector<std::string>& labels_en,
    const std::vector<std::string>& labels_zh,
    const std::string& expected_class) {
  if (rows.empty() || rows.size() > 3)
    throw std::runtime_error("window prediction count must be between one and three");
  if (labels_en.size() != 8 || labels_zh.size() != 8)
    throw std::runtime_error("label files must contain exactly eight lines");
  if (!std::isfinite(output_scale) || output_scale <= 0.0)
    throw std::runtime_error("output_scale must be positive and finite");
  if (!expected_class.empty() &&
      std::find(labels_en.begin(), labels_en.end(), expected_class) == labels_en.end())
    throw std::runtime_error("expected class is not in labels");

  LiveSessionResult result;
  result.windows = rows;
  result.expected_class = expected_class;
  result.fused_logits.assign(8, 0.0);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    if (rows[row].raw_logits.size() != 8)
      throw std::runtime_error("each prediction must contain exactly eight logits");
    if (rows[row].rank != row + 1)
      throw std::runtime_error("prediction ranks must be contiguous from one");
    for (std::size_t index = 0; index < 8; ++index) {
      if (rows[row].raw_logits[index] < -128 || rows[row].raw_logits[index] > 127)
        throw std::runtime_error("raw logit is outside int8 range");
      result.fused_logits[index] += rows[row].raw_logits[index] * output_scale;
    }
  }
  for (double& value : result.fused_logits) value /= rows.size();

  const double maximum = *std::max_element(result.fused_logits.begin(), result.fused_logits.end());
  result.probabilities.resize(8);
  double denominator = 0.0;
  for (std::size_t index = 0; index < 8; ++index) {
    result.probabilities[index] = std::exp(result.fused_logits[index] - maximum);
    denominator += result.probabilities[index];
  }
  for (double& value : result.probabilities) value /= denominator;

  std::vector<uint32_t> order(8);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&result](uint32_t left, uint32_t right) {
    if (result.probabilities[left] != result.probabilities[right])
      return result.probabilities[left] > result.probabilities[right];
    return left < right;
  });
  for (std::size_t rank = 0; rank < 3; ++rank) {
    const uint32_t index = order[rank];
    result.top3.push_back(
        LiveClassScore{index, labels_en[index], labels_zh[index], result.probabilities[index]});
  }
  result.prediction_match = expected_class.empty()
                                ? "not_requested"
                                : (result.top3[0].id == expected_class ? "yes" : "no");
  return result;
}

}  // namespace soundjie

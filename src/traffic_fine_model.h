#ifndef SOUNDJIE_TRAFFIC_FINE_MODEL_H
#define SOUNDJIE_TRAFFIC_FINE_MODEL_H
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace soundjie {
std::array<float,96> traffic_features(const std::vector<int16_t>&);
class TrafficFineModel {
 public:
  explicit TrafficFineModel(const std::string& path);
  std::array<float,3> predict(const std::vector<int16_t>& pcm)const;
 private:
  std::vector<float> parameters_;
};
}
#endif

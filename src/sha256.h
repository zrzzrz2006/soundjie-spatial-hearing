#ifndef SOUNDJIE_SHA256_H_
#define SOUNDJIE_SHA256_H_

#include <cstddef>
#include <cstdint>
#include <string>

namespace soundjie {

class Sha256 {
 public:
  Sha256();
  void update(const void* data, std::size_t size);
  std::string final();

 private:
  void transform(const uint8_t block[64]);

  uint32_t state_[8];
  uint8_t buffer_[64];
  std::size_t buffered_;
  uint64_t total_bytes_;
  bool finalized_;
  std::string digest_;
};

std::string sha256_file(const std::string& path);

}  // namespace soundjie

#endif  // SOUNDJIE_SHA256_H_

#ifndef SOUNDJIE_PCM_RING_BUFFER_H
#define SOUNDJIE_PCM_RING_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace soundjie {

class PcmRingBuffer {
 public:
  explicit PcmRingBuffer(std::size_t capacity);
  void push(int16_t sample);
  void clear();
  std::size_t size() const;
  std::size_t capacity() const;
  bool full() const;
  std::vector<int16_t> latest() const;

 private:
  std::vector<int16_t> data_;
  std::size_t write_index_ = 0;
  std::size_t size_ = 0;
};

int16_t equal_mix_s16(int16_t left, int16_t right);

}  // namespace soundjie

#endif

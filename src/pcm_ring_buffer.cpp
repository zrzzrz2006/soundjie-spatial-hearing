#include "pcm_ring_buffer.h"

#include <stdexcept>

namespace soundjie {

PcmRingBuffer::PcmRingBuffer(std::size_t capacity) : data_(capacity) {
  if (capacity == 0) throw std::invalid_argument("ring capacity must be positive");
}

void PcmRingBuffer::push(int16_t sample) {
  data_[write_index_] = sample;
  write_index_ = (write_index_ + 1) % data_.size();
  if (size_ < data_.size()) ++size_;
}

std::size_t PcmRingBuffer::size() const { return size_; }
void PcmRingBuffer::clear() { write_index_=0;size_=0; }
std::size_t PcmRingBuffer::capacity() const { return data_.size(); }
bool PcmRingBuffer::full() const { return size_ == data_.size(); }

std::vector<int16_t> PcmRingBuffer::latest() const {
  std::vector<int16_t> output;
  output.reserve(size_);
  const std::size_t start = size_ == data_.size() ? write_index_ : 0;
  for (std::size_t index = 0; index < size_; ++index) {
    output.push_back(data_[(start + index) % data_.size()]);
  }
  return output;
}

int16_t equal_mix_s16(int16_t left, int16_t right) {
  const int32_t sum = static_cast<int32_t>(left) + static_cast<int32_t>(right);
  return static_cast<int16_t>(sum / 2);
}

}  // namespace soundjie

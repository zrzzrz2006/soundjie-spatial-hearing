#include "tensor_stats.h"

#include <stdexcept>

TensorStats compute_tensor_stats(const int8_t* data, std::size_t size) {
  if (data == nullptr || size == 0) {
    throw std::invalid_argument("tensor must be nonempty");
  }
  TensorStats result = {
      static_cast<int>(data[0]),
      static_cast<int>(data[0]),
      0,
  };
  for (std::size_t index = 0; index < size; ++index) {
    const int value = static_cast<int>(data[index]);
    if (value < result.minimum) result.minimum = value;
    if (value > result.maximum) result.maximum = value;
    if (value != 0) ++result.nonzero;
  }
  return result;
}

bool tensors_equal(
    const int8_t* left,
    const int8_t* right,
    std::size_t size,
    std::size_t* first_mismatch) {
  if (first_mismatch == nullptr) {
    throw std::invalid_argument("first_mismatch is required");
  }
  if ((left == nullptr || right == nullptr) && size != 0) {
    throw std::invalid_argument("tensor pointer is null");
  }
  for (std::size_t index = 0; index < size; ++index) {
    if (left[index] != right[index]) {
      *first_mismatch = index;
      return false;
    }
  }
  *first_mismatch = size;
  return true;
}

void fill_deterministic_pattern(int8_t* data, std::size_t size, uint32_t seed) {
  if (data == nullptr && size != 0) {
    throw std::invalid_argument("tensor pointer is null");
  }
  uint32_t state = seed == 0 ? 0x6d2b79f5u : seed;
  for (std::size_t index = 0; index < size; ++index) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    data[index] = static_cast<int8_t>(static_cast<int>(state & 0xffu) - 128);
  }
}

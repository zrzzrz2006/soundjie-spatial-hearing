#ifndef MZ7100_TENSOR_STATS_H
#define MZ7100_TENSOR_STATS_H

#include <cstddef>
#include <cstdint>

struct TensorStats {
  int minimum;
  int maximum;
  std::size_t nonzero;
};

TensorStats compute_tensor_stats(const int8_t* data, std::size_t size);

bool tensors_equal(
    const int8_t* left,
    const int8_t* right,
    std::size_t size,
    std::size_t* first_mismatch);

void fill_deterministic_pattern(int8_t* data, std::size_t size, uint32_t seed);

#endif

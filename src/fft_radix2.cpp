#include "fft_radix2.h"

#include <cmath>

namespace soundjie {

std::array<std::complex<double>, 257> fft_radix2_512(
    const std::array<double, 512>& input) {
  const double pi = 3.141592653589793238462643383279502884;
  std::array<std::complex<double>, 512> values;
  for (std::size_t index = 0; index < values.size(); ++index)
    values[index] = std::complex<double>(input[index], 0.0);

  for (std::size_t index = 1, reversed = 0; index < values.size(); ++index) {
    std::size_t bit = values.size() >> 1;
    while (reversed & bit) {
      reversed ^= bit;
      bit >>= 1;
    }
    reversed ^= bit;
    if (index < reversed) std::swap(values[index], values[reversed]);
  }

  for (std::size_t length = 2; length <= values.size(); length <<= 1) {
    const double angle = -2.0 * pi / static_cast<double>(length);
    const std::complex<double> step(std::cos(angle), std::sin(angle));
    for (std::size_t begin = 0; begin < values.size(); begin += length) {
      std::complex<double> weight(1.0, 0.0);
      for (std::size_t offset = 0; offset < length / 2; ++offset) {
        const std::complex<double> even = values[begin + offset];
        const std::complex<double> odd = values[begin + offset + length / 2] * weight;
        values[begin + offset] = even + odd;
        values[begin + offset + length / 2] = even - odd;
        weight *= step;
      }
    }
  }
  std::array<std::complex<double>, 257> output;
  std::copy(values.begin(), values.begin() + output.size(), output.begin());
  return output;
}

}  // namespace soundjie

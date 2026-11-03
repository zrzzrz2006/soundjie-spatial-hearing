#ifndef SOUNDJIE_FFT_RADIX2_H_
#define SOUNDJIE_FFT_RADIX2_H_

#include <array>
#include <complex>

namespace soundjie {

std::array<std::complex<double>, 257> fft_radix2_512(
    const std::array<double, 512>& input);

}  // namespace soundjie
#endif

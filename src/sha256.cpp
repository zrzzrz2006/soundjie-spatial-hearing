#include "sha256.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace soundjie {
namespace {

const uint32_t kRound[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

uint32_t rotate_right(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32u - bits));
}

bool same_file_state(const struct stat& left, const struct stat& right) {
  return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
         left.st_size == right.st_size && left.st_mtime == right.st_mtime &&
         left.st_ctime == right.st_ctime;
}

}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
             0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u},
      buffer_{}, buffered_(0), total_bytes_(0), finalized_(false), digest_() {}

void Sha256::transform(const uint8_t block[64]) {
  uint32_t words[64] = {};
  for (std::size_t index = 0; index < 16; ++index) {
    const std::size_t offset = index * 4;
    words[index] = (static_cast<uint32_t>(block[offset]) << 24) |
                   (static_cast<uint32_t>(block[offset + 1]) << 16) |
                   (static_cast<uint32_t>(block[offset + 2]) << 8) |
                   static_cast<uint32_t>(block[offset + 3]);
  }
  for (std::size_t index = 16; index < 64; ++index) {
    const uint32_t s0 = rotate_right(words[index - 15], 7) ^
                        rotate_right(words[index - 15], 18) ^
                        (words[index - 15] >> 3);
    const uint32_t s1 = rotate_right(words[index - 2], 17) ^
                        rotate_right(words[index - 2], 19) ^
                        (words[index - 2] >> 10);
    words[index] = words[index - 16] + s0 + words[index - 7] + s1;
  }

  uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (std::size_t index = 0; index < 64; ++index) {
    const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
    const uint32_t choose = (e & f) ^ ((~e) & g);
    const uint32_t temporary1 = h + sum1 + choose + kRound[index] + words[index];
    const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void Sha256::update(const void* data, std::size_t size) {
  if (finalized_) throw std::logic_error("SHA-256 already finalized");
  if (size != 0 && data == nullptr) throw std::invalid_argument("null SHA-256 input");
  if (total_bytes_ > UINT64_MAX / 8u || size > UINT64_MAX / 8u - total_bytes_)
    throw std::overflow_error("SHA-256 input too large");
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  total_bytes_ += size;
  while (size != 0) {
    const std::size_t count = std::min<std::size_t>(size, 64 - buffered_);
    std::memcpy(buffer_ + buffered_, bytes, count);
    buffered_ += count;
    bytes += count;
    size -= count;
    if (buffered_ == 64) {
      transform(buffer_);
      buffered_ = 0;
    }
  }
}

std::string Sha256::final() {
  if (finalized_) return digest_;
  const uint64_t bit_length = total_bytes_ * 8u;
  buffer_[buffered_++] = 0x80u;
  if (buffered_ > 56) {
    std::memset(buffer_ + buffered_, 0, 64 - buffered_);
    transform(buffer_);
    buffered_ = 0;
  }
  std::memset(buffer_ + buffered_, 0, 56 - buffered_);
  for (std::size_t index = 0; index < 8; ++index)
    buffer_[63 - index] = static_cast<uint8_t>(bit_length >> (index * 8));
  transform(buffer_);

  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < 8; ++index) output << std::setw(8) << state_[index];
  digest_ = output.str();
  finalized_ = true;
  return digest_;
}

std::string sha256_file(const std::string& path) {
  struct stat before {};
  if (stat(path.c_str(), &before) != 0 || !S_ISREG(before.st_mode) || before.st_size < 0)
    throw std::runtime_error("cannot stat SHA-256 input file");
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open SHA-256 input file");

  Sha256 digest;
  std::vector<char> buffer(32768);
  uint64_t remaining = static_cast<uint64_t>(before.st_size);
  while (remaining != 0) {
    const std::streamsize requested = static_cast<std::streamsize>(
        std::min<uint64_t>(remaining, buffer.size()));
    stream.read(buffer.data(), requested);
    if (stream.gcount() != requested) throw std::runtime_error("short SHA-256 file read");
    digest.update(buffer.data(), static_cast<std::size_t>(requested));
    remaining -= static_cast<uint64_t>(requested);
  }
  char extra = 0;
  if (stream.read(&extra, 1)) throw std::runtime_error("long SHA-256 file read");
  if (stream.bad()) throw std::runtime_error("SHA-256 file I/O error");

  struct stat after {};
  if (stat(path.c_str(), &after) != 0 || !same_file_state(before, after))
    throw std::runtime_error("SHA-256 input file changed while reading");
  return digest.final();
}

}  // namespace soundjie

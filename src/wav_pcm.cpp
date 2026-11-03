#include "wav_pcm.h"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace soundjie {
namespace {

uint16_t u16(const std::vector<uint8_t>& data, uint64_t offset) {
  if (offset > data.size() || data.size() - static_cast<std::size_t>(offset) < 2)
    throw std::runtime_error("truncated uint16");
  return static_cast<uint16_t>(data[offset]) |
         static_cast<uint16_t>(data[offset + 1]) << 8;
}

uint32_t u32(const std::vector<uint8_t>& data, uint64_t offset) {
  if (offset > data.size() || data.size() - static_cast<std::size_t>(offset) < 4)
    throw std::runtime_error("truncated uint32");
  return static_cast<uint32_t>(data[offset]) |
         static_cast<uint32_t>(data[offset + 1]) << 8 |
         static_cast<uint32_t>(data[offset + 2]) << 16 |
         static_cast<uint32_t>(data[offset + 3]) << 24;
}

bool fourcc(const std::vector<uint8_t>& data, uint64_t offset, const char* value) {
  return offset <= data.size() && data.size() - static_cast<std::size_t>(offset) >= 4 &&
         data[offset] == static_cast<uint8_t>(value[0]) &&
         data[offset + 1] == static_cast<uint8_t>(value[1]) &&
         data[offset + 2] == static_cast<uint8_t>(value[2]) &&
         data[offset + 3] == static_cast<uint8_t>(value[3]);
}

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open WAV");
  stream.seekg(0, std::ios::end);
  const std::streamoff length = stream.tellg();
  if (length < 0) throw std::runtime_error("cannot size WAV");
  stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<std::size_t>(length));
  if (!data.empty()) stream.read(reinterpret_cast<char*>(&data[0]), length);
  if (!stream && length != 0) throw std::runtime_error("cannot read WAV");
  return data;
}

}  // namespace

PcmWindow read_pcm_window(const std::string& path, uint64_t start_sample,
                          uint32_t window_samples) {
  if (window_samples == 0) throw std::invalid_argument("window_samples must be positive");
  const std::vector<uint8_t> bytes = read_file(path);
  if (bytes.size() < 12 || !fourcc(bytes, 0, "RIFF") || !fourcc(bytes, 8, "WAVE"))
    throw std::runtime_error("not a RIFF/WAVE file");
  const uint64_t end = 8ULL + u32(bytes, 4);
  if (end < 12 || end > bytes.size()) throw std::runtime_error("truncated RIFF payload");

  bool have_fmt = false, have_data = false;
  uint16_t format = 0, channels = 0, block_align = 0, bits = 0;
  uint32_t rate = 0, byte_rate = 0;
  uint64_t data_offset = 0, data_bytes = 0;
  for (uint64_t offset = 12; offset < end;) {
    if (end - offset < 8) throw std::runtime_error("truncated chunk header");
    const uint64_t size = u32(bytes, offset + 4);
    const uint64_t payload = offset + 8;
    if (size > end - payload) throw std::runtime_error("truncated chunk payload");
    const uint64_t payload_end = payload + size;
    if ((size & 1ULL) && payload_end == end)
      throw std::runtime_error("missing chunk padding");
    const uint64_t next = payload_end + (size & 1ULL);
    if (fourcc(bytes, offset, "fmt ") && !have_fmt) {
      if (size < 16) throw std::runtime_error("short fmt chunk");
      format = u16(bytes, payload);
      channels = u16(bytes, payload + 2);
      rate = u32(bytes, payload + 4);
      byte_rate = u32(bytes, payload + 8);
      block_align = u16(bytes, payload + 12);
      bits = u16(bytes, payload + 14);
      have_fmt = true;
    } else if (fourcc(bytes, offset, "data") && !have_data) {
      data_offset = payload;
      data_bytes = size;
      have_data = true;
    }
    offset = next;
  }
  if (!have_fmt || !have_data) throw std::runtime_error("missing fmt or data chunk");
  if (format != 1 || (channels != 1 && channels != 2) || rate != 16000 || bits != 16)
    throw std::runtime_error("unsupported PCM WAV format");
  const uint16_t expected_align = static_cast<uint16_t>(channels * 2);
  if (block_align != expected_align || byte_rate != rate * expected_align)
    throw std::runtime_error("invalid PCM alignment");
  if (data_bytes % block_align) throw std::runtime_error("partial PCM frame");
  const uint64_t total = data_bytes / block_align;
  if (start_sample > total || window_samples > total - start_sample)
    throw std::runtime_error("requested WAV window is incomplete");

  PcmWindow result;
  result.metadata = WavMetadata{channels, rate, bits, total, data_offset, data_bytes};
  result.samples.resize(window_samples);
  uint64_t offset = data_offset + start_sample * block_align;
  for (uint32_t index = 0; index < window_samples; ++index, offset += block_align) {
    const int16_t left = static_cast<int16_t>(u16(bytes, offset));
    if (channels == 1) {
      result.samples[index] = static_cast<double>(left) / 32768.0;
    } else {
      const int16_t right = static_cast<int16_t>(u16(bytes, offset + 2));
      const int32_t sum = static_cast<int32_t>(left) + static_cast<int32_t>(right);
      result.samples[index] = static_cast<double>(sum) / 65536.0;
    }
  }
  return result;
}

}  // namespace soundjie

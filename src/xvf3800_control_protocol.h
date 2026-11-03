#ifndef SOUNDJIE_XVF3800_CONTROL_PROTOCOL_H_
#define SOUNDJIE_XVF3800_CONTROL_PROTOCOL_H_

#include <cstdint>
#include <string>
#include <vector>

namespace soundjie {

constexpr uint16_t kXvfVendorId = 0x2886;
constexpr uint16_t kXvfProductId = 0x001a;
constexpr uint8_t kXvfVendorDeviceIn = 0xc0;
constexpr uint8_t kXvfControlRequest = 0;
constexpr uint8_t kXvfControlSuccess = 0;
constexpr uint8_t kXvfControlRetry = 64;
constexpr uint16_t kXvfDoaValue = 0x0092;
constexpr uint16_t kXvfDoaIndex = 0x0014;
constexpr uint16_t kXvfAecAzimuthValue = 0x00cb;
constexpr uint16_t kXvfAecAzimuthIndex = 0x0021;

struct XvfDoaReading {
  int status;
  int raw_angle_deg;
  bool speech_detected;
  bool valid;
  bool retryable;
  std::string reason;
  std::string raw_hex;
};

struct XvfAecAzimuthReading {
  int status;
  bool valid;
  bool retryable;
  std::string reason;
  std::string raw_hex;
  std::vector<double> radians;
  std::vector<double> degrees;
};

std::vector<uint8_t> parse_hex_bytes(const std::string& text);
std::string bytes_to_hex(const std::vector<uint8_t>& bytes);
XvfDoaReading parse_xvf3800_doa_response(const std::vector<uint8_t>& response);
XvfAecAzimuthReading parse_xvf3800_aec_azimuth_response(
    const std::vector<uint8_t>& response);
std::string xvf_doa_json(const XvfDoaReading& reading, uint64_t monotonic_ms);
std::string xvf_aec_json(const XvfAecAzimuthReading& reading,
                         uint64_t monotonic_ms);

}  // namespace soundjie

#endif

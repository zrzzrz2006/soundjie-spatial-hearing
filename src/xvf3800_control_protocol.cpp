#include "xvf3800_control_protocol.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace soundjie {
namespace {

int hex_digit(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

double read_float_le(const uint8_t* source) {
  const uint32_t bits = static_cast<uint32_t>(source[0]) |
                        (static_cast<uint32_t>(source[1]) << 8) |
                        (static_cast<uint32_t>(source[2]) << 16) |
                        (static_cast<uint32_t>(source[3]) << 24);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return static_cast<double>(value);
}

std::string json_number_or_null(bool valid, double value) {
  if (!valid) return "null";
  std::ostringstream stream;
  stream << std::setprecision(12) << value;
  return stream.str();
}

}  // namespace

std::vector<uint8_t> parse_hex_bytes(const std::string& text) {
  if (text.size() % 2 != 0) throw std::invalid_argument("odd_hex_length");
  std::vector<uint8_t> result;
  result.reserve(text.size() / 2);
  for (std::size_t i = 0; i < text.size(); i += 2) {
    const int high = hex_digit(text[i]);
    const int low = hex_digit(text[i + 1]);
    if (high < 0 || low < 0) throw std::invalid_argument("invalid_hex_digit");
    result.push_back(static_cast<uint8_t>((high << 4) | low));
  }
  return result;
}

std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (uint8_t value : bytes) stream << std::setw(2) << static_cast<int>(value);
  return stream.str();
}

XvfDoaReading parse_xvf3800_doa_response(
    const std::vector<uint8_t>& response) {
  XvfDoaReading result{-1, -1, false, false, false, "wrong_length",
                       bytes_to_hex(response)};
  if (response.size() != 5) return result;
  result.status = response[0];
  result.retryable = result.status == kXvfControlRetry;
  if (result.status != kXvfControlSuccess) {
    result.reason = result.retryable ? "retry" : "status_" + std::to_string(result.status);
    return result;
  }
  result.raw_angle_deg = static_cast<int>(response[1]) |
                         (static_cast<int>(response[2]) << 8);
  const uint16_t speech = static_cast<uint16_t>(response[3]) |
                          (static_cast<uint16_t>(response[4]) << 8);
  result.speech_detected = speech != 0;
  if (result.raw_angle_deg < 0 || result.raw_angle_deg > 359) {
    result.reason = "angle_out_of_range";
    return result;
  }
  if (!result.speech_detected) {
    result.reason = "no_speech";
    return result;
  }
  result.valid = true;
  result.reason = "ok";
  return result;
}

XvfAecAzimuthReading parse_xvf3800_aec_azimuth_response(
    const std::vector<uint8_t>& response) {
  XvfAecAzimuthReading result{-1, false, false, "wrong_length",
                              bytes_to_hex(response), {}, {}};
  if (response.size() != 17) return result;
  result.status = response[0];
  result.retryable = result.status == kXvfControlRetry;
  if (result.status != kXvfControlSuccess) {
    result.reason = result.retryable ? "retry" : "status_" + std::to_string(result.status);
    return result;
  }
  constexpr double kPi = 3.14159265358979323846;
  for (std::size_t i = 0; i < 4; ++i) {
    const double radians = read_float_le(&response[1 + i * 4]);
    if (!std::isfinite(radians)) {
      result.radians.clear();
      result.degrees.clear();
      result.reason = "nonfinite_angle";
      return result;
    }
    result.radians.push_back(radians);
    result.degrees.push_back(radians * 180.0 / kPi);
  }
  result.valid = true;
  result.reason = "ok";
  return result;
}

std::string xvf_doa_json(const XvfDoaReading& reading,
                         uint64_t monotonic_ms) {
  std::ostringstream stream;
  stream << "{\"monotonic_ms\":" << monotonic_ms
         << ",\"protocol\":\"DOA_VALUE\""
         << ",\"raw_hex\":\"" << reading.raw_hex << "\""
         << ",\"status\":" << reading.status
         << ",\"raw_angle_deg\":"
         << (reading.raw_angle_deg >= 0 ? std::to_string(reading.raw_angle_deg) : "null")
         << ",\"angle_deg\":"
         << (reading.valid ? std::to_string(reading.raw_angle_deg) : "null")
         << ",\"speech_detected\":"
         << (reading.speech_detected ? "true" : "false")
         << ",\"valid\":" << (reading.valid ? "true" : "false")
         << ",\"retryable\":" << (reading.retryable ? "true" : "false")
         << ",\"reason\":\"" << reading.reason << "\"}";
  return stream.str();
}

std::string xvf_aec_json(const XvfAecAzimuthReading& reading,
                         uint64_t monotonic_ms) {
  std::ostringstream stream;
  stream << "{\"monotonic_ms\":" << monotonic_ms
         << ",\"protocol\":\"AEC_AZIMUTH_VALUES\""
         << ",\"raw_hex\":\"" << reading.raw_hex << "\""
         << ",\"status\":" << reading.status << ",\"radians\":[";
  for (std::size_t i = 0; i < reading.radians.size(); ++i) {
    if (i) stream << ',';
    stream << json_number_or_null(reading.valid, reading.radians[i]);
  }
  stream << "],\"degrees\":[";
  for (std::size_t i = 0; i < reading.degrees.size(); ++i) {
    if (i) stream << ',';
    stream << json_number_or_null(reading.valid, reading.degrees[i]);
  }
  stream << "],\"valid\":" << (reading.valid ? "true" : "false")
         << ",\"retryable\":" << (reading.retryable ? "true" : "false")
         << ",\"reason\":\"" << reading.reason << "\"}";
  return stream.str();
}

}  // namespace soundjie

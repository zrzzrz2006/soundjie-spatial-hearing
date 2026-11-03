#include "xvf3800_control_protocol.h"
#include "bounded_jsonl.h"

#include <dirent.h>
#include <fcntl.h>
#include <linux/usbdevice_fs.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string decode_doa_hex;
  std::string decode_aec_hex;
  std::string fixture_sequence;
  std::string jsonl;
  std::string usb_node;
  std::string sysfs_root = "/sys/bus/usb/devices";
  std::string dev_root = "/dev/bus/usb";
  uint32_t interval_ms = 100;
  uint32_t count = 0;
  std::size_t log_max_bytes = 0;
  bool read_aec = false;
};

uint32_t parse_u32(const std::string& value, const char* name) {
  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
  if (errno || !end || *end || parsed > 0xffffffffUL) {
    throw std::invalid_argument(std::string("invalid_") + name);
  }
  return static_cast<uint32_t>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (++i >= argc) throw std::invalid_argument(std::string("missing_") + name);
      return argv[i];
    };
    if (arg == "--decode-doa-hex") options.decode_doa_hex = next("decode_doa_hex");
    else if (arg == "--decode-aec-hex") options.decode_aec_hex = next("decode_aec_hex");
    else if (arg == "--fixture-sequence") options.fixture_sequence = next("fixture_sequence");
    else if (arg == "--jsonl") options.jsonl = next("jsonl");
    else if (arg == "--log-max-bytes") options.log_max_bytes = soundjie::jsonl_limit(next("log_max_bytes"));
    else if (arg == "--usb-node") options.usb_node = next("usb_node");
    else if (arg == "--sysfs-root") options.sysfs_root = next("sysfs_root");
    else if (arg == "--dev-root") options.dev_root = next("dev_root");
    else if (arg == "--interval-ms") options.interval_ms = parse_u32(next("interval_ms"), "interval_ms");
    else if (arg == "--count") options.count = parse_u32(next("count"), "count");
    else if (arg == "--read-aec-azimuths") options.read_aec = true;
    else throw std::invalid_argument("unknown_argument:" + arg);
  }
  return options;
}

std::string trim(const std::string& value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return std::string();
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string read_text(const std::string& path) {
  std::ifstream stream(path.c_str());
  if (!stream) return std::string();
  std::string value;
  stream >> value;
  return value;
}

std::string discover_usb_node(const Options& options) {
  DIR* directory = opendir(options.sysfs_root.c_str());
  if (!directory) throw std::runtime_error("cannot_open_usb_sysfs");
  std::string match;
  for (dirent* entry = readdir(directory); entry; entry = readdir(directory)) {
    if (entry->d_name[0] == '.') continue;
    const std::string base = options.sysfs_root + "/" + entry->d_name;
    if (trim(read_text(base + "/idVendor")) != "2886" ||
        trim(read_text(base + "/idProduct")) != "001a") {
      continue;
    }
    const std::string bus = trim(read_text(base + "/busnum"));
    const std::string dev = trim(read_text(base + "/devnum"));
    if (bus.empty() || dev.empty()) continue;
    char node[512];
    std::snprintf(node, sizeof(node), "%s/%03u/%03u", options.dev_root.c_str(),
                  parse_u32(bus, "busnum"), parse_u32(dev, "devnum"));
    if (!match.empty()) {
      closedir(directory);
      throw std::runtime_error("multiple_xvf3800_devices");
    }
    match = node;
  }
  closedir(directory);
  if (match.empty()) throw std::runtime_error("xvf3800_not_found");
  return match;
}

uint64_t monotonic_ms() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    throw std::runtime_error("clock_gettime_failed");
  }
  return static_cast<uint64_t>(value.tv_sec) * 1000ULL +
         static_cast<uint64_t>(value.tv_nsec) / 1000000ULL;
}

std::vector<uint8_t> usb_control_read(int fd, uint16_t value, uint16_t index,
                                      uint16_t length) {
  std::vector<uint8_t> response(length, 0);
  usbdevfs_ctrltransfer transfer{};
  transfer.bRequestType = soundjie::kXvfVendorDeviceIn;
  transfer.bRequest = soundjie::kXvfControlRequest;
  transfer.wValue = value;
  transfer.wIndex = index;
  transfer.wLength = length;
  transfer.timeout = 1000;
  transfer.data = response.data();
  const int received = ioctl(fd, USBDEVFS_CONTROL, &transfer);
  if (received < 0) throw std::runtime_error("usb_control_read_failed:" + std::to_string(errno));
  response.resize(static_cast<std::size_t>(received));
  return response;
}

int run_fixture(const Options& options) {
  if (options.jsonl.empty()) throw std::invalid_argument("fixture_requires_jsonl");
  std::ifstream input(options.fixture_sequence.c_str());
  if (!input) throw std::runtime_error("cannot_open_fixture_sequence");
  soundjie::BoundedJsonl output(options.jsonl,options.log_max_bytes);
  uint64_t timestamp = 0;
    std::string line;
    while (std::getline(input, line)) {
      line = trim(line);
      if (line.empty()) continue;
      const auto reading = soundjie::parse_xvf3800_doa_response(
          soundjie::parse_hex_bytes(line));
      output.write_line(soundjie::xvf_doa_json(reading, timestamp));
      timestamp += options.interval_ms;
    }
  return 0;
}

int run_usb(const Options& options) {
  if (options.jsonl.empty()) throw std::invalid_argument("usb_mode_requires_jsonl");
  const std::string node = options.usb_node.empty() ? discover_usb_node(options) : options.usb_node;
  const int fd = open(node.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("cannot_open_usb_node:" + std::to_string(errno));
  try {
    soundjie::BoundedJsonl output(options.jsonl,options.log_max_bytes);
    uint32_t emitted = 0;
    while (options.count == 0 || emitted < options.count) {
      if (options.read_aec) {
        const auto bytes = usb_control_read(fd, soundjie::kXvfAecAzimuthValue,
                                            soundjie::kXvfAecAzimuthIndex, 17);
        const auto reading = soundjie::parse_xvf3800_aec_azimuth_response(bytes);
        output.write_line(soundjie::xvf_aec_json(reading, monotonic_ms()));
      } else {
        soundjie::XvfDoaReading reading;
        int attempts = 0;
        do {
          const auto bytes = usb_control_read(fd, soundjie::kXvfDoaValue,
                                              soundjie::kXvfDoaIndex, 5);
          reading = soundjie::parse_xvf3800_doa_response(bytes);
          if (reading.retryable) usleep(10000);
        } while (reading.retryable && ++attempts < 100);
        output.write_line(soundjie::xvf_doa_json(reading, monotonic_ms()));
      }
      ++emitted;
      if (options.count == 0 || emitted < options.count) usleep(options.interval_ms * 1000U);
    }
    close(fd);
  } catch (...) {
    close(fd);
    throw;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (!options.decode_doa_hex.empty()) {
      const auto reading = soundjie::parse_xvf3800_doa_response(
          soundjie::parse_hex_bytes(options.decode_doa_hex));
      std::cout << soundjie::xvf_doa_json(reading, 0) << '\n';
      return 0;
    }
    if (!options.decode_aec_hex.empty()) {
      const auto reading = soundjie::parse_xvf3800_aec_azimuth_response(
          soundjie::parse_hex_bytes(options.decode_aec_hex));
      std::cout << soundjie::xvf_aec_json(reading, 0) << '\n';
      return 0;
    }
    if (!options.fixture_sequence.empty()) return run_fixture(options);
    return run_usb(options);
  } catch (const std::runtime_error& error) {
    std::cerr << error.what() << '\n';
    return std::string(error.what()) == "output already exists" ? 3 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 64;
  }
}

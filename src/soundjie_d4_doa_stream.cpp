#include "soundjie_d4_doa_stream.h"
#include "xvf3800_control_protocol.h"

#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace soundjie {
namespace {
const std::size_t kReadLimit = 65536;
const std::size_t kRowLimit = 4096;

struct File {
  explicit File(int descriptor) : fd(descriptor) {}
  ~File() { if (fd >= 0) close(fd); }
  int fd;
};

// Not a general JSON parser: the local producer is a versioned canonical
// serializer. Round-trip equality rejects duplicate/contradictory fields,
// truncation, trailing junk and forged decoded values. Derive validity ONLY
// from the USB raw bytes, using the same decoder as the producer.
bool decode(const std::string& row, uint64_t* timestamp, XvfDoaReading* reading) {
  const std::string prefix = "{\"monotonic_ms\":";
  const std::string marker = ",\"protocol\":\"DOA_VALUE\",\"raw_hex\":\"";
  if (row.compare(0, prefix.size(), prefix) != 0) return false;
  std::size_t pos = prefix.size();
  const std::size_t start = pos;
  uint64_t value = 0;
  while (pos < row.size() && row[pos] >= '0' && row[pos] <= '9') {
    const uint64_t digit = row[pos++] - '0';
    if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) return false;
    value = value * 10 + digit;
  }
  if (pos == start || row.compare(pos, marker.size(), marker) != 0) return false;
  pos += marker.size();
  const auto end = row.find('"', pos);
  if (end == std::string::npos) return false;
  try {
    *reading = parse_xvf3800_doa_response(parse_hex_bytes(row.substr(pos, end - pos)));
  } catch (const std::exception&) {
    return false;
  }
  if (xvf_doa_json(*reading, value) != row) return false;
  *timestamp = value;
  return true;
}
}  // namespace

D4DoaStream::D4DoaStream(const std::string& path, double alpha, uint64_t stale_ms)
    : path_(path), alpha_(alpha), stale_ms_(stale_ms), filter_(alpha, stale_ms) {}

void D4DoaStream::invalidate(const std::string& reason, uint64_t now_ms) {
  // After an outage, the first new direction must not blend with an old one.
  filter_ = CircularDoaFilter(alpha_, stale_ms_);
  filter_.update(0, false, reason, now_ms);
}

void D4DoaStream::consume(const std::string& row, uint64_t now_ms) {
  XvfDoaReading reading;
  uint64_t timestamp = 0;
  if (!decode(row, &timestamp, &reading)) {
    invalidate("malformed_row", now_ms);
    return;
  }
  if (timestamp > now_ms) {
    invalidate("future_sample", now_ms);
    return;
  }
  if (have_timestamp_ && timestamp < last_timestamp_) {
    invalidate("out_of_order", now_ms);
    return;
  }
  if (have_timestamp_ && timestamp == last_timestamp_) {
    if (row != last_row_) invalidate("conflicting_timestamp", now_ms);
    return;  // Re-reading a duplicate never renews age or updates the EMA.
  }
  have_timestamp_ = true;
  last_timestamp_ = timestamp;
  last_row_ = row;
  if (now_ms - timestamp > stale_ms_) {
    invalidate("stale", now_ms);
    return;
  }
  if (!reading.valid) {
    invalidate(reading.reason, now_ms);
    return;
  }
  if (!filter_.snapshot(timestamp).valid) filter_ = CircularDoaFilter(alpha_, stale_ms_);
  filter_.update(reading.raw_angle_deg, true, "ok", timestamp);
}

SoundjieD4DoaState D4DoaStream::poll(const std::function<uint64_t()>& clock) {
  // Nonblocking + regular-file check avoids hanging on a substituted FIFO.
  File file(open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW));
  struct stat info;
  if (file.fd < 0 || fstat(file.fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0) {
    const uint64_t now_ms = clock();
    invalidate("source_unavailable", now_ms);
    identity_known_ = false;
    pending_.clear();
    return filter_.snapshot(now_ms);
  }
  const uint64_t size = static_cast<uint64_t>(info.st_size);
  if (!identity_known_ || device_ != static_cast<uint64_t>(info.st_dev) ||
      inode_ != static_cast<uint64_t>(info.st_ino) || size < offset_) {
    invalidate("awaiting_sample", 0);
    identity_known_ = true;
    device_ = info.st_dev;
    inode_ = info.st_ino;
    offset_ = 0;
    pending_.clear();
    discard_until_newline_ = false;
    // Preserve the high-water timestamp across rotation; a replay is not new.
  }
  if (size - offset_ > kReadLimit) {
    offset_ = size - kReadLimit;
    pending_.clear();
    discard_until_newline_ = true;
    invalidate("backlog_gap", 0);
  }
  const std::size_t wanted = static_cast<std::size_t>(size - offset_);
  std::vector<char> bytes(wanted);
  std::size_t received = 0;
  while (received < wanted) {
    const ssize_t count = pread(file.fd, bytes.data() + received, wanted - received,
                                static_cast<off_t>(offset_ + received));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      const uint64_t now_ms = clock();
      invalidate("source_read_error", now_ms);
      pending_.clear();
      identity_known_ = false;
      return filter_.snapshot(now_ms);
    }
    received += static_cast<std::size_t>(count);
  }
  offset_ += received;
  const uint64_t now_ms = clock();
  for (char ch : bytes) {
    if (discard_until_newline_) {
      if (ch == '\n') discard_until_newline_ = false;
      continue;
    }
    if (ch == '\n') {
      consume(pending_, now_ms);
      pending_.clear();
    } else if (pending_.size() == kRowLimit) {
      pending_.clear();
      discard_until_newline_ = true;
      invalidate("oversized_row", now_ms);
    } else {
      pending_ += ch;
    }
  }
  return filter_.snapshot(now_ms);
}

}  // namespace soundjie

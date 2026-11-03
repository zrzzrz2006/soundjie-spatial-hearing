#ifndef SOUNDJIE_D4_DOA_STREAM_H
#define SOUNDJIE_D4_DOA_STREAM_H

#include "soundjie_d4_state.h"
#include <cstdint>
#include <functional>
#include <string>

namespace soundjie {

// Read-only consumer of xvf3800_doa_cli's canonical DOA_VALUE JSONL.
// Call from the presentation clock, independently of blocking inference.
// Each poll reads at most 64 KiB and retains at most one 4 KiB partial row.
class D4DoaStream {
 public:
  D4DoaStream(const std::string& path, double alpha, uint64_t stale_ms);
  // Clock is sampled AFTER the bounded read, avoiding producer/read races.
  SoundjieD4DoaState poll(const std::function<uint64_t()>& clock);

 private:
  void invalidate(const std::string& reason, uint64_t now_ms);
  void consume(const std::string& row, uint64_t now_ms);
  std::string path_;
  double alpha_;
  uint64_t stale_ms_;
  CircularDoaFilter filter_;
  bool identity_known_ = false;
  uint64_t device_ = 0, inode_ = 0, offset_ = 0;
  std::string pending_;
  bool discard_until_newline_ = false;
  bool have_timestamp_ = false;
  uint64_t last_timestamp_ = 0;
  std::string last_row_;
};

}  // namespace soundjie
#endif

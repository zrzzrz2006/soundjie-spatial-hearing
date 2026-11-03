#include "pcm_ring_buffer.h"
#include "soundjie_d4_state.h"
#include "soundjie_class_contract.h"
#include "known_clip_matcher.h"
#include "traffic_fine_model.h"
#include "bounded_jsonl.h"
#include <memory>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
volatile std::sig_atomic_t stop_requested = 0;
void request_stop(int) { stop_requested = 1; }

struct Options {
  std::string ring_test;
  std::string mix_test;
  std::string input;
  std::string output_dir;
  std::string current;
  std::string classifier_command;
  std::string tv_reference, tv_cues, tv_bank, traffic_model;
  double dialogue_horn_start=0.0, dialogue_horn_end=0.0;
  std::size_t capacity = 0;
  std::size_t window_samples = 16000;
  std::size_t hop_samples = 12000;
  unsigned cadence_ms = 750;
  unsigned classifier_timeout_ms = 5000;
  unsigned retain_windows = 0;
  std::size_t log_max_bytes = 0;
  unsigned mock_delay_ms = 0;
  int mock_class_id = -1;
  std::string class_contract = "soundjie-legacy8-v1";
  std::vector<double> alert_thresholds{.5,.5,.5,.5};
};

std::vector<std::string> split(const std::string& value, char delimiter) {
  std::vector<std::string> output;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, delimiter)) output.push_back(item);
  return output;
}

std::size_t positive_size(const std::string& value, const std::string& name) {
  std::size_t used = 0;
  const unsigned long long parsed = std::stoull(value, &used, 10);
  if (used != value.size() || parsed == 0) throw std::invalid_argument(name + " must be positive");
  return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string key(argv[index]);
    if (++index >= argc) throw std::invalid_argument("missing value for " + key);
    const std::string value(argv[index]);
    if (key == "--ring-test") options.ring_test = value;
    else if (key == "--mix-test") options.mix_test = value;
    else if (key == "--capacity") options.capacity = positive_size(value, key);
    else if (key == "--fixture-pcm" || key == "--input") options.input = value;
    else if (key == "--output-dir") options.output_dir = value;
    else if (key == "--log-max-bytes") options.log_max_bytes = soundjie::jsonl_limit(value);
    else if (key == "--current") options.current = value;
    else if (key == "--window-samples") options.window_samples = positive_size(value, key);
    else if (key == "--hop-samples") options.hop_samples = positive_size(value, key);
    else if (key == "--cadence-ms") options.cadence_ms = static_cast<unsigned>(positive_size(value, key));
    else if (key == "--mock-delay-ms") options.mock_delay_ms = static_cast<unsigned>(std::stoul(value));
    else if (key == "--mock-class-id") options.mock_class_id = std::stoi(value);
    else if (key == "--classifier-command") options.classifier_command = value;
    else if (key == "--class-contract") { soundjie::siren5_contract(value); options.class_contract=value; }
    else if (key == "--alert-thresholds") {
      options.alert_thresholds.clear();
      for(const auto& part:split(value,',')){
        std::size_t used=0;const double t=std::stod(part,&used);
        if(used!=part.size()||!std::isfinite(t)||t<0||t>1)throw std::invalid_argument("invalid alert threshold");
        options.alert_thresholds.push_back(t);
      }
      if(options.alert_thresholds.size()!=4&&options.alert_thresholds.size()!=5)throw std::invalid_argument("four or five class thresholds required");
    }
    else if (key == "--tv-reference") options.tv_reference = value;
    else if (key == "--tv-cues") options.tv_cues = value;
    else if (key == "--tv-bank") options.tv_bank = value;
    else if (key == "--tv-dialogue-horn-start" || key == "--tv-dialogue-horn-end") {
      std::size_t used=0;const double seconds=std::stod(value,&used);
      if(used!=value.size()||!std::isfinite(seconds)||seconds<0||seconds>120)throw std::invalid_argument("invalid dialogue horn interval");
      if(key=="--tv-dialogue-horn-start")options.dialogue_horn_start=seconds;else options.dialogue_horn_end=seconds;
    }
    else if (key == "--traffic-model") options.traffic_model = value;
    else if (key == "--retain-windows") {
      if (value != "0") {
        const auto count = positive_size(value, key);
        if (count > 240) throw std::invalid_argument("retain-windows exceeds 240");
        options.retain_windows = static_cast<unsigned>(count);
      }
    }
    else if (key == "--classifier-timeout-ms") {
      const auto timeout = positive_size(value, key);
      if (timeout > 60000) throw std::invalid_argument("classifier timeout exceeds 60000 ms");
      options.classifier_timeout_ms = static_cast<unsigned>(timeout);
    }
    else if (key == "--expected-class") { (void)value; }
    else throw std::invalid_argument("unknown option: " + key);
  }
  return options;
}

void require_absolute(const std::string& path, const std::string& name) {
  if (path.empty() || path[0] != '/') throw std::invalid_argument(name + " must be absolute");
}

void write_all(int fd, const std::string& payload) {
  const char* cursor = payload.data();
  std::size_t remaining = payload.size();
  while (remaining != 0) {
    const ssize_t count = write(fd, cursor, remaining);
    if (count < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write failed");
    }
    cursor += count;
    remaining -= static_cast<std::size_t>(count);
  }
}

void publish_atomic(const std::string& path, const std::string& payload) {
  const std::string temporary = path + ".tmp." + std::to_string(getpid());
  const int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) throw std::runtime_error("cannot create atomic snapshot temporary");
  try {
    write_all(fd, payload + "\n");
    if (fsync(fd) != 0) throw std::runtime_error("snapshot fsync failed");
    if (close(fd) != 0) throw std::runtime_error("snapshot close failed");
    if (rename(temporary.c_str(), path.c_str()) != 0) throw std::runtime_error("snapshot rename failed");
  } catch (...) {
    close(fd);
    unlink(temporary.c_str());
    throw;
  }
}

int16_t decode_s16_le(const char* bytes) {
  const uint16_t bits = static_cast<unsigned char>(bytes[0]) |
                        (static_cast<uint16_t>(static_cast<unsigned char>(bytes[1])) << 8);
  return static_cast<int16_t>(bits);
}

struct Classification {
  int class_id = -1;
  double confidence = 0.0;
  double inference_ms = 0.0;
  std::vector<double> scores;
};

Classification run_classifier(const std::string& command,
                              const std::vector<int16_t>& window,
                              const std::string& output_dir,
                              uint64_t sequence, unsigned timeout_ms,
                              unsigned physical_outputs,
                              const std::function<void()>& receive_audio = {}) {
  const std::string input_path = output_dir + "/window-" + std::to_string(sequence) + ".s16";
  const int input_fd = open(input_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (input_fd < 0) throw std::runtime_error("cannot create classifier input");
  try {
    write_all(input_fd, std::string(reinterpret_cast<const char*>(window.data()),
                                    window.size() * sizeof(int16_t)));
    if (close(input_fd) != 0) throw std::runtime_error("classifier input close failed");
  } catch (...) {
    close(input_fd);
    unlink(input_path.c_str());
    throw;
  }
  int output_pipe[2];
  if (pipe(output_pipe) != 0) {
    unlink(input_path.c_str());
    throw std::runtime_error("classifier pipe failed");
  }
  const pid_t parent = getpid();
  const pid_t child = fork();
  if (child == 0) {
    if (setpgid(0, 0) != 0 || prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent) _exit(126);
    const int null_fd = open("/dev/null", O_RDONLY);
    if (null_fd < 0 || dup2(null_fd, STDIN_FILENO) < 0) _exit(126);
    if (null_fd != STDIN_FILENO) close(null_fd);
    close(output_pipe[0]);
    dup2(output_pipe[1], STDOUT_FILENO);
    close(output_pipe[1]);
    execl(command.c_str(), command.c_str(), input_path.c_str(), static_cast<char*>(nullptr));
    _exit(127);
  }
  close(output_pipe[1]);
  if (child < 0) {
    close(output_pipe[0]);
    unlink(input_path.c_str());
    throw std::runtime_error("classifier fork failed");
  }
  // Child also sets the group before exec. Parent closes the fork scheduling race.
  setpgid(child, child);
  std::string output;
  char buffer[256];
  int status = 0;
  bool reaped = false, eof = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  try {
    const int flags = fcntl(output_pipe[0], F_GETFL);
    if (flags < 0 || fcntl(output_pipe[0], F_SETFL, flags | O_NONBLOCK) != 0)
      throw std::runtime_error("classifier nonblocking pipe failed");
    while (!reaped || !eof) {
      if (stop_requested) throw std::runtime_error("classifier interrupted");
      if (receive_audio) receive_audio();
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("classifier deadline exceeded");
      if (!eof) {
        const ssize_t count = read(output_pipe[0], buffer, sizeof(buffer));
        if (count > 0) {
          if (output.size() + static_cast<std::size_t>(count) > 1024)
            throw std::runtime_error("classifier output limit exceeded");
          output.append(buffer, static_cast<std::size_t>(count));
        } else if (count == 0) eof = true;
        else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
          throw std::runtime_error("classifier pipe read failed");
        if (count > 0) continue;
      }
      if (!reaped) {
        const pid_t waited = waitpid(child, &status, WNOHANG);
        if (waited == child) reaped = true;
        else if (waited < 0 && errno != EINTR) throw std::runtime_error("classifier wait failed");
      }
      if (!reaped || !eof) {
        if (eof) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        else { struct pollfd descriptor{output_pipe[0], POLLIN | POLLHUP, 0}; poll(&descriptor, 1, 5); }
      }
    }
  } catch (...) {
    kill(-child, SIGKILL);
    if (!reaped) {
      kill(child, SIGKILL);
      // Never block forever on an uninterruptible kernel/DPU wait.
      const auto reap_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
      while (waitpid(child, &status, WNOHANG) == 0 && std::chrono::steady_clock::now() < reap_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    close(output_pipe[0]);
    unlink(input_path.c_str());
    throw;
  }
  close(output_pipe[0]);
  unlink(input_path.c_str());
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("classifier command failed");
  Classification result;
  std::istringstream parser(output);
  if (!(parser >> result.class_id >> result.confidence >> result.inference_ms) ||
      result.class_id < 0 || static_cast<unsigned>(result.class_id) >= physical_outputs || result.confidence < 0.0 ||
      result.confidence > 1.0 || result.inference_ms < 0.0 ||
      !std::isfinite(result.confidence) || !std::isfinite(result.inference_ms))
    throw std::runtime_error("invalid classifier output");
  parser >> std::ws;
  if (!parser.eof()) {
    double sum = 0.0;
    for (unsigned i = 0; i < physical_outputs; ++i) {
      double score;
      if (!(parser >> score) || !std::isfinite(score) || score < 0.0 || score > 1.0)
        throw std::runtime_error("invalid classifier score vector");
      result.scores.push_back(score);
      sum += score;
    }
    std::string extra;
    if ((parser >> extra) || std::abs(sum - 1.0) > 1e-6 ||
        std::abs(result.scores[result.class_id] - result.confidence) > 1e-6 ||
        *std::max_element(result.scores.begin(), result.scores.end()) > result.confidence + 1e-6)
      throw std::runtime_error("inconsistent classifier score vector");
  }
  if(physical_outputs!=8 && result.scores.size()!=physical_outputs)throw std::runtime_error("new contract requires full score vector");
  return result;
}

const char* class_label(int id) {
  static const char* const labels[] = {
      "火灾或烟雾报警器", "应急车辆警笛", "汽车鸣笛", "玻璃破碎",
      "敲门或门铃", "紧急人声", "普通说话声", "背景或未知声音"};
  return id < 0 || id >= 8 ? "不可用" : labels[id];
}

std::string retain_window(const Options& options, uint64_t sequence,
                          const std::vector<int16_t>& window) {
  if (sequence > options.retain_windows) return "";
  if (window.size() > 16000) throw std::runtime_error("retained window exceeds one second");
  const std::string name = "capture-" + std::to_string(sequence) + ".s16";
  const std::string path = options.output_dir + "/" + name;
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) throw std::runtime_error("cannot create retained PCM");
  std::string bytes;
  bytes.reserve(window.size() * 2);
  for (int16_t value : window) {
    const uint16_t bits = static_cast<uint16_t>(value);
    bytes.push_back(static_cast<char>(bits & 255));
    bytes.push_back(static_cast<char>(bits >> 8));
  }
  try { write_all(fd, bytes); }
  catch (...) { close(fd); throw; }
  if (close(fd) != 0) throw std::runtime_error("retained PCM close failed");
  return name;
}

soundjie::SoundjieD4Snapshot make_snapshot(uint64_t sequence, int class_id,
                                           double confidence,
                                           bool usb_ok, double inference_ms,
                                           double end_to_end_ms,
                                           const std::vector<int16_t>& window) {
  soundjie::SoundjieD4Snapshot snapshot;
  snapshot.sequence = sequence;
  snapshot.timestamp_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  snapshot.class_id = class_id;
  snapshot.label_zh = class_label(class_id);
  snapshot.confidence = class_id < 0 ? 0.0 : confidence;
  if (class_id >= 0) {
    snapshot.top3.push_back({class_id, snapshot.label_zh, confidence});
  }
  snapshot.doa.valid = false;
  snapshot.doa.reason = "not_sampled";
  snapshot.audio = soundjie::measure_s16_audio(window, soundjie::SoundjieD4RuntimeConfig());
  snapshot.usb_ok = usb_ok;
  snapshot.dpu_ok = class_id >= 0;
  snapshot.framebuffer_ok = false;
  snapshot.inference_ms = inference_ms;
  snapshot.end_to_end_ms = end_to_end_ms;
  return snapshot;
}

int run_fixture(const Options& options) {
  const bool five=soundjie::siren5_contract(options.class_contract);
  const bool alert4=soundjie::alert4_contract(options.class_contract);
  const bool alert5=soundjie::alert5_contract(options.class_contract);
  if((alert4&&options.alert_thresholds.size()!=4)||(alert5&&options.alert_thresholds.size()!=5))throw std::invalid_argument("class threshold count differs from contract");
  if(options.dialogue_horn_end<options.dialogue_horn_start)throw std::invalid_argument("reversed dialogue horn interval");
  const unsigned physical_outputs=soundjie::physical_class_count(options.class_contract);
  if((five||alert4||alert5) && options.mock_class_id>=0)throw std::invalid_argument("compact class mode requires physical probability protocol");
  require_absolute(options.output_dir, "output-dir");
  require_absolute(options.current, "current");
  if (options.input != "-") require_absolute(options.input, "input");
  if (options.mock_class_id < 0 && options.classifier_command.empty())
    throw std::invalid_argument("classifier-command is required outside explicit mock mode");
  std::unique_ptr<soundjie::KnownClipMatcher> tv;
  std::unique_ptr<soundjie::KnownClipBank> bank;
  std::unique_ptr<soundjie::TrafficFineModel> traffic;
  if(!options.tv_bank.empty()) {
    if(!options.tv_reference.empty()||!options.tv_cues.empty()||options.traffic_model.empty()||options.window_samples!=16000)
      throw std::invalid_argument("TV bank requires traffic model and 16000 samples; single reference options are exclusive");
    bank.reset(new soundjie::KnownClipBank(soundjie::read_clip_bank(options.tv_bank)));
    traffic.reset(new soundjie::TrafficFineModel(options.traffic_model));
  } else if (!options.tv_reference.empty() || !options.tv_cues.empty() || !options.traffic_model.empty()) {
    if(options.tv_reference.empty() || options.tv_cues.empty() || options.traffic_model.empty() || options.window_samples!=16000)
      throw std::invalid_argument("TV mode requires reference, cues, traffic model and 16000-sample window");
    const auto reference=soundjie::read_clip_reference(options.tv_reference);
    const auto cues=soundjie::read_clip_cues(options.tv_cues,((reference.size()-1)*80+512)/16000.0);
    tv.reset(new soundjie::KnownClipMatcher(reference,cues));
    traffic.reset(new soundjie::TrafficFineModel(options.traffic_model));
  }
  const bool tv_enabled=bool(tv)||bool(bank);
  if (mkdir(options.output_dir.c_str(), 0755) != 0) {
    if (errno == EEXIST) throw std::runtime_error("output directory already exists");
    throw std::runtime_error("cannot create output directory");
  }
  soundjie::BoundedJsonl timeline(options.output_dir + "/timeline.jsonl",options.log_max_bytes);
  std::istream* input = &std::cin;
  std::ifstream input_file;
  if (options.input != "-") {
    input_file.open(options.input.c_str(), std::ios::binary);
    if (!input_file) throw std::runtime_error("cannot open PCM input");
    input = &input_file;
  }
  soundjie::PcmRingBuffer ring(options.window_samples);
  std::size_t since_inference = 0;
  uint64_t sequence = 0;
  uint64_t dpu_calls = 0;
  uint64_t deadline_misses = 0;
  int predicted = -1;
  double confidence = 0.0;
  double reported_inference_ms = 0.0;
  char frame[4];
  const bool live = options.input == "-";
  bool live_eof = false;
  std::size_t frame_bytes = 0;
  uint64_t received_samples = 0, previous_window_end = 0, skipped_samples = 0;
  uint64_t received_ms = 0;
  uint64_t capture_epoch = 0;
  struct RestoreFlags {
    int flags = -1;
    ~RestoreFlags() { if (flags >= 0) fcntl(STDIN_FILENO, F_SETFL, flags); }
  } restore_flags;
  if (live) {
    restore_flags.flags = fcntl(STDIN_FILENO, F_GETFL);
    if (restore_flags.flags < 0 || fcntl(STDIN_FILENO, F_SETFL, restore_flags.flags | O_NONBLOCK) != 0)
      throw std::runtime_error("cannot set nonblocking live input");
  }
  // One event loop, not a second inference thread. Keep draining while the
  // classifier runs; only the latest bounded window survives slow inference.
  const auto receive_audio = [&]() {
    const auto now=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    if(tv_enabled && received_ms && now>received_ms && now-received_ms>1000) {
      // A wall-clock gap is not contiguous audio. Invalidate both histories,
      // including an in-flight classifier's older capture epoch. Keep partial
      // PCM bytes only to preserve stereo frame alignment when the FIFO resumes.
      ring.clear();if(tv)tv->reset();if(bank)bank->reset();since_inference=0;received_ms=0;
      previous_window_end=received_samples;++capture_epoch;
      auto gap=make_snapshot(++sequence,-1,0,false,0,0,{});
      gap.class_contract=options.class_contract;
      gap.tv_enabled=tv_enabled;gap.audio.audio_state="unavailable";gap.doa.reason="capture_gap";
      const auto json=soundjie::soundjie_d4_snapshot_json(gap);
      timeline.write_line(json);publish_atomic(options.current,json);
    }
    char block[4096];
    for (unsigned batch = 0; batch < 64 && !live_eof; ++batch) {
      const ssize_t count = read(STDIN_FILENO, block, sizeof(block));
      if (count == 0) {
        live_eof = true;
        if (frame_bytes != 0) throw std::runtime_error("truncated stereo PCM frame");
        return;
      }
      if (count < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return;
        if (errno == EINTR) continue;
        throw std::runtime_error("live PCM read failed");
      }
      const uint64_t samples_before = received_samples;
      for (ssize_t index = 0; index < count; ++index) {
        frame[frame_bytes++] = block[index];
        if (frame_bytes == 4) {
          const int16_t sample=soundjie::equal_mix_s16(decode_s16_le(frame), decode_s16_le(frame + 2));
          ring.push(sample); if(tv)tv->push(sample);if(bank)bank->push(sample);
          frame_bytes = 0;
          ++received_samples;
          ++since_inference;
        }
      }
      if (received_samples != samples_before)
        received_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }
  };
  try {
  while (true) {
    if (stop_requested) throw std::runtime_error("runtime interrupted");
    if (live) {
      receive_audio();
      if (!ring.full() || (dpu_calls != 0 && since_inference < options.hop_samples)) {
        if (live_eof) break;
        struct pollfd descriptor{STDIN_FILENO, POLLIN | POLLHUP, 0};
        poll(&descriptor, 1, 20);
        continue;
      }
      const uint64_t expected_step = dpu_calls == 0 ? options.window_samples : options.hop_samples;
      const uint64_t step = received_samples - previous_window_end;
      if (step > expected_step) skipped_samples += step - expected_step;
      previous_window_end = received_samples;
    } else {
      if (!input->read(frame, sizeof(frame))) break;
      const int16_t sample=soundjie::equal_mix_s16(decode_s16_le(frame), decode_s16_le(frame + 2));
      ring.push(sample); if(tv)tv->push(sample);if(bank)bank->push(sample);
      ++received_samples;
      ++since_inference;
      if (!ring.full() || (dpu_calls != 0 && since_inference < options.hop_samples)) continue;
    }
    since_inference = 0;
    const auto window = ring.latest();
    const uint64_t window_received_ms = received_ms;
    const uint64_t window_end_sample = received_samples;
    const uint64_t window_capture_epoch = capture_epoch;
    const std::string capture_file = retain_window(options, sequence + 1, window);
    std::vector<double> class_scores;
    int raw_class_id=-1;
    bool rejected=false;
    const auto start = std::chrono::steady_clock::now();
    // Freeze both extra branches at the same capture boundary as the DPU window.
    // More PCM may arrive while the asynchronous DPU child is executing.
    auto clip = bank ? bank->match() : tv ? tv->match() : soundjie::ClipMatch();
    if(tv&&clip.matched){clip.clip_id="ambulance_escort_45_75";clip.scene_profile="ambulance_traffic";}
    const auto clip_done=std::chrono::steady_clock::now();
    const auto fine = traffic&&(!bank||clip.scene_profile=="ambulance_traffic") ? traffic->predict(window) : std::array<float,3>{{0,0,0}};
    const auto fine_done=std::chrono::steady_clock::now();
    if (options.mock_class_id >= 0) {
      if (options.mock_delay_ms) std::this_thread::sleep_for(
          std::chrono::milliseconds(options.mock_delay_ms));
      predicted = options.mock_class_id;
      confidence = 0.8;
      reported_inference_ms = options.mock_delay_ms;
    } else {
      const Classification result = run_classifier(
          options.classifier_command, window, options.output_dir, sequence + 1,
          options.classifier_timeout_ms, physical_outputs, live ? std::function<void()>(receive_audio) : std::function<void()>());
      predicted = result.class_id;
      confidence = result.confidence;
      reported_inference_ms = result.inference_ms;
      class_scores = result.scores;
      if(five){
        class_scores=soundjie::merge_siren_probabilities(class_scores);
        raw_class_id=std::max_element(class_scores.begin(),class_scores.end())-class_scores.begin();
        predicted=soundjie::siren_decision(class_scores);rejected=predicted!=raw_class_id;
        confidence=class_scores[predicted];
      }
      if(alert4){
        raw_class_id=predicted;
        predicted=soundjie::alert4_decision(class_scores,options.alert_thresholds);
        rejected=predicted!=raw_class_id;confidence=class_scores[predicted];
      }
      if(alert5){
        raw_class_id=predicted;
        predicted=soundjie::alert5_decision(class_scores,options.alert_thresholds);
        rejected=predicted!=raw_class_id;confidence=class_scores[predicted];
      }
    }
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (elapsed > options.cadence_ms) ++deadline_misses;
    ++dpu_calls;
    if(window_capture_epoch!=capture_epoch)continue;
    auto snapshot = make_snapshot(
        ++sequence, predicted, confidence, true, reported_inference_ms, elapsed,
        window);
    snapshot.capture_file = capture_file;
    snapshot.clip_match_ms=std::chrono::duration<double,std::milli>(clip_done-start).count();
    snapshot.fine_model_ms=std::chrono::duration<double,std::milli>(fine_done-clip_done).count();
    snapshot.classifier_ms=elapsed-snapshot.clip_match_ms-snapshot.fine_model_ms;
    snapshot.window_end_sample = window_end_sample;
    snapshot.class_scores = class_scores;
    snapshot.class_contract=options.class_contract;
    snapshot.classification_rejected=rejected;
    snapshot.raw_class_id=raw_class_id<0?predicted:raw_class_id;
    snapshot.label_zh=soundjie::class_name(predicted,options.class_contract);
    snapshot.tv_enabled=tv_enabled;snapshot.tv_matched=clip.matched;
    snapshot.tv_clip_id=clip.matched?clip.clip_id:"";snapshot.tv_scene_profile=clip.matched?clip.scene_profile:"";
    snapshot.scene_horn_start=options.dialogue_horn_start;snapshot.scene_horn_end=options.dialogue_horn_end;
    snapshot.tv_position_seconds=clip.position_seconds;snapshot.tv_score=clip.score;
    snapshot.caption_text=clip.text;snapshot.caption_speaker=clip.speaker;snapshot.caption_cue=clip.cue_id;
    // This small classifier is validated for the enrolled clip only. Keep raw
    // scores for diagnostics, and publish events only inside its tested scope.
    snapshot.fine_valid=clip.matched && clip.scene_profile=="ambulance_traffic" && traffic && snapshot.audio.audio_state!="silence";
    snapshot.ambulance_score=fine[0];snapshot.road_score=fine[1];snapshot.speech_score=fine[2];
    if (class_scores.size() == soundjie::logical_class_count(options.class_contract)) {
      std::vector<int> indices;
      for(unsigned i=0;i<class_scores.size();++i)indices.push_back(i);
      std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
        return class_scores[a] > class_scores[b];
      });
      snapshot.top2_margin = class_scores[indices[0]] - class_scores[indices[1]];
      snapshot.top3.clear();
      for (unsigned rank = 0; rank < 3; ++rank) {
        const int id = indices[rank];
        snapshot.top3.push_back({id, soundjie::class_name(id,options.class_contract), class_scores[id]});
      }
    }
    if (live) {
      // Timestamp refers to this window's reception, NOT inference completion.
      // This is host reception age, not a measured USB/physical capture age.
      snapshot.end_to_end_ms = static_cast<double>(snapshot.timestamp_ms - window_received_ms);
      snapshot.timestamp_ms = window_received_ms;
    }
    const std::string json = soundjie::soundjie_d4_snapshot_json(snapshot);
    timeline.write_line(json);
    publish_atomic(options.current, json);
  }
  if (!live && !input->eof()) throw std::runtime_error("PCM read failed");
  if (!live && input->gcount() != 0) throw std::runtime_error("truncated stereo PCM frame");
  } catch (...) {
    auto failed = make_snapshot(++sequence, -1, 0.0, false, 0.0, 0.0, {});
    failed.class_contract=options.class_contract;
    const auto json = soundjie::soundjie_d4_snapshot_json(failed);
    publish_atomic(options.current, json);
    // The original failure may be this very writer. A diagnostic failure must
    // not suppress the unavailable snapshot or replace the original error.
    try { timeline.write_line(json); } catch (...) {}
    throw;
  }
  auto final_snapshot = make_snapshot(
      ++sequence, -1, 0.0, false, 0.0, 0.0, {});
  final_snapshot.class_contract=options.class_contract;
  const std::string final_json = soundjie::soundjie_d4_snapshot_json(final_snapshot);
  publish_atomic(options.current, final_json);
  timeline.write_line(final_json);
  std::cout << "{\"dpu_calls\":" << dpu_calls
            << ",\"max_concurrent_inference\":1"
            << ",\"deadline_misses\":" << deadline_misses
            << ",\"live_received_samples\":" << received_samples
            << ",\"live_skipped_samples\":" << skipped_samples
            << ",\"usb_ok\":false,\"predicted_class\":" << predicted << "}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    struct sigaction action{};
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGINT, &action, nullptr);
    const Options options = parse_options(argc, argv);
    if (!options.ring_test.empty()) {
      soundjie::PcmRingBuffer ring(options.capacity);
      for (const auto& value : split(options.ring_test, ',')) ring.push(static_cast<int16_t>(std::stoi(value)));
      const auto latest = ring.latest();
      std::cout << "{\"latest\":[";
      for (std::size_t i = 0; i < latest.size(); ++i) { if (i) std::cout << ','; std::cout << latest[i]; }
      std::cout << "]}\n";
      return 0;
    }
    if (!options.mix_test.empty()) {
      const auto values = split(options.mix_test, ',');
      if (values.size() % 2 != 0) throw std::invalid_argument("mix test needs stereo pairs");
      std::cout << "{\"mono\":[";
      for (std::size_t i = 0; i < values.size(); i += 2) {
        if (i) std::cout << ',';
        std::cout << soundjie::equal_mix_s16(static_cast<int16_t>(std::stoi(values[i])),
                                             static_cast<int16_t>(std::stoi(values[i + 1])));
      }
      std::cout << "]}\n";
      return 0;
    }
    return run_fixture(options);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}

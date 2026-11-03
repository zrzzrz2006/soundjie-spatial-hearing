// One mono S16LE window -> real frontend -> linked DPU model -> runtime protocol.
#include <n2cube.h>
#include "logmel_frontend.h"
#include "sha256.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#ifndef SOUNDJIE_LINKED_MODEL_SHA256
#error "Build must bind the linked DPU model hash"
#endif
#ifndef SOUNDJIE_FRONTEND_SHA256
#error "Build must bind the frontend configuration hash"
#endif
#ifndef SOUNDJIE_KERNEL_NAME
#define SOUNDJIE_KERNEL_NAME "soundjie_d3"
#endif
#ifndef SOUNDJIE_MODEL_BASENAME
#define SOUNDJIE_MODEL_BASENAME "dpu_soundjie_d3.elf"
#endif

#ifndef SOUNDJIE_PHYSICAL_OUTPUTS
#define SOUNDJIE_PHYSICAL_OUTPUTS 8
#endif
static_assert(SOUNDJIE_PHYSICAL_OUTPUTS==8 || SOUNDJIE_PHYSICAL_OUTPUTS==6 || SOUNDJIE_PHYSICAL_OUTPUTS==5,"unsupported output contract");
#ifndef SOUNDJIE_STORAGE_OUTPUTS
#define SOUNDJIE_STORAGE_OUTPUTS SOUNDJIE_PHYSICAL_OUTPUTS
#endif
static_assert(SOUNDJIE_STORAGE_OUTPUTS==SOUNDJIE_PHYSICAL_OUTPUTS ||
              (SOUNDJIE_PHYSICAL_OUTPUTS==5 && SOUNDJIE_STORAGE_OUTPUTS==6),
              "unsupported DPU output storage layout");

namespace {
struct Handles {
  bool opened = false;
  DPUKernel* kernel = nullptr;
  DPUTask* task = nullptr;
  ~Handles() {
    if (task) dpuDestroyTask(task);
    if (kernel) dpuDestroyKernel(kernel);
    if (opened) dpuClose();
  }
};

// n2cube may write diagnostics to stdout. Reserve stdout for the numeric protocol.
class DiagnosticOutput {
 public:
  DiagnosticOutput() : saved_(dup(STDOUT_FILENO)) {
    if (saved_ < 0) throw std::runtime_error("cannot preserve protocol output");
    std::cout.flush();
    std::fflush(stdout);
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
      close(saved_);
      throw std::runtime_error("cannot redirect DPU diagnostics");
    }
  }
  ~DiagnosticOutput() {
    std::cout.flush();
    std::fflush(stdout);
    dup2(saved_, STDOUT_FILENO);
    close(saved_);
  }
 private:
  int saved_;
};

std::vector<double> read_window(const std::string& path) {
  struct stat info {};
  if (path.empty() || path[0] != '/' || stat(path.c_str(), &info) != 0 ||
      !S_ISREG(info.st_mode) || info.st_size != 32000)
    throw std::runtime_error("PCM must be an absolute regular file of exactly 32000 bytes");
  std::ifstream stream(path.c_str(), std::ios::binary);
  std::vector<unsigned char> bytes(32000);
  stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (stream.gcount() != 32000) throw std::runtime_error("PCM read incomplete");
  char extra;
  if (stream.read(&extra, 1) || stream.bad()) throw std::runtime_error("PCM changed while reading");
  std::vector<double> samples(16000);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const unsigned bits = bytes[2 * i] | (static_cast<unsigned>(bytes[2 * i + 1]) << 8);
    const int signed_value = bits >= 32768 ? static_cast<int>(bits) - 65536 : static_cast<int>(bits);
    samples[i] = signed_value / 32768.0;
  }
  return samples;
}

struct Result { int id; double confidence; double inference_ms; std::vector<double> scores; };

Result classify(const std::string& input) {
  const char* base = std::getenv("SOUNDJIE_BUNDLE_DIR");
  if (!base || base[0] != '/') throw std::runtime_error("SOUNDJIE_BUNDLE_DIR must be absolute");
  const std::string root(base);
  const std::string model = root + "/model/" + SOUNDJIE_MODEL_BASENAME;
  const std::string config = root + "/config/logmel_frontend_v1.json";
  const auto samples = read_window(input);
  if (soundjie::sha256_file(model) != SOUNDJIE_LINKED_MODEL_SHA256)
    throw std::runtime_error("model does not match linked classifier");
  if (soundjie::sha256_file(config) != SOUNDJIE_FRONTEND_SHA256)
    throw std::runtime_error("frontend does not match classifier");
  const auto frontend = soundjie::load_frontend_config(config);
  const auto features = soundjie::compute_logmel(samples, frontend);
  if (features.size() != 6144) throw std::runtime_error("invalid frontend shape");

  Handles h;
  dpuSetExceptionMode(N2CUBE_EXCEPTION_MODE_RET_ERR_CODE);
  if (dpuOpen() != 0) throw std::runtime_error("dpuOpen failed");
  h.opened = true;
  h.kernel = dpuLoadKernel(SOUNDJIE_KERNEL_NAME);
  if (!h.kernel) throw std::runtime_error("dpuLoadKernel failed");
  h.task = dpuCreateTask(h.kernel, T_MODE_NORMAL);
  if (!h.task) throw std::runtime_error("dpuCreateTask failed");
  const int input_size = dpuGetInputTensorSize(h.task, "conv1");
  const int output_size = dpuGetOutputTensorSize(h.task, "fc8");
  if (input_size != 6144 || output_size != SOUNDJIE_STORAGE_OUTPUTS)
    throw std::runtime_error("invalid DPU tensor shape: input=" + std::to_string(input_size) +
                            " output=" + std::to_string(output_size) +
                            " expected_storage=" + std::to_string(SOUNDJIE_STORAGE_OUTPUTS));
  int8_t* address = dpuGetInputTensorAddress(h.task, "conv1");
  if (!address) throw std::runtime_error("null DPU input");
  const double input_scale = dpuGetInputTensorScale(h.task, "conv1");
  const double output_scale = dpuGetOutputTensorScale(h.task, "fc8");
  if (!std::isfinite(input_scale) || input_scale <= 0 ||
      !std::isfinite(output_scale) || output_scale <= 0)
    throw std::runtime_error("invalid DPU scale");
  const auto tensor = soundjie::quantize_int8(features, input_scale);
  std::memcpy(address, tensor.data(), tensor.size());
  const auto start = std::chrono::steady_clock::now();
  if (dpuRunTask(h.task) != 0) throw std::runtime_error("dpuRunTask failed");
  const double elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  const int8_t* output = dpuGetOutputTensorAddress(h.task, "fc8");
  if (!output || dpuGetOutputTensorSize(h.task, "fc8") != SOUNDJIE_STORAGE_OUTPUTS)
    throw std::runtime_error("invalid DPU output");
  // DNNC v2.05 may pad the five trained outputs to six storage lanes. The
  // package audits that lane's weights/bias; never treat it as a class.
  for (int i=SOUNDJIE_PHYSICAL_OUTPUTS; i<SOUNDJIE_STORAGE_OUTPUTS; ++i)
    if (output[i]!=0) throw std::runtime_error("nonzero DPU padding lane");
  if (std::all_of(output, output + SOUNDJIE_PHYSICAL_OUTPUTS, [](int8_t value) { return value == 0; }))
    throw std::runtime_error("all-zero DPU logits");
  const int id = static_cast<int>(std::max_element(output, output + SOUNDJIE_PHYSICAL_OUTPUTS) - output);
  double sum = 0;
  for (int i = 0; i < SOUNDJIE_PHYSICAL_OUTPUTS; ++i)
    sum += std::exp((static_cast<int>(output[i]) - output[id]) * output_scale);
  std::vector<double> scores(SOUNDJIE_PHYSICAL_OUTPUTS);
  for (int i = 0; i < SOUNDJIE_PHYSICAL_OUTPUTS; ++i)
    scores[i] = std::exp((static_cast<int>(output[i]) - output[id]) * output_scale) / sum;
  return Result{id, scores[id], elapsed, scores};
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: soundjie_d4_classifier /absolute/window.s16");
    Result result;
    {
      DiagnosticOutput diagnostics;
      result = classify(argv[1]);
    }
    std::cout << result.id << ' ' << std::setprecision(17) << result.confidence
              << ' ' << result.inference_ms;
    for (double score : result.scores) std::cout << ' ' << score;
    std::cout << '\n';
    return std::cout ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "D4_CLASSIFIER_FAIL: " << error.what() << '\n';
    return 2;
  }
}

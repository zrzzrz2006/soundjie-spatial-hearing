#include <n2cube.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "sha256.h"
#include "tensor_stats.h"

#ifndef SOUNDJIE_LINKED_MODEL_SHA256
#error "SOUNDJIE_LINKED_MODEL_SHA256 must identify the model linked into this runner"
#endif

namespace {
const std::size_t kInputBytes = 64u * 96u;

struct Options {
  std::string kernel, input_node, output_node, input_mode, input_file;
  std::string model_file, dump, report;
  int iterations = 0;
  bool describe_tensor = false;
};

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

int positive_int(const std::string& text) {
  std::size_t used = 0;
  long value = std::stol(text, &used, 10);
  if (used != text.size() || value <= 0 || value > std::numeric_limits<int>::max())
    throw std::invalid_argument("iterations must be positive");
  return static_cast<int>(value);
}

bool absolute_path(const std::string& path) {
  return !path.empty() && path[0] == '/';
}

std::string canonical_output_path(const std::string& path, unsigned depth = 0) {
  if (depth > 32) throw std::invalid_argument("too many output path symlinks");
  char resolved_path[PATH_MAX];
  if (realpath(path.c_str(), resolved_path)) return std::string(resolved_path);

  const std::string::size_type slash = path.rfind('/');
  const std::string parent = slash == 0 ? "/" : path.substr(0, slash);
  const std::string base = path.substr(slash + 1);
  if (base.empty()) throw std::invalid_argument("output path must name a file");

  struct stat link_status {};
  if (lstat(path.c_str(), &link_status) == 0 && S_ISLNK(link_status.st_mode)) {
    char target[PATH_MAX];
    const ssize_t length = readlink(path.c_str(), target, sizeof(target) - 1);
    if (length < 0 || static_cast<std::size_t>(length) >= sizeof(target) - 1)
      throw std::invalid_argument("cannot resolve output path symlink");
    target[length] = '\0';
    const std::string target_path = target[0] == '/'
        ? std::string(target) : parent + "/" + target;
    return canonical_output_path(target_path, depth + 1);
  }

  char resolved[PATH_MAX];
  if (!realpath(parent.c_str(), resolved))
    throw std::invalid_argument("output parent directory must exist");
  return std::string(resolved) + (std::string(resolved) == "/" ? "" : "/") + base;
}

bool same_output_file(const std::string& left, const std::string& right) {
  if (canonical_output_path(left) == canonical_output_path(right)) return true;
  struct stat left_status {};
  struct stat right_status {};
  if (stat(left.c_str(), &left_status) == 0 && stat(right.c_str(), &right_status) == 0)
    return left_status.st_dev == right_status.st_dev && left_status.st_ino == right_status.st_ino;
  return false;
}

Options options(int argc, char** argv) {
  Options out;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--describe-tensor") {
      out.describe_tensor = true;
      continue;
    }
    if (++i >= argc) throw std::invalid_argument("missing value for " + key);
    const std::string value(argv[i]);
    if (key == "--kernel") out.kernel = value;
    else if (key == "--input-node") out.input_node = value;
    else if (key == "--output-node") out.output_node = value;
    else if (key == "--input") out.input_mode = value;
    else if (key == "--input-file") out.input_file = value;
    else if (key == "--model-file") out.model_file = value;
    else if (key == "--iterations") out.iterations = positive_int(value);
    else if (key == "--dump") out.dump = value;
    else if (key == "--report") out.report = value;
    else throw std::invalid_argument("unknown option: " + key);
  }
  if (out.kernel.empty() || out.input_node.empty())
    throw std::invalid_argument("required option missing");
  if (out.describe_tensor) {
    if (out.output_node.empty() || !out.input_mode.empty() || !out.input_file.empty() ||
        !out.model_file.empty() || out.iterations != 0 || !out.dump.empty() || !out.report.empty())
      throw std::invalid_argument("describe mode is mutually exclusive with input, iteration and output options");
    return out;
  }
  if (out.output_node.empty() || out.iterations == 0 || out.dump.empty() || out.report.empty())
    throw std::invalid_argument("required option missing");
  if (!out.input_mode.empty() && !out.input_file.empty())
    throw std::invalid_argument("input file and --input are mutually exclusive");
  if (out.input_mode.empty() && out.input_file.empty())
    throw std::invalid_argument("input source required");
  if (!out.input_mode.empty()) {
    if (out.input_mode != "zero" && out.input_mode != "pattern")
      throw std::invalid_argument("input must be zero or pattern");
    if (!out.model_file.empty())
      throw std::invalid_argument("--model-file is only valid for file input");
  } else {
    if (!absolute_path(out.input_file)) throw std::invalid_argument("input file must be absolute");
    if (out.model_file.empty())
      throw std::invalid_argument("--model-file is required for file input");
    if (!absolute_path(out.model_file)) throw std::invalid_argument("model file must be absolute");
  }
  if (!absolute_path(out.dump) || !absolute_path(out.report))
    throw std::invalid_argument("dump and report must be absolute");
  if (same_output_file(out.dump, out.report))
    throw std::invalid_argument("dump and report paths must differ");
  return out;
}

std::vector<int8_t> read_input_file(const std::string& path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open input file");
  std::vector<int8_t> data(kInputBytes);
  stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
  if (stream.gcount() != static_cast<std::streamsize>(data.size()))
    throw std::runtime_error("input file must be exactly 6144 bytes");
  char extra = 0;
  if (stream.read(&extra, 1) || stream.bad())
    throw std::runtime_error("input file must be exactly 6144 bytes");
  return data;
}

void populate_input(int8_t* data, std::size_t size, const std::string& mode) {
  if (mode == "zero") std::memset(data, 0, size);
  else fill_deterministic_pattern(data, size, 0x534a4430u);
}

void copy_input(int8_t* destination, std::size_t size, const Options& opt,
                const std::vector<int8_t>& file_input) {
  if (!opt.input_file.empty()) {
    if (file_input.size() != size) throw std::runtime_error("validated input size changed");
    std::memcpy(destination, file_input.data(), size);
  } else {
    populate_input(destination, size, opt.input_mode);
  }
}

void write_dump(const std::string& path, const std::vector<int8_t>& data) {
  std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot open dump");
  stream.write(reinterpret_cast<const char*>(data.data()), data.size());
  stream.close();
  if (!stream) throw std::runtime_error("cannot write or close dump");
}

std::string hash_bytes(const std::vector<int8_t>& data) {
  soundjie::Sha256 digest;
  digest.update(data.data(), data.size());
  return digest.final();
}

std::string signed_values(const std::vector<int8_t>& data) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < data.size(); ++index) {
    if (index != 0) stream << ',';
    stream << static_cast<int>(data[index]);
  }
  return stream.str();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = options(argc, argv);
    const std::vector<int8_t> file_input = opt.input_file.empty()
        ? std::vector<int8_t>() : read_input_file(opt.input_file);
    const std::string input_sha = file_input.empty() ? "" : hash_bytes(file_input);
    const std::string model_sha = opt.model_file.empty()
        ? "" : soundjie::sha256_file(opt.model_file);
    if (!model_sha.empty() && model_sha != SOUNDJIE_LINKED_MODEL_SHA256)
      throw std::runtime_error("model file does not match linked model");

    Handles h;
    dpuSetExceptionMode(N2CUBE_EXCEPTION_MODE_RET_ERR_CODE);
    if (dpuOpen() != 0) throw std::runtime_error("dpuOpen failed");
    h.opened = true;
    h.kernel = dpuLoadKernel(opt.kernel.c_str());
    if (!h.kernel) throw std::runtime_error("dpuLoadKernel failed");
    h.task = dpuCreateTask(h.kernel, T_MODE_NORMAL);
    if (!h.task) throw std::runtime_error("dpuCreateTask failed");

    const int input_size = dpuGetInputTensorSize(h.task, opt.input_node.c_str());
    int8_t* input = dpuGetInputTensorAddress(h.task, opt.input_node.c_str());
    if (input_size != static_cast<int>(kInputBytes) || !input)
      throw std::runtime_error("invalid input tensor");
    float input_scale = 0.0f;
    float output_scale = 0.0f;
    int output_size = 0;
    if (opt.describe_tensor || !opt.input_file.empty()) {
      input_scale = dpuGetInputTensorScale(h.task, opt.input_node.c_str());
      if (!std::isfinite(input_scale) || input_scale <= 0.0f)
        throw std::runtime_error("invalid input tensor scale");
      output_size = dpuGetOutputTensorSize(h.task, opt.output_node.c_str());
      output_scale = dpuGetOutputTensorScale(h.task, opt.output_node.c_str());
      if (output_size != 8) throw std::runtime_error("invalid output tensor");
      if (!std::isfinite(output_scale) || output_scale <= 0.0f)
        throw std::runtime_error("invalid output tensor scale");
    }

    if (opt.describe_tensor) {
      std::cout << std::setprecision(std::numeric_limits<float>::max_digits10)
                << "DPU_TENSOR_STATUS=pass\n"
                << "KERNEL=" << opt.kernel << '\n'
                << "INPUT_NODE=" << opt.input_node << '\n'
                << "OUTPUT_NODE=" << opt.output_node << '\n'
                << "INPUT_SIZE=" << input_size << '\n'
                << "INPUT_SCALE=" << input_scale << '\n'
                << "OUTPUT_SIZE=" << output_size << '\n'
                << "OUTPUT_SCALE=" << output_scale << '\n'
                << "DPU_RUN_CALLS=0\n";
      return 0;
    }

    std::vector<int8_t> baseline;
    TensorStats stats{};
    std::size_t mismatches = 0;
    std::vector<long long> durations;
    for (int iteration = 1; iteration <= opt.iterations; ++iteration) {
      copy_input(input, static_cast<std::size_t>(input_size), opt, file_input);
      const auto start = std::chrono::steady_clock::now();
      if (dpuRunTask(h.task) != 0) throw std::runtime_error("dpuRunTask failed");
      durations.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - start).count());
      const int size = dpuGetOutputTensorSize(h.task, opt.output_node.c_str());
      int8_t* address = dpuGetOutputTensorAddress(h.task, opt.output_node.c_str());
      if (size != 8 || !address) throw std::runtime_error("invalid output tensor");
      if (iteration == 1) {
        baseline.assign(address, address + size);
        stats = compute_tensor_stats(address, static_cast<std::size_t>(size));
        if (stats.nonzero == 0) throw std::runtime_error("all-zero logits");
        write_dump(opt.dump, baseline);
      } else {
        std::size_t offset = 0;
        if (!tensors_equal(baseline.data(), address, baseline.size(), &offset)) ++mismatches;
      }
    }

    const long long total = std::accumulate(durations.begin(), durations.end(), 0LL);
    std::ofstream report(opt.report.c_str(), std::ios::trunc);
    if (!report) throw std::runtime_error("cannot open report");
    report << std::setprecision(std::numeric_limits<float>::max_digits10)
           << "DPU_SOUND_STATUS=" << (mismatches == 0 ? "pass" : "fail") << '\n'
           << "KERNEL=" << opt.kernel << '\n'
           << "INPUT_NODE=" << opt.input_node << '\n'
           << "OUTPUT_NODE=" << opt.output_node << '\n'
           << "INPUT_MODE=" << (opt.input_file.empty() ? opt.input_mode : "file") << '\n'
           << "INPUT_SIZE=" << input_size << '\n'
           << "OUTPUT_BYTES=" << baseline.size() << '\n'
           << "OUTPUT_NONZERO_BYTES=" << stats.nonzero << '\n'
           << "DPU_RUN_CALLS=" << opt.iterations << '\n'
           << "MISMATCH_ITERATIONS=" << mismatches << '\n'
           << "RUN_TOTAL_US=" << total << '\n';
    if (!opt.input_file.empty()) {
      report << "INPUT_SCALE=" << input_scale << '\n'
             << "OUTPUT_SCALE=" << output_scale << '\n'
             << "INPUT_SOURCE=file\n"
             << "INPUT_BYTES=" << file_input.size() << '\n'
             << "MODEL_SHA256=" << model_sha << '\n'
             << "INPUT_SHA256=" << input_sha << '\n'
             << "OUTPUT_SHA256=" << hash_bytes(baseline) << '\n'
             << "OUTPUT_VALUES=" << signed_values(baseline) << '\n';
    }
    report.close();
    if (!report) throw std::runtime_error("cannot write or close report");
    if (mismatches != 0) return 1;
    std::cout << "DPU_SOUND_STATUS=pass\nDPU_RUN_CALLS=" << opt.iterations << '\n';
    return 0;
  } catch (const std::invalid_argument& error) {
    std::cerr << "DPU_SOUND_STATUS=fail\nREASON=" << error.what() << '\n';
    return 64;
  } catch (const std::exception& error) {
    std::cerr << "DPU_SOUND_STATUS=fail\nREASON=" << error.what() << '\n';
    return 1;
  }
}

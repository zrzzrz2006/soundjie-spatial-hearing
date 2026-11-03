#include <n2cube.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor_stats.h"

namespace {

struct Options {
  std::string kernel;
  std::string input_node;
  std::vector<std::string> output_nodes;
  std::string input_mode;
  int iterations = 0;
  std::string dump_dir;
  std::string report_path;
};

struct DpuHandles {
  bool opened = false;
  DPUKernel* kernel = nullptr;
  DPUTask* task = nullptr;

  ~DpuHandles() {
    if (task != nullptr) dpuDestroyTask(task);
    if (kernel != nullptr) dpuDestroyKernel(kernel);
    if (opened) dpuClose();
  }
};

struct OutputRecord {
  std::string node;
  int height;
  int width;
  int channel;
  int size;
  float scale;
  TensorStats stats;
  std::vector<int8_t> baseline;
  std::string dump_path;
};

void usage(const char* program) {
  std::cerr
      << "usage: " << program
      << " --kernel NAME --input-node NAME --output-node NAME [--output-node NAME ...]"
      << " --input zero|pattern --iterations N --dump-dir ABS --report ABS\n";
}

int parse_positive_int(const std::string& text) {
  std::size_t consumed = 0;
  const long value = std::stol(text, &consumed, 10);
  if (consumed != text.size() || value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("iterations must be a positive integer");
  }
  return static_cast<int>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    if (arg == "--output-node") {
      if (++index >= argc) throw std::invalid_argument("missing output-node value");
      options.output_nodes.push_back(argv[index]);
      continue;
    }
    if (++index >= argc) throw std::invalid_argument("missing value for " + arg);
    const std::string value(argv[index]);
    if (arg == "--kernel") options.kernel = value;
    else if (arg == "--input-node") options.input_node = value;
    else if (arg == "--input") options.input_mode = value;
    else if (arg == "--iterations") options.iterations = parse_positive_int(value);
    else if (arg == "--dump-dir") options.dump_dir = value;
    else if (arg == "--report") options.report_path = value;
    else throw std::invalid_argument("unknown argument: " + arg);
  }

  if (options.kernel.empty() || options.input_node.empty() || options.output_nodes.empty() ||
      options.iterations <= 0 || options.dump_dir.empty() || options.report_path.empty()) {
    throw std::invalid_argument("required argument is missing");
  }
  if (options.input_mode != "zero" && options.input_mode != "pattern") {
    throw std::invalid_argument("input must be zero or pattern");
  }
  if (options.dump_dir[0] != '/' || options.report_path[0] != '/') {
    throw std::invalid_argument("dump-dir and report must be absolute");
  }
  std::string prefix = options.dump_dir;
  while (prefix.size() > 1 && prefix.back() == '/') prefix.pop_back();
  prefix += '/';
  if (options.report_path.compare(0, prefix.size(), prefix) != 0) {
    throw std::invalid_argument("report must be inside dump-dir");
  }
  options.dump_dir = prefix.substr(0, prefix.size() - 1);
  return options;
}

void write_binary(const std::string& path, const std::vector<int8_t>& data) {
  std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot open output dump: " + path);
  stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  if (!stream) throw std::runtime_error("cannot write output dump: " + path);
}

std::string safe_dump_name(const std::string& node) {
  std::string result;
  for (char value : node) {
    const bool safe = (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
                      (value >= '0' && value <= '9') || value == '_' || value == '-';
    if (!safe) throw std::invalid_argument("unsafe output node name");
    result.push_back(value);
  }
  return result + ".bin";
}

void populate_input(int8_t* input, std::size_t size, const std::string& mode) {
  if (mode == "zero") {
    std::memset(input, 0, size);
  } else {
    fill_deterministic_pattern(input, size, 0x12345678u);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    DpuHandles handles;
    dpuSetExceptionMode(N2CUBE_EXCEPTION_MODE_RET_ERR_CODE);
    if (dpuOpen() != 0) throw std::runtime_error("dpuOpen failed");
    handles.opened = true;
    handles.kernel = dpuLoadKernel(options.kernel.c_str());
    if (handles.kernel == nullptr) throw std::runtime_error("dpuLoadKernel failed");
    handles.task = dpuCreateTask(handles.kernel, T_MODE_NORMAL);
    if (handles.task == nullptr) throw std::runtime_error("dpuCreateTask failed");

    const int input_size = dpuGetInputTensorSize(handles.task, options.input_node.c_str());
    int8_t* const input = dpuGetInputTensorAddress(handles.task, options.input_node.c_str());
    if (input_size <= 0 || input == nullptr) throw std::runtime_error("invalid input tensor");

    const int input_height = dpuGetInputTensorHeight(handles.task, options.input_node.c_str());
    const int input_width = dpuGetInputTensorWidth(handles.task, options.input_node.c_str());
    const int input_channel = dpuGetInputTensorChannel(handles.task, options.input_node.c_str());
    const float input_scale = dpuGetInputTensorScale(handles.task, options.input_node.c_str());

    std::vector<OutputRecord> outputs;
    outputs.reserve(options.output_nodes.size());
    std::vector<long long> run_us;
    run_us.reserve(static_cast<std::size_t>(options.iterations));
    std::size_t mismatch_iterations = 0;
    std::size_t first_mismatch_iteration = 0;
    std::string first_mismatch_node;
    std::size_t first_mismatch_offset = 0;

    for (int iteration = 1; iteration <= options.iterations; ++iteration) {
      // N2Cube may reuse task tensor memory for intermediate data.  Refill the
      // input before every invocation so repeated runs receive identical data.
      populate_input(input, static_cast<std::size_t>(input_size), options.input_mode);
      const auto start = std::chrono::steady_clock::now();
      const int run_result = dpuRunTask(handles.task);
      const auto end = std::chrono::steady_clock::now();
      if (run_result != 0) throw std::runtime_error("dpuRunTask failed");
      run_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

      if (iteration == 1) {
        for (const std::string& node : options.output_nodes) {
          const int size = dpuGetOutputTensorSize(handles.task, node.c_str());
          int8_t* const address = dpuGetOutputTensorAddress(handles.task, node.c_str());
          if (size <= 0 || address == nullptr) throw std::runtime_error("invalid output tensor: " + node);
          OutputRecord record = {
              node,
              dpuGetOutputTensorHeight(handles.task, node.c_str()),
              dpuGetOutputTensorWidth(handles.task, node.c_str()),
              dpuGetOutputTensorChannel(handles.task, node.c_str()),
              size,
              dpuGetOutputTensorScale(handles.task, node.c_str()),
              compute_tensor_stats(address, static_cast<std::size_t>(size)),
              std::vector<int8_t>(address, address + size),
              options.dump_dir + "/" + safe_dump_name(node),
          };
          write_binary(record.dump_path, record.baseline);
          outputs.push_back(record);
        }
      } else {
        bool iteration_mismatch = false;
        for (const OutputRecord& record : outputs) {
          int8_t* const address = dpuGetOutputTensorAddress(handles.task, record.node.c_str());
          if (address == nullptr) throw std::runtime_error("output tensor disappeared: " + record.node);
          std::size_t mismatch_offset = 0;
          if (!tensors_equal(
                  record.baseline.data(),
                  address,
                  static_cast<std::size_t>(record.size),
                  &mismatch_offset)) {
            iteration_mismatch = true;
            if (first_mismatch_iteration == 0) {
              first_mismatch_iteration = static_cast<std::size_t>(iteration);
              first_mismatch_node = record.node;
              first_mismatch_offset = mismatch_offset;
            }
          }
        }
        if (iteration_mismatch) ++mismatch_iterations;
      }
    }

    const long long total_us = std::accumulate(run_us.begin(), run_us.end(), 0LL);
    const auto minmax = std::minmax_element(run_us.begin(), run_us.end());
    std::ofstream report(options.report_path.c_str(), std::ios::trunc);
    if (!report) throw std::runtime_error("cannot open report");
    report << std::setprecision(9);
    report << "DPU_RUN_STATUS=" << (mismatch_iterations == 0 ? "pass" : "fail") << '\n';
    report << "KERNEL=" << options.kernel << '\n';
    report << "INPUT_NODE=" << options.input_node << '\n';
    report << "INPUT_MODE=" << options.input_mode << '\n';
    report << "INPUT_SIZE=" << input_size << '\n';
    report << "INPUT_HEIGHT=" << input_height << '\n';
    report << "INPUT_WIDTH=" << input_width << '\n';
    report << "INPUT_CHANNEL=" << input_channel << '\n';
    report << "INPUT_SCALE=" << input_scale << '\n';
    report << "ITERATIONS=" << options.iterations << '\n';
    report << "DPU_RUN_CALLS=" << options.iterations << '\n';
    report << "RUN_TOTAL_US=" << total_us << '\n';
    report << "RUN_AVG_US=" << (static_cast<double>(total_us) / options.iterations) << '\n';
    report << "RUN_MIN_US=" << *minmax.first << '\n';
    report << "RUN_MAX_US=" << *minmax.second << '\n';
    report << "MISMATCH_ITERATIONS=" << mismatch_iterations << '\n';
    report << "FIRST_MISMATCH_ITERATION=" << first_mismatch_iteration << '\n';
    report << "FIRST_MISMATCH_NODE=" << first_mismatch_node << '\n';
    report << "FIRST_MISMATCH_OFFSET=" << first_mismatch_offset << '\n';
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const OutputRecord& output = outputs[index];
      const std::string prefix = "OUTPUT_" + std::to_string(index) + "_";
      report << prefix << "NODE=" << output.node << '\n';
      report << prefix << "SIZE=" << output.size << '\n';
      report << prefix << "HEIGHT=" << output.height << '\n';
      report << prefix << "WIDTH=" << output.width << '\n';
      report << prefix << "CHANNEL=" << output.channel << '\n';
      report << prefix << "SCALE=" << output.scale << '\n';
      report << prefix << "MIN=" << output.stats.minimum << '\n';
      report << prefix << "MAX=" << output.stats.maximum << '\n';
      report << prefix << "NONZERO=" << output.stats.nonzero << '\n';
      report << prefix << "DUMP=" << output.dump_path << '\n';
    }
    if (!report) throw std::runtime_error("cannot write report");
    std::cout << "DPU_RUN_STATUS=" << (mismatch_iterations == 0 ? "pass" : "fail") << '\n';
    std::cout << "DPU_RUN_CALLS=" << options.iterations << '\n';
    return mismatch_iterations == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    usage(argv[0]);
    std::cerr << "DPU_RUN_STATUS=fail\nREASON=" << error.what() << '\n';
    return 1;
  }
}

#include "logmel_frontend.h"
#include "wav_pcm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct UsageError : std::runtime_error { explicit UsageError(const std::string& s) : std::runtime_error(s) {} };

struct Options {
  std::string config, wav, float_output, int8_output, report, golden_float, golden_int8;
  uint64_t start_sample = std::numeric_limits<uint64_t>::max();
  double input_scale = std::numeric_limits<double>::quiet_NaN();
};

uint64_t uint64_value(const std::string& value) {
  if (value.empty() || value[0] == '-') throw UsageError("start-sample must be non-negative");
  std::size_t used = 0;
  const unsigned long long parsed = std::stoull(value, &used, 10);
  if (used != value.size()) throw UsageError("invalid start-sample");
  return static_cast<uint64_t>(parsed);
}

double double_value(const std::string& value) {
  std::size_t used = 0;
  const double parsed = std::stod(value, &used);
  if (used != value.size() || !std::isfinite(parsed) || parsed <= 0.0)
    throw UsageError("input-scale must be finite and positive");
  return parsed;
}

Options options(int argc, char** argv) {
  Options out;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (++i >= argc) throw UsageError("missing value for " + key);
    const std::string value(argv[i]);
    if (key == "--config") out.config = value;
    else if (key == "--wav") out.wav = value;
    else if (key == "--start-sample") out.start_sample = uint64_value(value);
    else if (key == "--input-scale") out.input_scale = double_value(value);
    else if (key == "--float-output") out.float_output = value;
    else if (key == "--int8-output") out.int8_output = value;
    else if (key == "--report") out.report = value;
    else if (key == "--golden-float") out.golden_float = value;
    else if (key == "--golden-int8") out.golden_int8 = value;
    else throw UsageError("unknown option: " + key);
  }
  if (out.config.empty() || out.wav.empty() || out.float_output.empty() ||
      out.int8_output.empty() || out.report.empty() ||
      out.start_sample == std::numeric_limits<uint64_t>::max() ||
      !std::isfinite(out.input_scale))
    throw UsageError("required option missing");
  if (out.golden_float.empty() != out.golden_int8.empty())
    throw UsageError("golden files must be provided together");
  return out;
}

bool exists(const std::string& path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  return static_cast<bool>(stream);
}

template <typename T>
void write_binary(const std::string& path, const std::vector<T>& values) {
  std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot create output");
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!stream) throw std::runtime_error("cannot write output");
}

std::vector<uint8_t> read_exact(const std::string& path, std::size_t size) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open golden file");
  std::vector<uint8_t> data(size);
  stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
  if (stream.gcount() != static_cast<std::streamsize>(size))
    throw std::runtime_error("short golden file");
  char extra = 0;
  if (stream.read(&extra, 1)) throw std::runtime_error("long golden file");
  return data;
}

void publish_no_replace(const std::string& temporary, const std::string& final_path,
                        std::vector<std::string>* published) {
  if (::link(temporary.c_str(), final_path.c_str()) != 0)
    throw std::runtime_error("cannot publish output without replacement");
  published->push_back(final_path);
  if (::unlink(temporary.c_str()) != 0) {
    std::remove(final_path.c_str());
    published->pop_back();
    throw std::runtime_error("cannot remove temporary output");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> temporary;
  std::vector<std::string> published;
  try {
    const Options opt = options(argc, argv);
    for (const std::string* path : {&opt.float_output, &opt.int8_output, &opt.report})
      if (exists(*path)) throw std::runtime_error("refusing to replace existing output");
    const soundjie::FrontendConfig config = soundjie::load_frontend_config(opt.config);
    const soundjie::PcmWindow pcm = soundjie::read_pcm_window(
        opt.wav, opt.start_sample, config.window_samples);
    const std::vector<float> features = soundjie::compute_logmel(pcm.samples, config);
    const std::vector<int8_t> tensor = soundjie::quantize_int8(features, opt.input_scale);

    double max_abs = 0.0, squared = 0.0;
    std::size_t max_index = 0;
    long first_mismatch = -1;
    if (!opt.golden_float.empty()) {
      const std::vector<uint8_t> golden_float = read_exact(opt.golden_float, features.size() * 4);
      const std::vector<uint8_t> golden_int8 = read_exact(opt.golden_int8, tensor.size());
      for (std::size_t index = 0; index < features.size(); ++index) {
        float golden = 0.0f;
        std::memcpy(&golden, &golden_float[index * 4], 4);
        const double difference = std::abs(static_cast<double>(golden) - features[index]);
        squared += difference * difference;
        if (difference > max_abs) { max_abs = difference; max_index = index; }
        if (first_mismatch < 0 && golden_int8[index] != static_cast<uint8_t>(tensor[index]))
          first_mismatch = static_cast<long>(index);
      }
    }
    const double rmse = std::sqrt(squared / features.size());
    const bool parity = opt.golden_float.empty() ||
        (max_abs <= 1e-4 && rmse <= 1e-5 && first_mismatch < 0);

    const std::string suffix = ".new." + std::to_string(static_cast<long long>(getpid()));
    const std::string float_tmp = opt.float_output + suffix;
    const std::string int8_tmp = opt.int8_output + suffix;
    const std::string report_tmp = opt.report + suffix;
    temporary = {float_tmp, int8_tmp, report_tmp};
    write_binary(float_tmp, features);
    write_binary(int8_tmp, tensor);
    std::ofstream report(report_tmp.c_str(), std::ios::trunc);
    if (!report) throw std::runtime_error("cannot create report");
    report << std::setprecision(17)
           << "LOGMEL_STATUS=" << (parity ? "pass" : "fail") << '\n'
           << "FLOAT_COUNT=" << features.size() << '\n'
           << "FLOAT_BYTES=" << features.size() * sizeof(float) << '\n'
           << "INT8_BYTES=" << tensor.size() << '\n'
           << "INPUT_SCALE=" << opt.input_scale << '\n'
           << "MAX_ABS=" << max_abs << '\n'
           << "RMSE=" << rmse << '\n'
           << "FIRST_MAX_ERROR_INDEX=" << max_index << '\n'
           << "INT8_EXACT=" << (first_mismatch < 0 ? "yes" : "no") << '\n'
           << "FIRST_INT8_MISMATCH_INDEX=" << first_mismatch << '\n';
    report.close();
    if (!report) throw std::runtime_error("cannot write report");
    publish_no_replace(float_tmp, opt.float_output, &published);
    publish_no_replace(int8_tmp, opt.int8_output, &published);
    publish_no_replace(report_tmp, opt.report, &published);
    temporary.clear();
    published.clear();
    std::cout << "LOGMEL_STATUS=" << (parity ? "pass" : "fail") << '\n';
    return parity ? 0 : 1;
  } catch (const UsageError& error) {
    std::cerr << "LOGMEL_STATUS=fail\nREASON=" << error.what() << '\n';
    return 64;
  } catch (const std::exception& error) {
    for (const std::string& path : temporary) std::remove(path.c_str());
    for (const std::string& path : published) std::remove(path.c_str());
    std::cerr << "LOGMEL_STATUS=fail\nREASON=" << error.what() << '\n';
    return 1;
  }
}

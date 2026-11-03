#include "live_result_fuser.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void usage() {
  std::cerr << "usage: live_result_fuser --predictions TSV --labels-en FILE "
               "--labels-zh FILE --output-scale NUMBER --session-result NEW.json "
               "--display-status NEW.txt [--expected-class ID]\n";
}

bool exists(const std::string& path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  return stream.good();
}

std::vector<std::string> split(const std::string& value, char delimiter) {
  std::vector<std::string> fields;
  std::string field;
  std::istringstream stream(value);
  while (std::getline(stream, field, delimiter)) fields.push_back(field);
  if (!value.empty() && value.back() == delimiter) fields.push_back(std::string());
  return fields;
}

uint64_t parse_u64(const std::string& value, const char* name) {
  std::size_t used = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &used, 10);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid integer field: ") + name);
  }
  if (used != value.size()) throw std::runtime_error(std::string("invalid integer field: ") + name);
  return static_cast<uint64_t>(parsed);
}

int parse_int(const std::string& value, const char* name) {
  std::size_t used = 0;
  long parsed = 0;
  try {
    parsed = std::stol(value, &used, 10);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid integer field: ") + name);
  }
  if (used != value.size() || parsed < -2147483647L - 1L || parsed > 2147483647L)
    throw std::runtime_error(std::string("invalid integer field: ") + name);
  return static_cast<int>(parsed);
}

double parse_double(const std::string& value, const char* name) {
  std::size_t used = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &used);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid floating field: ") + name);
  }
  if (used != value.size() || !std::isfinite(parsed))
    throw std::runtime_error(std::string("invalid floating field: ") + name);
  return parsed;
}

std::vector<std::string> read_labels(const std::string& path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open label file");
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) throw std::runtime_error("label line must not be empty");
    labels.push_back(line);
  }
  if (labels.size() != 8) throw std::runtime_error("label file must contain exactly eight lines");
  return labels;
}

std::vector<soundjie::LiveWindowPrediction> read_predictions(const std::string& path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open predictions TSV");
  std::string line;
  if (!std::getline(stream, line)) throw std::runtime_error("predictions TSV is empty");
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line != "rank\tstart_sample\truntime_us\traw_logits_csv\tinput_sha256\toutput_sha256")
    throw std::runtime_error("predictions TSV header mismatch");
  std::vector<soundjie::LiveWindowPrediction> rows;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::vector<std::string> fields = split(line, '\t');
    if (fields.size() != 6) throw std::runtime_error("prediction row must contain six fields");
    const std::vector<std::string> logits_text = split(fields[3], ',');
    std::vector<int> logits;
    for (const std::string& value : logits_text) logits.push_back(parse_int(value, "raw_logits_csv"));
    rows.push_back(soundjie::LiveWindowPrediction{
        static_cast<uint32_t>(parse_u64(fields[0], "rank")),
        parse_u64(fields[1], "start_sample"), parse_u64(fields[2], "runtime_us"),
        logits, fields[4], fields[5]});
  }
  return rows;
}

std::string json_string(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (unsigned char byte : value) {
    switch (byte) {
      case '"': stream << "\\\""; break;
      case '\\': stream << "\\\\"; break;
      case '\b': stream << "\\b"; break;
      case '\f': stream << "\\f"; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default:
        if (byte < 0x20) {
          stream << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                 << static_cast<unsigned int>(byte) << std::dec << std::setfill(' ');
        } else {
          stream << static_cast<char>(byte);
        }
    }
  }
  stream << '"';
  return stream.str();
}

std::string result_json(const soundjie::LiveSessionResult& result) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\n  \"schema\":1,\n  \"LIVE_STATUS\":\"pass\",\n";
  out << "  \"DPU_RUN_CALLS\":" << result.windows.size() << ",\n";
  out << "  \"expected_class\":"
      << (result.expected_class.empty() ? "null" : json_string(result.expected_class)) << ",\n";
  out << "  \"PREDICTION_MATCH\":" << json_string(result.prediction_match) << ",\n";
  const soundjie::LiveClassScore& top1 = result.top3[0];
  out << "  \"top1\":{\"index\":" << top1.index << ",\"id\":" << json_string(top1.id)
      << ",\"label_zh\":" << json_string(top1.label_zh)
      << ",\"confidence\":" << top1.confidence << "},\n";
  out << "  \"top3\":[";
  for (std::size_t index = 0; index < result.top3.size(); ++index) {
    if (index) out << ',';
    const auto& item = result.top3[index];
    out << "{\"index\":" << item.index << ",\"id\":" << json_string(item.id)
        << ",\"label_zh\":" << json_string(item.label_zh)
        << ",\"confidence\":" << item.confidence << '}';
  }
  out << "],\n  \"fused_logits\":[";
  for (std::size_t index = 0; index < result.fused_logits.size(); ++index) {
    if (index) out << ',';
    out << result.fused_logits[index];
  }
  out << "],\n  \"windows\":[";
  for (std::size_t row = 0; row < result.windows.size(); ++row) {
    if (row) out << ',';
    const auto& window = result.windows[row];
    out << "{\"rank\":" << window.rank << ",\"start_sample\":" << window.start_sample
        << ",\"runtime_us\":" << window.runtime_us << ",\"raw_logits\":[";
    for (std::size_t index = 0; index < window.raw_logits.size(); ++index) {
      if (index) out << ',';
      out << window.raw_logits[index];
    }
    out << "],\"input_sha256\":" << json_string(window.input_sha256)
        << ",\"output_sha256\":" << json_string(window.output_sha256) << '}';
  }
  out << "]\n}\n";
  return out.str();
}

std::string display_text(const soundjie::LiveSessionResult& result) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "LIVE_STATUS=pass\nCLASS_ZH=" << result.top3[0].label_zh
      << "\nCLASS_ID=" << result.top3[0].id
      << "\nCONFIDENCE=" << result.top3[0].confidence << "\nTOP3=";
  for (std::size_t index = 0; index < result.top3.size(); ++index) {
    if (index) out << ',';
    out << result.top3[index].id << ':' << result.top3[index].confidence;
  }
  out << "\nPREDICTION_MATCH=" << result.prediction_match << "\nERROR=none\n";
  return out.str();
}

std::string temporary_path(const std::string& destination, const char* tag) {
  return destination + ".new." + std::to_string(static_cast<long long>(getpid())) + "." + tag;
}

void write_file(const std::string& path, const std::string& content) {
  std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot create temporary output");
  stream.write(content.data(), static_cast<std::streamsize>(content.size()));
  stream.close();
  if (!stream) throw std::runtime_error("cannot write temporary output");
}

void publish_pair(const std::string& json_path, const std::string& display_path,
                  const std::string& json, const std::string& display) {
  if (exists(json_path) || exists(display_path)) throw std::runtime_error("output already exists");
  const std::string json_temp = temporary_path(json_path, "json");
  const std::string display_temp = temporary_path(display_path, "display");
  try {
    write_file(json_temp, json);
    write_file(display_temp, display);
    if (link(json_temp.c_str(), json_path.c_str()) != 0)
      throw std::runtime_error(std::string("cannot publish JSON output: ") + std::strerror(errno));
    if (link(display_temp.c_str(), display_path.c_str()) != 0) {
      unlink(json_path.c_str());
      throw std::runtime_error(std::string("cannot publish display output: ") + std::strerror(errno));
    }
    unlink(json_temp.c_str());
    unlink(display_temp.c_str());
  } catch (...) {
    unlink(json_temp.c_str());
    unlink(display_temp.c_str());
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      usage();
      return 64;
    }
    const std::string key = argv[index];
    if (key != "--predictions" && key != "--labels-en" && key != "--labels-zh" &&
        key != "--output-scale" && key != "--session-result" &&
        key != "--display-status" && key != "--expected-class") {
      usage();
      return 64;
    }
    if (!options.insert(std::make_pair(key, argv[index + 1])).second) {
      usage();
      return 64;
    }
  }
  const char* required[] = {"--predictions", "--labels-en", "--labels-zh", "--output-scale",
                            "--session-result", "--display-status"};
  for (const char* key : required) {
    if (options.count(key) == 0) {
      usage();
      return 64;
    }
  }
  if (exists(options["--session-result"]) || exists(options["--display-status"])) {
    std::cerr << "output already exists\n";
    return 3;
  }
  try {
    const std::vector<soundjie::LiveWindowPrediction> predictions =
        read_predictions(options["--predictions"]);
    const soundjie::LiveSessionResult result = soundjie::fuse_live_results(
        predictions, parse_double(options["--output-scale"], "output_scale"),
        read_labels(options["--labels-en"]), read_labels(options["--labels-zh"]),
        options.count("--expected-class") ? options["--expected-class"] : std::string());
    publish_pair(options["--session-result"], options["--display-status"],
                 result_json(result), display_text(result));
    std::cout << "LIVE_STATUS=pass\nDPU_RUN_CALLS=" << result.windows.size()
              << "\nPREDICTION_MATCH=" << result.prediction_match << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 4;
  }
}

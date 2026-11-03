#include "live_window_selector.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void usage() {
  std::cerr << "usage: live_window_selector --wav PATH --config TSV --metrics NEW.tsv "
               "--selected NEW.tsv\n";
}

bool exists(const std::string& path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  return stream.good();
}

std::string temporary_path(const std::string& destination, const char* tag) {
  return destination + ".new." + std::to_string(static_cast<long long>(getpid())) + "." + tag;
}

void write_metrics(const std::string& path, const soundjie::LiveWindowSelection& selection) {
  std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot create temporary metrics output");
  stream << "start_sample\tstart_ms\trms\tpeak\tclipping_rate\taccepted\trejection\tselected_rank\n";
  stream << std::setprecision(17);
  for (const auto& row : selection.metrics) {
    stream << row.start_sample << '\t' << row.start_ms << '\t' << row.rms << '\t'
           << row.peak << '\t' << row.clipping_rate << '\t' << (row.accepted ? 1 : 0)
           << '\t' << row.rejection << '\t';
    if (row.selected_rank != 0) stream << row.selected_rank;
    stream << '\n';
  }
  stream.close();
  if (!stream) throw std::runtime_error("cannot write temporary metrics output");
}

void write_selected(const std::string& path, const soundjie::LiveWindowSelection& selection) {
  std::ofstream stream(path.c_str(), std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot create temporary selected output");
  stream << "rank\tstart_sample\trms\tpeak\tclipping_rate\n";
  stream << std::setprecision(17);
  for (const auto& row : selection.selected) {
    stream << row.rank << '\t' << row.start_sample << '\t' << row.rms << '\t'
           << row.peak << '\t' << row.clipping_rate << '\n';
  }
  stream.close();
  if (!stream) throw std::runtime_error("cannot write temporary selected output");
}

void publish_pair(const std::string& metrics, const std::string& selected,
                  const soundjie::LiveWindowSelection& selection) {
  if (exists(metrics) || exists(selected)) throw std::runtime_error("output already exists");
  const std::string metrics_temp = temporary_path(metrics, "metrics");
  const std::string selected_temp = temporary_path(selected, "selected");
  if (exists(metrics_temp) || exists(selected_temp))
    throw std::runtime_error("temporary output already exists");
  try {
    write_metrics(metrics_temp, selection);
    write_selected(selected_temp, selection);
    if (link(metrics_temp.c_str(), metrics.c_str()) != 0)
      throw std::runtime_error(std::string("cannot publish metrics output: ") + std::strerror(errno));
    if (link(selected_temp.c_str(), selected.c_str()) != 0) {
      unlink(metrics.c_str());
      throw std::runtime_error(std::string("cannot publish selected output: ") + std::strerror(errno));
    }
    unlink(metrics_temp.c_str());
    unlink(selected_temp.c_str());
  } catch (...) {
    unlink(metrics_temp.c_str());
    unlink(selected_temp.c_str());
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
    if (key != "--wav" && key != "--config" && key != "--metrics" && key != "--selected") {
      usage();
      return 64;
    }
    if (!options.insert(std::make_pair(key, argv[index + 1])).second) {
      usage();
      return 64;
    }
  }
  if (options.size() != 4) {
    usage();
    return 64;
  }
  if (exists(options["--metrics"]) || exists(options["--selected"])) {
    std::cerr << "output already exists\n";
    return 3;
  }
  try {
    const soundjie::LiveWindowConfig config =
        soundjie::load_live_window_config(options["--config"]);
    const soundjie::LiveWindowSelection selection =
        soundjie::select_live_windows(options["--wav"], config);
    publish_pair(options["--metrics"], options["--selected"], selection);
    std::cout << "CANDIDATE_WINDOWS=" << selection.metrics.size() << '\n';
    std::cout << "SELECTED_WINDOWS=" << selection.selected.size() << '\n';
    if (selection.selected.empty()) {
      std::cout << "REASON=audio_signal_absent\n";
      return 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 4;
  }
}

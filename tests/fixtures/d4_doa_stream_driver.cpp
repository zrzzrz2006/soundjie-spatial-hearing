#include "soundjie_d4_doa_stream.h"
#include "xvf3800_control_protocol.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Test-only filesystem driver; production has no fixture/control commands.
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string path = argv[1];
  soundjie::D4DoaStream reader(path, 0.5, 750);
  std::string command;
  while (std::getline(std::cin, command)) {
    const auto split = command.find(' ');
    const std::string op = command.substr(0, split);
    const std::string arg = split == std::string::npos ? "" : command.substr(split + 1);
    if (op == "tick") {
      soundjie::SoundjieD4Snapshot snapshot;
      snapshot.doa = reader.poll([&]() { return std::stoull(arg); });
      std::cout << soundjie::soundjie_d4_snapshot_json(snapshot) << '\n';
    } else if (op == "clock-race") {
      // The clock callback appends a newer row. It must run AFTER the bounded
      // read, so this row belongs to the next poll, not to the old clock tick.
      int calls = 0;
      auto clock = [&]() -> uint64_t {
        ++calls;
        std::ofstream out(path, std::ios::app);
        out << soundjie::xvf_doa_json(soundjie::parse_xvf3800_doa_response(
            soundjie::parse_hex_bytes("005a000100")), 1001) << '\n';
        return 1001;
      };
      soundjie::SoundjieD4Snapshot snapshot;
      snapshot.doa = reader.poll(clock);
      if (calls != 1) throw std::runtime_error("clock must be sampled once after read");
      std::cout << soundjie::soundjie_d4_snapshot_json(snapshot) << '\n';
    } else if (op == "remove") {
      std::remove(path.c_str());
    } else if (op == "truncate") {
      std::ofstream out(path, std::ios::trunc);
    } else if (op == "pad") {
      std::ofstream out(path, std::ios::app);
      out << std::string(std::stoul(arg), 'x') << '\n';
    } else if (op == "sample") {
      std::istringstream values(arg);
      uint64_t timestamp;
      std::string raw;
      values >> timestamp >> raw;
      std::ofstream out(path, std::ios::app);
      out << soundjie::xvf_doa_json(soundjie::parse_xvf3800_doa_response(
          soundjie::parse_hex_bytes(raw)), timestamp) << '\n';
    } else if (op == "append" || op == "fragment" || op == "replace") {
      const std::string target = op == "replace" ? path + ".new" : path;
      std::ofstream out(target, op == "replace" ? std::ios::trunc : std::ios::app);
      out << arg;
      if (op != "fragment") out << '\n';
      out.close();
      if (op == "replace" && std::rename(target.c_str(), path.c_str()) != 0) return 3;
    } else {
      throw std::runtime_error("unknown driver command");
    }
  }
}

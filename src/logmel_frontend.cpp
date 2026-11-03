#include "logmel_frontend.h"

#include "fft_radix2.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace soundjie {
namespace {

const char kProfileId[] = "bounded-log-power-64-v1";
const char kProfileDomain[] = "natural_log_mel_power";
const double kProfileLowerBound = -23.025850929940457;
const double kProfileUpperBound = 5.9914645471079817;
const double kProfileMean = -8.5171931914162382;
const double kProfileStd = 14.508657738524219;
const double kNormalizedBoundTolerance = 1e-6;

std::string read_text(const std::string& path) {
  std::ifstream stream(path.c_str(), std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open frontend config");
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

class JsonSyntaxParser {
 public:
  explicit JsonSyntaxParser(const std::string& text) : text_(text), pos_(0), object_depth_(0) {}
  void parse() {
    value("");
    whitespace();
    if (pos_ != text_.size()) fail();
  }
  const std::set<std::string>& root_keys() const { return root_keys_; }
  std::string raw(const std::string& key) const {
    auto it=paths_.find(key);
    if(it==paths_.end())throw std::runtime_error("missing "+key);
    return it->second;
  }
  std::string at_path(const std::string& path) const {return paths_.at(path);}

  const std::set<std::string>& keys(const std::string& path) const {
    const auto found = object_keys_.find(path);
    if (found == object_keys_.end()) fail();
    return found->second;
  }

 private:
  const std::string& text_;
  std::size_t pos_;
  int object_depth_;
  std::map<std::string,std::string> paths_;
  std::set<std::string> root_keys_;
  std::map<std::string, std::set<std::string> > object_keys_;
  void fail() const { throw std::runtime_error("invalid JSON frontend config"); }
  void whitespace() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
  }
  bool take(char value) {
    whitespace();
    if (pos_ < text_.size() && text_[pos_] == value) { ++pos_; return true; }
    return false;
  }
  void literal(const char* value) {
    whitespace();
    const std::size_t size = std::strlen(value);
    if (text_.compare(pos_, size, value) != 0) fail();
    pos_ += size;
  }
  std::string string(bool key=false) {
    whitespace();
    if (pos_ >= text_.size() || text_[pos_++] != '"') fail();
    std::string output;
    while (pos_ < text_.size()) {
      const unsigned char ch = static_cast<unsigned char>(text_[pos_++]);
      if (ch == '"') return output;
      if (ch < 0x20) fail();
      if (key && (ch == '/' || ch == '[' || ch == ']')) fail();
      if (ch == '\\') {
        // Configuration keys are pinned ASCII identifiers, never escape aliases.
        if(key)fail();
        if (pos_ >= text_.size()) fail();
        const char escaped = text_[pos_++];
        if (std::string("\"\\/bfnrt").find(escaped) != std::string::npos) {
          output.push_back(escaped);
        } else if (escaped == 'u') {
          for (int i = 0; i < 4; ++i) {
            if (pos_ >= text_.size() || !std::isxdigit(static_cast<unsigned char>(text_[pos_++]))) fail();
          }
          output.push_back('?');
        } else fail();
      } else output.push_back(static_cast<char>(ch));
    }
    fail();
    return output;
  }
  void number_value() {
    whitespace();
    // strtod accepts C forms (+2, 0x2, 02) that are not JSON numbers.
    const auto digit=[&](){return pos_<text_.size()&&text_[pos_]>='0'&&text_[pos_]<='9';};
    if(pos_<text_.size()&&text_[pos_]=='-')++pos_;
    if(!digit())fail();
    if(text_[pos_]=='0'){++pos_;if(digit())fail();}
    else while(digit())++pos_;
    if(pos_<text_.size()&&text_[pos_]=='.'){
      ++pos_;
      if(!digit())fail();
      while(digit())++pos_;
    }
    if(pos_<text_.size()&&(text_[pos_]=='e'||text_[pos_]=='E')){
      ++pos_;if(pos_<text_.size()&&(text_[pos_]=='+'||text_[pos_]=='-'))++pos_;
      if(!digit())fail();
      while(digit())++pos_;
    }
  }
  void array(const std::string& path) {
    if (take(']')) return;
    do { value(path + "[]"); } while (take(','));
    if (!take(']')) fail();
  }
  void object(const std::string& path) {
    const bool root = object_depth_++ == 0;
    std::set<std::string> keys;
    if (take('}')) { --object_depth_; return; }
    do {
      const std::string key = string(true);
      if (!keys.insert(key).second || !take(':')) fail();
      if (root) root_keys_.insert(key);
      value(path + "/" + key);
    } while (take(','));
    if (!take('}')) fail();
    object_keys_[path] = keys;
    --object_depth_;
  }
  void value(const std::string& path) {
    whitespace();
    if (pos_ >= text_.size()) fail();
    const auto start=pos_;
    if (take('{')) object(path);
    else if (take('[')) array(path);
    else if (text_[pos_] == '"') string();
    else if (text_[pos_] == 't') literal("true");
    else if (text_[pos_] == 'f') literal("false");
    else if (text_[pos_] == 'n') literal("null");
    else number_value();
    // Keep the whole path and value, including objects. An object wrapping a
    // scalar must never alias the expected scalar at its parent path.
    paths_[path]=text_.substr(start,pos_-start);
  }
};

double number(const JsonSyntaxParser& parser,const std::string& key) {
  const auto raw=parser.raw(key);std::size_t used=0;const double v=std::stod(raw,&used);
  if(used!=raw.size()||!std::isfinite(v))throw std::runtime_error("invalid number "+key);
  return v;
}
std::string string_value(const JsonSyntaxParser& parser,const std::string& key) {
  const auto raw=parser.raw(key);
  if(raw.size()<2||raw.front()!='"'||raw.back()!='"')throw std::runtime_error("invalid string "+key);
  return raw.substr(1,raw.size()-2);
}
bool bool_value(const JsonSyntaxParser& parser,const std::string& key) {
  const auto raw=parser.raw(key);if(raw!="true"&&raw!="false")throw std::runtime_error("invalid bool "+key);
  return raw=="true";
}
std::vector<double> array_value(const JsonSyntaxParser& parser,const std::string& key) {
  const auto raw=parser.raw(key);
  if(raw.size()<2||raw.front()!='['||raw.back()!=']')throw std::runtime_error("invalid array "+key);
  std::vector<double> output;std::stringstream stream(raw.substr(1,raw.size()-2));std::string token;
  while(std::getline(stream,token,',')){
    if(token.find_first_not_of(" \t\r\n")==std::string::npos)continue;
    std::size_t used=0;const double v=std::stod(token,&used);
    if(token.find_first_not_of(" \t\r\n",used)!=std::string::npos||!std::isfinite(v))throw std::runtime_error("invalid array value "+key);
    output.push_back(v);
  }
  return output;
}

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double hz_to_mel(double frequency) {
  return 2595.0 * std::log10(1.0 + frequency / 700.0);
}

double mel_to_hz(double value) {
  return 700.0 * (std::pow(10.0, value / 2595.0) - 1.0);
}

}  // namespace

FrontendConfig load_frontend_config(const std::string& path) {
  const std::string text = read_text(path);
  JsonSyntaxParser parser(text);
  parser.parse();
  const std::set<std::string> expected_root = {
      "schema", "audio", "stft", "power_spectrum", "mel", "log",
      "tensor", "normalization", "quantization"};
  require(parser.root_keys() == expected_root, "invalid config structure");
  require(parser.keys("/audio") == std::set<std::string>({
      "sample_rate_hz", "sample_format", "supported_channels",
      "window_samples", "stream_hop_samples"}), "invalid config structure");
  require(parser.keys("/stft") == std::set<std::string>({
      "frame_length", "hop_length", "fft_length", "window", "center",
      "expected_frames", "keep_frame_start", "keep_frame_count"}),
      "invalid config structure");
  require(parser.keys("/power_spectrum") == std::set<std::string>({
      "formula", "divisor", "single_sided_doubling",
      "window_energy_normalization"}), "invalid config structure");
  require(parser.keys("/mel") == std::set<std::string>({
      "bands", "min_hz", "max_hz", "scale", "normalization"}),
      "invalid config structure");
  require(parser.keys("/log") == std::set<std::string>({"base", "floor"}),
      "invalid config structure");
  require(parser.keys("/tensor") == std::set<std::string>({
      "layout", "shape", "order", "dtype"}), "invalid config structure");
  require(parser.keys("/normalization") == std::set<std::string>({
      "id", "domain", "lower_bound", "upper_bound", "mean", "std"}),
      "invalid config structure");
  require(parser.keys("/quantization") == std::set<std::string>({
      "dtype", "rounding", "clamp", "scale_source"}),
      "invalid config structure");
  require(number(parser, "/schema") == 2.0, "unsupported schema");
  require(number(parser, "/audio/sample_rate_hz") == 16000.0, "wrong sample rate");
  require(string_value(parser, "/audio/sample_format") == "s16le", "wrong sample format");
  const std::vector<double> channels = array_value(parser, "/audio/supported_channels");
  require(channels == std::vector<double>({1.0, 2.0}), "wrong supported channels");
  require(number(parser, "/audio/window_samples") == 16000.0, "wrong window samples");
  require(number(parser, "/audio/stream_hop_samples") == 4000.0, "wrong stream hop");
  require(number(parser, "/stft/frame_length") == 400.0, "wrong frame length");
  require(number(parser, "/stft/hop_length") == 160.0, "wrong hop length");
  require(number(parser, "/stft/fft_length") == 512.0, "wrong FFT length");
  require(string_value(parser, "/stft/window") == "periodic_hann", "wrong window type");
  require(!bool_value(parser, "/stft/center"), "center padding is forbidden");
  require(number(parser, "/stft/expected_frames") == 98.0, "wrong frame count");
  require(number(parser, "/stft/keep_frame_start") == 1.0, "wrong first kept frame");
  require(number(parser, "/stft/keep_frame_count") == 96.0, "wrong kept frame count");
  require(string_value(parser, "/power_spectrum/formula") == "magnitude_squared_div_fft_length",
          "wrong power formula");
  require(number(parser, "/power_spectrum/divisor") == 512.0, "wrong power divisor");
  require(!bool_value(parser, "/power_spectrum/single_sided_doubling"), "single-sided doubling forbidden");
  require(!bool_value(parser, "/power_spectrum/window_energy_normalization"), "window normalization forbidden");
  require(number(parser, "/mel/bands") == 64.0, "wrong Mel band count");
  require(number(parser, "/mel/min_hz") == 20.0 && number(parser, "/mel/max_hz") == 7600.0,
          "wrong Mel frequency limits");
  require(string_value(parser, "/mel/scale") == "htk", "wrong Mel scale");
  require(string_value(parser, "/mel/normalization") == "none", "wrong Mel normalization");
  require(string_value(parser, "/log/base") == "natural", "wrong log base");
  require(number(parser, "/log/floor") == 1e-10, "wrong log floor");
  require(string_value(parser, "/tensor/layout") == "NCHW", "wrong tensor layout");
  const std::vector<double> shape = array_value(parser, "/tensor/shape");
  require(shape == std::vector<double>({1.0, 1.0, 64.0, 96.0}), "wrong tensor shape");
  require(string_value(parser, "/tensor/order") == "mel_major_time_increasing", "wrong tensor order");
  require(string_value(parser, "/tensor/dtype") == "float32-le",
          "missing float32 tensor dtype");
  require(string_value(parser, "/quantization/dtype") == "int8",
          "missing int8 tensor dtype");
  require(string_value(parser, "/normalization/id") == kProfileId, "wrong profile id");
  require(string_value(parser, "/normalization/domain") == kProfileDomain, "wrong profile domain");
  const double lower_bound = number(parser, "/normalization/lower_bound");
  const double upper_bound = number(parser, "/normalization/upper_bound");
  require(lower_bound == kProfileLowerBound, "wrong profile lower bound");
  require(upper_bound == kProfileUpperBound, "wrong profile upper bound");
  require(kProfileMean == (lower_bound + upper_bound) / 2.0,
          "profile mean formula mismatch");
  require(kProfileStd == (upper_bound - lower_bound) / 2.0,
          "profile std formula mismatch");
  require(lower_bound == std::log(number(parser, "/log/floor")),
          "profile lower bound does not match log floor");
  require(string_value(parser, "/quantization/rounding") == "half_away_from_zero", "wrong rounding");
  require(string_value(parser, "/quantization/scale_source") == "runtime_dpu_input_tensor",
          "wrong scale source");
  const std::vector<double> clamp = array_value(parser, "/quantization/clamp");
  require(clamp == std::vector<double>({-128.0, 127.0}), "wrong int8 clamp");
  const std::vector<double> mean = array_value(parser, "/normalization/mean");
  const std::vector<double> stddev = array_value(parser, "/normalization/std");
  require(mean.size() == 64 && stddev.size() == 64, "profile must contain 64 bands");
  require(std::all_of(mean.begin(), mean.end(), [](double v) { return v == kProfileMean; }),
          "profile mean must match bounded profile");
  require(std::all_of(stddev.begin(), stddev.end(),
                      [](double v) { return v == kProfileStd && v > 0.0; }),
          "profile std must match bounded profile and be positive");

  FrontendConfig config{};
  config.sample_rate_hz = 16000;
  config.window_samples = 16000;
  config.stream_hop_samples = 4000;
  config.frame_length = 400;
  config.hop_length = 160;
  config.fft_length = 512;
  config.expected_frames = 98;
  config.keep_frame_start = 1;
  config.keep_frame_count = 96;
  config.mel_bands = 64;
  config.mel_min_hz = 20.0;
  config.mel_max_hz = 7600.0;
  config.log_floor = 1e-10;
  config.profile_id = kProfileId;
  config.profile_domain = kProfileDomain;
  config.profile_lower_bound = lower_bound;
  config.profile_upper_bound = upper_bound;
  std::copy(mean.begin(), mean.end(), config.profile_mean.begin());
  std::copy(stddev.begin(), stddev.end(), config.profile_std.begin());
  return config;
}

std::vector<float> compute_logmel(const std::vector<double>& samples,
                                  const FrontendConfig& config) {
  require(samples.size() == config.window_samples, "audio must contain 16000 samples");
  require(std::all_of(samples.begin(), samples.end(),
                      [](double value) { return std::isfinite(value); }),
          "audio samples must be finite");
  const double pi = 3.141592653589793238462643383279502884;
  std::array<double, 400> hann;
  for (std::size_t n = 0; n < hann.size(); ++n)
    hann[n] = 0.5 - 0.5 * std::cos(2.0 * pi * n / hann.size());

  std::vector<double> power(config.expected_frames * 257);
  for (uint32_t frame = 0; frame < config.expected_frames; ++frame) {
    std::array<double, 512> input{};
    const uint32_t begin = frame * config.hop_length;
    for (uint32_t n = 0; n < config.frame_length; ++n)
      input[n] = samples[begin + n] * hann[n];
    const std::array<std::complex<double>, 257> spectrum = fft_radix2_512(input);
    for (std::size_t bin = 0; bin < spectrum.size(); ++bin)
      power[frame * 257 + bin] = std::norm(spectrum[bin]) / config.fft_length;
  }

  const double mel_low = hz_to_mel(config.mel_min_hz);
  const double mel_high = hz_to_mel(config.mel_max_hz);
  std::array<double, 66> points;
  for (std::size_t index = 0; index < points.size(); ++index)
    points[index] = mel_to_hz(mel_low + (mel_high - mel_low) * index / 65.0);
  std::vector<double> filters(64 * 257, 0.0);
  for (std::size_t band = 0; band < 64; ++band) {
    const double left = points[band], center = points[band + 1], right = points[band + 2];
    require(center > left && right > center, "degenerate Mel triangle");
    for (std::size_t bin = 0; bin < 257; ++bin) {
      const double frequency = bin * config.sample_rate_hz / 512.0;
      const double rising = (frequency - left) / (center - left);
      const double falling = (right - frequency) / (right - center);
      filters[band * 257 + bin] = std::max(0.0, std::min(rising, falling));
    }
  }

  std::vector<float> output(64 * 96);
  for (std::size_t band = 0; band < 64; ++band) {
    // Preserve summation order while omitting the exactly-zero triangle tails.
    std::size_t first=0,last=257;
    while(first<last&&filters[band*257+first]==0)++first;
    while(last>first&&filters[band*257+last-1]==0)--last;
    for (std::size_t kept = 0; kept < 96; ++kept) {
      const std::size_t frame = kept + config.keep_frame_start;
      double energy = 0.0;
      for (std::size_t bin = first; bin < last; ++bin)
        energy += filters[band * 257 + bin] * power[frame * 257 + bin];
      const double logged = std::log(std::max(energy, config.log_floor));
      const double normalized = (logged - config.profile_mean[band]) / config.profile_std[band];
      require(std::isfinite(normalized), "non-finite Log-Mel feature");
      require(normalized >= -1.0 - kNormalizedBoundTolerance &&
              normalized <= 1.0 + kNormalizedBoundTolerance,
              "normalized Log-Mel feature exceeds D2 bounds");
      output[band * 96 + kept] = static_cast<float>(normalized);
    }
  }
  return output;
}

std::vector<int8_t> quantize_int8(const std::vector<float>& features,
                                 double input_scale) {
  require(features.size() == 64 * 96, "feature tensor must contain 6144 values");
  require(std::isfinite(input_scale) && input_scale > 0.0, "input scale must be positive");
  std::vector<int8_t> output(features.size());
  for (std::size_t index = 0; index < features.size(); ++index) {
    require(std::isfinite(features[index]), "feature tensor must be finite");
    const double scaled = static_cast<double>(features[index]) * input_scale;
    const double rounded = std::copysign(std::floor(std::abs(scaled) + 0.5), scaled);
    const double clamped = std::max(-128.0, std::min(127.0, rounded));
    output[index] = static_cast<int8_t>(clamped);
  }
  return output;
}

}  // namespace soundjie

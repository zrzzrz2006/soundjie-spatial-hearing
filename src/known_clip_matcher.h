#ifndef SOUNDJIE_KNOWN_CLIP_MATCHER_H
#define SOUNDJIE_KNOWN_CLIP_MATCHER_H
#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>
namespace soundjie {
using ClipFeature=std::array<float,32>;
struct ClipCue { std::string id,speaker,text; double start=0,end=0; };
struct ClipReference {
  std::string id,profile;
  std::vector<ClipFeature> features;
  std::vector<ClipCue> cues;
};
struct ClipMatch {
  bool matched=false; double position_seconds=0,score=0,margin=0;
  std::string cue_id,speaker,text;
  // Empty for the legacy matcher and for every rejected bank match.
  std::string clip_id,scene_profile;
};
ClipFeature clip_feature(const int16_t* pcm);
std::vector<ClipFeature> clip_features(const std::vector<int16_t>& pcm,std::size_t hop=320);
void write_clip_reference(const std::string&,const std::vector<ClipFeature>&);
std::vector<ClipFeature> read_clip_reference(const std::string&);
std::vector<ClipCue> read_clip_cues(const std::string&,double duration);
// 1..4 rows: id TAB profile TAB relative.features TAB relative.cues.tsv.
// Assets must resolve within the manifest directory; profiles are
// ambulance_traffic and dialogue. Throws on invalid or unbounded input.
std::vector<ClipReference> read_clip_bank(const std::string& path);
std::string json_escape_clip(const std::string&);
class KnownClipMatcher {
 public:
  KnownClipMatcher(const std::vector<ClipFeature>& reference,const std::vector<ClipCue>& cues,
                   double threshold=.65,double margin=.05);
  void push(int16_t value);
  void reset();
  ClipMatch match();
 private:
  std::vector<ClipFeature> ref_; std::vector<ClipCue> cues_;
  std::vector<double> reference_norm_;
  std::deque<int16_t> pcm_; uint64_t sample_count_=0; double threshold_,margin_;
  bool previous_matched_=false;
  unsigned consistent_matches_=0;
  uint64_t previous_sample_=0;
  double previous_position_=0;
};
class KnownClipBank {
 public:
  KnownClipBank(const std::vector<ClipReference>& references,
                double threshold=.65,double margin=.05,double cross_clip_margin=.05);
  void push(int16_t value);
  void reset();
  ClipMatch match();
 private:
  void clear_history();
  std::vector<ClipReference> refs_;
  std::vector<std::vector<double>> reference_norms_;
  std::deque<int16_t> pcm_;
  uint64_t sample_count_=0;
  double threshold_,margin_,cross_clip_margin_;
  std::string previous_clip_;
  bool previous_matched_=false;
  unsigned consistent_matches_=0;
  uint64_t previous_sample_=0;
  double previous_position_=0;
};
}
#endif

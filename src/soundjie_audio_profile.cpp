// Read-only CPU stage benchmark. Does not load DPU, write framebuffer or capture audio.
#include "logmel_frontend.h"
#include "sha256.h"
#include "known_clip_matcher.h"
#include "traffic_fine_model.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
int main(int argc,char**argv){try{
  if(argc!=7)throw std::runtime_error("usage: profile pcm config model reference cues traffic");
  std::ifstream in(argv[1],std::ios::binary);std::vector<int16_t> pcm(16000);
  in.read(reinterpret_cast<char*>(pcm.data()),32000);if(in.gcount()!=32000)throw std::runtime_error("PCM incomplete");
  std::vector<double> samples;for(auto s:pcm)samples.push_back(s/32768.0);
  auto ref=soundjie::read_clip_reference(argv[4]);
  auto cues=soundjie::read_clip_cues(argv[5],((ref.size()-1)*80+512)/16000.0);
  soundjie::KnownClipMatcher matcher(ref,cues);
  soundjie::TrafficFineModel traffic(argv[6]);
  for(int n=0;n<5;++n){
    auto a=Clock::now();auto hash=soundjie::sha256_file(argv[3]);auto b=Clock::now();
    auto config=soundjie::load_frontend_config(argv[2]);auto c=Clock::now();
    auto features=soundjie::compute_logmel(samples,config);auto d=Clock::now();
    auto fine=traffic.predict(pcm);auto e=Clock::now();
    for(auto s:pcm)matcher.push(s);auto f=Clock::now();
    auto clip=matcher.match();auto g=Clock::now();
    std::cout<<std::setprecision(9)<<"{\"trial\":"<<n<<",\"hash_ms\":"<<elapsed(a,b)
      <<",\"config_ms\":"<<elapsed(b,c)<<",\"logmel_ms\":"<<elapsed(c,d)
      <<",\"fine_ms\":"<<elapsed(d,e)<<",\"clip_push_ms\":"<<elapsed(e,f)<<",\"clip_match_ms\":"<<elapsed(f,g)
      <<",\"feature_first\":"<<features[0]<<",\"fine_first\":"<<fine[0]<<",\"clip_score\":"<<clip.score
      <<",\"model_sha256\":\""<<hash<<"\"}\n";
  }
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}

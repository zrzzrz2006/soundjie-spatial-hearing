#include "known_clip_matcher.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv){try{
  std::map<std::string,std::string> args;for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing argument");args[argv[i]]=argv[i+1];}
  if(args.count("--make-reference")){
    std::ifstream in(args.at("--make-reference").c_str(),std::ios::binary);if(!in)throw std::runtime_error("PCM missing");std::vector<int16_t>x;char b[2];
    while(in.read(b,2)){x.push_back(static_cast<int16_t>(static_cast<unsigned char>(b[0])|(static_cast<unsigned char>(b[1])<<8)));if(x.size()>16000*120)throw std::runtime_error("PCM too long");}
    if(in.gcount())throw std::runtime_error("odd PCM bytes");
    soundjie::write_clip_reference(args.at("--output"),soundjie::clip_features(x,80));return 0;
  }
  auto ref=soundjie::read_clip_reference(args.at("--reference"));auto cues=soundjie::read_clip_cues(args.at("--cues"),((ref.size()-1)*80+512)/16000.0);
  soundjie::KnownClipMatcher matcher(ref,cues,args.count("--threshold")?std::stod(args.at("--threshold")):.65,args.count("--margin")?std::stod(args.at("--margin")):.05);
  std::ifstream in(args.at("--pcm").c_str(),std::ios::binary);if(!in)throw std::runtime_error("PCM missing");char b[2];uint64_t count=0;
  while(in.read(b,2)){
    matcher.push(static_cast<int16_t>(static_cast<unsigned char>(b[0])|(static_cast<unsigned char>(b[1])<<8)));++count;
    if(count%3200)continue;
    auto r=matcher.match();
    std::cout<<std::setprecision(9)<<"{\"input_seconds\":"<<count/16000.0<<",\"matched\":"<<(r.matched?"true":"false")<<",\"position_seconds\":"<<r.position_seconds<<",\"score\":"<<r.score<<",\"margin\":"<<r.margin<<",\"cue_id\":\""<<soundjie::json_escape_clip(r.cue_id)<<"\",\"speaker\":\""<<soundjie::json_escape_clip(r.speaker)<<"\",\"text\":\""<<soundjie::json_escape_clip(r.text)<<"\"}\n";
  }if(in.gcount())throw std::runtime_error("odd PCM bytes");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}

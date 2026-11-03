#include "known_clip_matcher.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace {
using Clock=std::chrono::steady_clock;
double number(const std::string& text){
  std::size_t used=0;const double value=std::stod(text,&used);
  if(used!=text.size()||!std::isfinite(value))throw std::runtime_error("invalid numeric option");
  return value;
}
int16_t sample(const char* bytes){
  return static_cast<int16_t>(static_cast<unsigned char>(bytes[0])|(static_cast<unsigned char>(bytes[1])<<8));
}
void check_read(const std::ifstream& input){
  if(input.bad()||!input.eof())throw std::runtime_error("PCM read failed");
  if(input.gcount())throw std::runtime_error("odd PCM bytes");
}
}

int main(int argc,char** argv){try{
  if(argc==2&&std::string(argv[1])=="--help"){
    std::cout<<"Prepare: known_clip_bank_cli --make-reference mono16k.s16le --output reference.features\n"
        <<"Run: known_clip_bank_cli --bank bank.tsv --pcm mono16k.s16le "
        <<"[--threshold 0.65] [--margin 0.05] [--cross-clip-margin 0.05]\n"
        <<"Bank rows: clip_id TAB profile TAB relative.features TAB relative.cues.tsv\n"
        <<"Profiles: ambulance_traffic, dialogue. JSON every 3200 samples (0.2 s).\n";
    return 0;
  }
  std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2){
    if(i+1>=argc)throw std::runtime_error("missing argument value");
    if(!args.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("duplicate argument");
  }
  const bool prepare=args.count("--make-reference")!=0;
  const std::set<std::string> allowed=prepare?std::set<std::string>{"--make-reference","--output"}:
      std::set<std::string>{"--bank","--pcm","--threshold","--margin","--cross-clip-margin"};
  for(const auto& arg:args)if(!allowed.count(arg.first))throw std::runtime_error("unknown or incompatible argument: "+arg.first);
  if(prepare){
    if(!args.count("--output"))throw std::runtime_error("--output required");
    std::ifstream input(args.at("--make-reference").c_str(),std::ios::binary);
    if(!input)throw std::runtime_error("PCM missing");
    std::vector<int16_t> pcm;char bytes[2];
    while(input.read(bytes,2)){
      pcm.push_back(sample(bytes));
      if(pcm.size()>16000*120)throw std::runtime_error("PCM longer than 120 seconds");
    }
    check_read(input);
    soundjie::write_clip_reference(args.at("--output"),soundjie::clip_features(pcm,80));
    return 0;
  }
  if(!args.count("--bank")||!args.count("--pcm"))throw std::runtime_error("--bank and --pcm required (see --help)");
  soundjie::KnownClipBank bank(soundjie::read_clip_bank(args.at("--bank")),
      args.count("--threshold")?number(args.at("--threshold")):.65,
      args.count("--margin")?number(args.at("--margin")):.05,
      args.count("--cross-clip-margin")?number(args.at("--cross-clip-margin")):.05);
  std::ifstream input(args.at("--pcm").c_str(),std::ios::binary);
  if(!input)throw std::runtime_error("PCM missing");
  char bytes[2];uint64_t count=0;
  while(input.read(bytes,2)){
    bank.push(sample(bytes));++count;
    if(count%3200)continue;
    const auto start=Clock::now();const auto result=bank.match();
    const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    std::cout<<std::setprecision(9)
        <<"{\"input_seconds\":"<<count/16000.0
        <<",\"matched\":"<<(result.matched?"true":"false")
        <<",\"position_seconds\":"<<result.position_seconds
        <<",\"score\":"<<result.score<<",\"margin\":"<<result.margin
        <<",\"clip_id\":\""<<soundjie::json_escape_clip(result.clip_id)
        <<"\",\"scene_profile\":\""<<soundjie::json_escape_clip(result.scene_profile)
        <<"\",\"profile\":\""<<soundjie::json_escape_clip(result.scene_profile)
        <<"\",\"cue_id\":\""<<soundjie::json_escape_clip(result.cue_id)
        <<"\",\"speaker\":\""<<soundjie::json_escape_clip(result.speaker)
        <<"\",\"text\":\""<<soundjie::json_escape_clip(result.text)
        <<"\",\"caption\":\""<<soundjie::json_escape_clip(result.text)
        <<"\",\"match_ms\":"<<elapsed<<"}\n";
  }
  check_read(input);
  if(!std::cout)throw std::runtime_error("JSON output failed");
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}

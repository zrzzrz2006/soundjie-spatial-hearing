#include "traffic_fine_model.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
int main(int argc,char**argv){try{
  std::map<std::string,std::string>a;for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing argument");a[argv[i]]=argv[i+1];}
  soundjie::TrafficFineModel model(a.at("--model"));std::ifstream f(a.at("--pcm").c_str(),std::ios::binary);
  if(!f)throw std::runtime_error("PCM missing");
  std::vector<int16_t>x;char b[2];
  while(f.read(b,2)){x.push_back(static_cast<int16_t>(static_cast<unsigned char>(b[0])|(static_cast<unsigned char>(b[1])<<8)));if(x.size()>160000)throw std::runtime_error("PCM too long");}
  if(f.gcount())throw std::runtime_error("odd PCM bytes");
  const auto features=soundjie::traffic_features(x);const auto scores=model.predict(x);
  std::cout<<std::setprecision(9)<<"{\"features\":[";
  for(int j=0;j<96;++j)std::cout<<(j?",":"")<<features[j];
  std::cout<<"],\"scores\":[";
  for(int j=0;j<3;++j)std::cout<<(j?",":"")<<scores[j];
  std::cout<<"]}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}

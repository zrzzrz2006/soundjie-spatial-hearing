#include "traffic_fine_model.h"
#include "known_clip_matcher.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace soundjie {
std::array<float,96> traffic_features(const std::vector<int16_t>& pcm){
  if(pcm.size()<512||pcm.size()>160000)throw std::runtime_error("traffic PCM outside bounds");
  const auto frames=clip_features(pcm);std::array<double,32> mean{},variance{},delta{};
  for(const auto& f:frames)for(int j=0;j<32;++j)mean[j]+=f[j]/static_cast<double>(frames.size());
  double level=0;for(double v:mean)level+=v/32;
  for(std::size_t i=0;i<frames.size();++i)for(int j=0;j<32;++j){
    const double d=frames[i][j]-mean[j];variance[j]+=d*d/frames.size();
    if(i)delta[j]+=std::abs(static_cast<double>(frames[i][j])-frames[i-1][j])/(frames.size()-1);
  }
  std::array<float,96> out;
  for(int j=0;j<32;++j){out[j]=mean[j]-level;out[32+j]=std::sqrt(variance[j]);out[64+j]=delta[j];}
  return out;
}
TrafficFineModel::TrafficFineModel(const std::string& path){
  const std::size_t count=96*2+96*48+48+48*3+3;
  std::ifstream f(path.c_str(),std::ios::binary|std::ios::ate);
  if(!f||f.tellg()!=static_cast<std::streamoff>(20+count*4))throw std::runtime_error("traffic model size invalid");
  f.seekg(0);char magic[8];uint32_t dims[3];f.read(magic,8);f.read(reinterpret_cast<char*>(dims),12);
  if(std::memcmp(magic,"SJTF1\0\0\0",8)||dims[0]!=96||dims[1]!=48||dims[2]!=3)throw std::runtime_error("traffic model header invalid");
  parameters_.resize(count);f.read(reinterpret_cast<char*>(parameters_.data()),count*4);
  if(!f)throw std::runtime_error("traffic model truncated");
  for(float v:parameters_)if(!std::isfinite(v)||std::abs(v)>10000)throw std::runtime_error("traffic model nonfinite or unbounded");
  for(int j=96;j<192;++j)if(parameters_[j]<.00001)throw std::runtime_error("traffic scale invalid");
}
std::array<float,3> TrafficFineModel::predict(const std::vector<int16_t>& pcm)const{
  auto x=traffic_features(pcm);const auto& p=parameters_;
  for(int j=0;j<96;++j)x[j]=(x[j]-p[j])/p[96+j];
  std::array<float,48> h{};std::array<float,3> out{};
  for(int i=0;i<48;++i){double s=p[192+96*48+i];for(int j=0;j<96;++j)s+=p[192+i*96+j]*x[j];h[i]=std::max(0.0,s);}
  for(int i=0;i<3;++i){double s=p[192+96*48+48+48*3+i];for(int j=0;j<48;++j)s+=p[192+96*48+48+i*48+j]*h[j];out[i]=1/(1+std::exp(-std::max(-60.0,std::min(60.0,s))));}
  double sum=0;for(int16_t v:pcm)sum+=static_cast<double>(v)*v;
  if(sum/pcm.size()<100)out.fill(0);
  return out;
}
}

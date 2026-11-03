#include "known_clip_matcher.h"
#include "fft_radix2.h"
#include "utf8_text.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
namespace soundjie {
namespace {
const std::size_t kQuery=75,kHop=320,kLength=512,kRefHop=80,kStride=4,kSpan=(kQuery-1)*kStride+1;
const std::size_t kMaxClips=4,kMaxFrames=24000,kMaxBankBytes=4096,kMaxCueBytes=65536;
bool finite(double x){return std::isfinite(x);}
double strict_number(const std::string& s){std::size_t used=0; double x=std::stod(s,&used); if(used!=s.size()||!finite(x))throw std::runtime_error("invalid cue time");return x;}
bool valid_setting(double value){return finite(value)&&value>=0&&value<=1;}
void validate_text(const std::string& value,std::size_t limit){
  if(value.empty()||value.size()>limit)throw std::runtime_error("bank text size invalid");
  for(unsigned char c:value)if(c<32||c==127)throw std::runtime_error("bank control character");
  decode_utf8_text(value);
}
void validate_references(const std::vector<ClipReference>& refs){
  if(refs.empty()||refs.size()>kMaxClips)throw std::runtime_error("bank requires 1..4 clips");
  std::set<std::string> ids;
  for(const auto& ref:refs){
    validate_text(ref.id,64);
    if(!ids.insert(ref.id).second)throw std::runtime_error("duplicate clip id");
    if(ref.profile!="ambulance_traffic"&&ref.profile!="dialogue")throw std::runtime_error("unknown clip profile");
    if(ref.features.size()<kSpan||ref.features.size()>kMaxFrames)throw std::runtime_error("bank reference size invalid");
    for(const auto& feature:ref.features)for(float x:feature)
      if(!finite(x)||std::abs(x)>100)throw std::runtime_error("bank reference feature invalid");
    const double duration=((ref.features.size()-1)*kRefHop+kLength)/16000.0;
    if(ref.cues.size()>100)throw std::runtime_error("bank cue count invalid");
    std::set<std::string> cue_ids;double last_end=0;
    for(const auto& cue:ref.cues){
      validate_text(cue.id,32);validate_text(cue.text,256);
      if(!cue_ids.insert(cue.id).second||!finite(cue.start)||!finite(cue.end)||
          cue.start<last_end||cue.end<=cue.start||cue.end>duration||
          (cue.speaker!="A"&&cue.speaker!="B"&&cue.speaker!="C"&&cue.speaker!="?"))
        throw std::runtime_error("bank cue contract invalid");
      last_end=cue.end;
    }
  }
}
void bounded_regular_file(const std::string& path,std::size_t maximum){
  struct stat info;
  if(stat(path.c_str(),&info)!=0||!S_ISREG(info.st_mode)||info.st_size<0||
      static_cast<uint64_t>(info.st_size)>maximum)throw std::runtime_error("bank asset missing, nonregular or oversized");
}
std::string canonical_path(const std::string& path){
  if(path.find('\0')!=std::string::npos)throw std::runtime_error("bank path contains NUL");
  char resolved[PATH_MAX];
  if(!realpath(path.c_str(),resolved))throw std::runtime_error("bank path cannot resolve");
  return resolved;
}
std::string bank_asset(const std::string& directory,const std::string& relative,std::size_t maximum){
  if(relative.empty()||relative.size()>256)throw std::runtime_error("bank relative path size invalid");
  for(unsigned char c:relative)if(c<32||c==127||c=='\\'||c==':')throw std::runtime_error("bank relative path invalid");
  std::size_t start=0;
  while(start<=relative.size()){
    const auto slash=relative.find('/',start);
    const auto part=relative.substr(start,slash==std::string::npos?slash:slash-start);
    if(part.empty()||part=="."||part=="..")throw std::runtime_error("bank path traversal invalid");
    if(slash==std::string::npos)break;
    start=slash+1;
  }
  const auto prefix=directory=="/"?directory:directory+"/";
  const auto path=canonical_path(prefix+relative);
  if(path.compare(0,prefix.size(),prefix)!=0)throw std::runtime_error("bank asset escapes manifest directory");
  bounded_regular_file(path,maximum);
  return path;
}
std::vector<double> reference_norms(const std::vector<ClipFeature>& ref){
  // Immutable reference norms; retain the legacy accumulation order.
  std::vector<double> norms;norms.reserve(ref.size()-kSpan+1);
  for(std::size_t start=0;start+kSpan<=ref.size();++start){
    std::array<double,32> sum{};double r2=0;
    for(std::size_t i=0;i<kQuery;++i)for(int d=0;d<32;++d){const double v=ref[start+i*kStride][d];sum[d]+=v;r2+=v*v;}
    for(int d=0;d<32;++d)r2-=sum[d]*sum[d]/kQuery;
    norms.push_back(r2);
  }
  return norms;
}
bool prepare_query(const std::deque<int16_t>& pcm,uint64_t sample_count,std::vector<ClipFeature>& query,double& norm){
  const std::size_t needed=kLength+(kQuery-1)*kHop;
  if(pcm.size()<needed)return false;
  double recent=0;for(std::size_t i=pcm.size()-3200;i<pcm.size();++i)recent+=static_cast<double>(pcm[i])*pcm[i];
  if(recent/3200<100)return false;
  // Keep FFT frames on the acquisition grid, including non-hop-aligned calls.
  const std::size_t trim=(sample_count-kLength)%kHop;
  if(pcm.size()<needed+trim)return false;
  const auto end=pcm.end()-trim;
  query=clip_features(std::vector<int16_t>(end-needed,end));
  std::array<double,32> mean{};
  for(const auto& f:query)for(int d=0;d<32;++d)mean[d]+=f[d]/query.size();
  norm=0;
  for(auto& f:query)for(int d=0;d<32;++d){f[d]-=mean[d];norm+=f[d]*f[d];}
  return norm>=1e-6;
}
ClipMatch search_reference(const std::vector<ClipFeature>& ref,const std::vector<double>& norms,
                           const std::vector<ClipFeature>& query,double query_norm){
  std::vector<double> scores;scores.reserve(norms.size());
  // Search EVERY position of EVERY reference on every call. Restricting this
  // to a tracking window would hide jumps, switches and cross-clip ambiguity.
  for(std::size_t start=0;start<norms.size();++start){
    float accum[4]={0,0,0,0};
#if defined(__ARM_NEON)
    float32x4_t sum=vdupq_n_f32(0);
    for(std::size_t i=0;i<query.size();++i)for(int d=0;d<32;d+=4)
      sum=vmlaq_f32(sum,vld1q_f32(ref[start+i*kStride].data()+d),vld1q_f32(query[i].data()+d));
    vst1q_f32(accum,sum);
#else
    for(std::size_t i=0;i<query.size();++i)for(int d=0;d<32;d+=4)
      for(int k=0;k<4;++k)accum[k]+=ref[start+i*kStride][d+k]*query[i][d+k];
#endif
    // Do not use -ffast-math: non-finite validation and thresholds matter.
    const double cross=static_cast<double>(accum[0])+accum[1]+accum[2]+accum[3];
    const double r2=norms[start];
    scores.push_back(r2>1e-6?std::max(-1.0,std::min(1.0,cross/std::sqrt(r2*query_norm))):-1);
  }
  const auto best=std::max_element(scores.begin(),scores.end());
  const std::size_t index=best-scores.begin();double second=-1;
  for(std::size_t i=0;i<scores.size();++i)if(i+100<index||i>index+100)second=std::max(second,scores[i]);
  ClipMatch out;out.score=*best;out.margin=*best-second;
  out.position_seconds=(index*kRefHop+kLength+(kQuery-1)*kHop)/16000.0;
  return out;
}
double exact_tail_score(const std::vector<ClipFeature>& ref,const std::vector<double>& norms,
                     const std::vector<ClipFeature>& query,double query_norm,std::size_t start){
  float accum[4]={0,0,0,0};
#if defined(__ARM_NEON)
  float32x4_t sum=vdupq_n_f32(0);
  for(std::size_t i=0;i<query.size();++i)for(int d=0;d<32;d+=4)
    sum=vmlaq_f32(sum,vld1q_f32(ref[start+i*kStride].data()+d),vld1q_f32(query[i].data()+d));
  vst1q_f32(accum,sum);
#else
  for(std::size_t i=0;i<query.size();++i)for(int d=0;d<32;d+=4)
    for(int k=0;k<4;++k)accum[k]+=ref[start+i*kStride][d+k]*query[i][d+k];
#endif
  const double cross=static_cast<double>(accum[0])+accum[1]+accum[2]+accum[3];
  const double r2=norms[start];
  return r2>1e-6?std::max(-1.0,std::min(1.0,cross/std::sqrt(r2*query_norm))):-1;
}

// Exact bank-only interleave: every reference start is evaluated. Each of the
// four independent accumulators retains the original i,d/lane addition order.
ClipMatch search_bank_reference_exact(const std::vector<ClipFeature>& ref,const std::vector<double>& norms,
                                     const std::vector<ClipFeature>& query,double query_norm){
  std::vector<double> scores;scores.reserve(norms.size());
  std::size_t start=0;
  for(;start+4<=norms.size();start+=4){
    float accum[4][4]={{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
#if defined(__ARM_NEON)
    float32x4_t sum0=vdupq_n_f32(0),sum1=vdupq_n_f32(0),sum2=vdupq_n_f32(0),sum3=vdupq_n_f32(0);
    for(std::size_t i=0;i<query.size();++i)for(int d=0;d<32;d+=4){
      const float32x4_t q=vld1q_f32(query[i].data()+d);
      sum0=vmlaq_f32(sum0,vld1q_f32(ref[start+0+i*kStride].data()+d),q);
      sum1=vmlaq_f32(sum1,vld1q_f32(ref[start+1+i*kStride].data()+d),q);
      sum2=vmlaq_f32(sum2,vld1q_f32(ref[start+2+i*kStride].data()+d),q);
      sum3=vmlaq_f32(sum3,vld1q_f32(ref[start+3+i*kStride].data()+d),q);
    }
    vst1q_f32(accum[0],sum0);vst1q_f32(accum[1],sum1);
    vst1q_f32(accum[2],sum2);vst1q_f32(accum[3],sum3);
#else
    for(std::size_t i=0;i<query.size();++i)for(int d=0;d<32;d+=4)for(int k=0;k<4;++k){
      const float q=query[i][d+k];
      accum[0][k]+=ref[start+0+i*kStride][d+k]*q;
      accum[1][k]+=ref[start+1+i*kStride][d+k]*q;
      accum[2][k]+=ref[start+2+i*kStride][d+k]*q;
      accum[3][k]+=ref[start+3+i*kStride][d+k]*q;
    }
#endif
    for(unsigned p=0;p<4;++p){
      const double cross=static_cast<double>(accum[p][0])+accum[p][1]+accum[p][2]+accum[p][3];
      const double r2=norms[start+p];
      scores.push_back(r2>1e-6?std::max(-1.0,std::min(1.0,cross/std::sqrt(r2*query_norm))):-1);
    }
  }
  for(;start<norms.size();++start)scores.push_back(exact_tail_score(ref,norms,query,query_norm,start));
  // EXACT_AUDIT_INSERT
  const std::size_t index=std::max_element(scores.begin(),scores.end())-scores.begin();double second=-1;
  for(std::size_t i=0;i<scores.size();++i)if(i+100<index||i>index+100)second=std::max(second,scores[i]);
  ClipMatch out;out.score=scores[index];out.margin=out.score-second;
  out.position_seconds=(index*kRefHop+kLength+(kQuery-1)*kHop)/16000.0;
  return out;
}

void accept_evidence(ClipMatch& out,uint64_t sample_count,double threshold,double margin,
                     bool& previous_matched,unsigned& consistent_matches,uint64_t& previous_sample,double& previous_position){
  const bool strong=out.score>=threshold&&out.margin>=margin;
  const double step=sample_count>=previous_sample?(sample_count-previous_sample)/16000.0:-1;
  const bool continuous=previous_matched&&step>=0&&step<=1.25&&
      std::abs(out.position_seconds-previous_position-step)<=.10;
  // Weak continuation needs two earlier matches AND current acoustic evidence.
  const bool follow=consistent_matches>=2&&continuous&&
      out.score>=std::max(.45,threshold-.20)&&out.margin>=std::max(.10,margin);
  out.matched=strong||follow;
  if(out.matched){
    consistent_matches=continuous?std::min(2u,consistent_matches+(step>0?1u:0u)):1u;
    previous_matched=true;previous_sample=sample_count;previous_position=out.position_seconds;
  }else{previous_matched=false;consistent_matches=0;}
}
void attach_cue(ClipMatch& out,const std::vector<ClipCue>& cues){
  if(out.matched)for(const auto& c:cues)if(out.position_seconds>=c.start+.3&&out.position_seconds<=c.end+.4){
    out.cue_id=c.id;out.speaker=c.speaker;out.text=c.text;break;
  }
}
}
ClipFeature clip_feature(const int16_t* pcm){
  std::array<double,512> frame;
  for(std::size_t i=0;i<512;++i)frame[i]=pcm[i]/32768.0*(.5-.5*std::cos(2*3.141592653589793*i/511));
  const auto fft=fft_radix2_512(frame); ClipFeature out;
  int left=3;
  for(int j=0;j<32;++j){
    const int right=std::max(left+1,static_cast<int>(std::lround(3*std::pow(224.0/3,(j+1)/32.0))));
    double energy=0;for(int k=left;k<right;++k)energy+=std::norm(fft[k]);
    out[j]=static_cast<float>(std::log(std::max(energy/(right-left),1e-12)));left=right;
  }
  return out;
}
std::vector<ClipFeature> clip_features(const std::vector<int16_t>& pcm,std::size_t hop){
  if(hop!=80&&hop!=320)throw std::runtime_error("feature hop invalid");
  std::vector<ClipFeature> out;
  for(std::size_t i=0;i+512<=pcm.size();i+=hop)out.push_back(clip_feature(pcm.data()+i));
  return out;
}
void write_clip_reference(const std::string& path,const std::vector<ClipFeature>& f){
  if(f.size()<kSpan||f.size()>24000)throw std::runtime_error("reference duration outside bounds");
  const int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)throw std::runtime_error("output exists or cannot create");
  std::string bytes("SJTVF2\0\0",8);uint32_t n=static_cast<uint32_t>(f.size()),d=32;
  bytes.append(reinterpret_cast<const char*>(&n),4);bytes.append(reinterpret_cast<const char*>(&d),4);
  for(const auto& row:f)bytes.append(reinterpret_cast<const char*>(row.data()),128);
  std::size_t offset=0;
  while(offset<bytes.size()){const auto count=write(fd,bytes.data()+offset,bytes.size()-offset);if(count<=0){close(fd);throw std::runtime_error("reference write failed");}offset+=count;}
  if(close(fd)!=0)throw std::runtime_error("reference close failed");
}
std::vector<ClipFeature> read_clip_reference(const std::string& path){
  std::ifstream f(path.c_str(),std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("reference missing");
  const auto size=f.tellg();if(size<16||size>16+24000*128)throw std::runtime_error("reference size invalid");f.seekg(0);
  char magic[8];uint32_t n,d;f.read(magic,8);f.read(reinterpret_cast<char*>(&n),4);f.read(reinterpret_cast<char*>(&d),4);
  if(std::memcmp(magic,"SJTVF2\0\0",8)||d!=32||n<kSpan||n>24000||size!=16+n*128)throw std::runtime_error("reference header invalid");
  std::vector<ClipFeature> out(n);
  for(auto& row:out){f.read(reinterpret_cast<char*>(row.data()),128);for(float x:row)if(!finite(x)||std::abs(x)>100)throw std::runtime_error("reference feature invalid");}
  if(!f)throw std::runtime_error("truncated reference");
  return out;
}
std::vector<ClipCue> read_clip_cues(const std::string& path,double duration){
  std::ifstream f(path.c_str());if(!f)throw std::runtime_error("cues missing");std::vector<ClipCue> out;std::string line;std::set<std::string> ids;
  while(std::getline(f,line)){
    if(!line.empty()&&line.back()=='\r')line.pop_back();
    if(line.empty())continue;
    if(line.size()>512||out.size()>=100)throw std::runtime_error("cue size invalid");
    std::vector<std::string> fields;std::size_t p=0;
    for(int i=0;i<4;++i){auto q=line.find('\t',p);if(q==std::string::npos)throw std::runtime_error("cue fields missing");fields.push_back(line.substr(p,q-p));p=q+1;}fields.push_back(line.substr(p));
    ClipCue c;c.id=fields[0];c.start=strict_number(fields[1]);c.end=strict_number(fields[2]);c.speaker=fields[3];c.text=fields[4];
    if(c.id.empty()||c.id.size()>32||!ids.insert(c.id).second||c.start<0||c.end<=c.start||c.end>duration||(!out.empty()&&c.start<out.back().end)||c.text.empty()||c.text.size()>256||(c.speaker!="A"&&c.speaker!="B"&&c.speaker!="C"&&c.speaker!="?"))throw std::runtime_error("cue contract invalid");
    for(unsigned char ch:c.text)if(ch<32)throw std::runtime_error("cue control character");
    decode_utf8_text(c.text);decode_utf8_text(c.id);
    out.push_back(c);
  }return out;
}
std::vector<ClipReference> read_clip_bank(const std::string& path){
  const auto manifest=canonical_path(path);
  bounded_regular_file(manifest,kMaxBankBytes);
  const auto slash=manifest.find_last_of('/');
  const auto directory=slash==0?std::string("/"):manifest.substr(0,slash);
  std::ifstream file(manifest.c_str(),std::ios::binary);
  if(!file)throw std::runtime_error("bank missing");
  std::vector<ClipReference> refs;std::string line;std::size_t bytes=0;
  while(std::getline(file,line)){
    bytes+=line.size()+1;
    if(bytes>kMaxBankBytes+1||line.size()>1024)throw std::runtime_error("bank size invalid");
    if(!line.empty()&&line.back()=='\r')line.pop_back();
    if(line.empty())continue;
    if(refs.size()>=kMaxClips)throw std::runtime_error("too many bank clips");
    std::vector<std::string> fields;std::size_t start=0;
    for(int i=0;i<3;++i){const auto end=line.find('\t',start);if(end==std::string::npos)throw std::runtime_error("bank fields missing");fields.push_back(line.substr(start,end-start));start=end+1;}
    if(line.find('\t',start)!=std::string::npos)throw std::runtime_error("extra bank fields");
    fields.push_back(line.substr(start));
    ClipReference ref;ref.id=fields[0];ref.profile=fields[1];
    ref.features=read_clip_reference(bank_asset(directory,fields[2],16+kMaxFrames*128));
    ref.cues=read_clip_cues(bank_asset(directory,fields[3],kMaxCueBytes),((ref.features.size()-1)*kRefHop+kLength)/16000.0);
    refs.push_back(std::move(ref));
  }
  if(!file.eof())throw std::runtime_error("bank read failed");
  validate_references(refs);
  return refs;
}
std::string json_escape_clip(const std::string& s){std::ostringstream o;for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<static_cast<int>(c);else o<<c;}return o.str();}
KnownClipMatcher::KnownClipMatcher(const std::vector<ClipFeature>& r,const std::vector<ClipCue>& c,double t,double m):ref_(r),cues_(c),threshold_(t),margin_(m){
  if(r.size()<kSpan||r.size()>24000||!finite(t)||!finite(m)||t<0||t>1||m<0||m>1)throw std::runtime_error("matcher configuration invalid");
  reference_norm_=reference_norms(ref_);
}
void KnownClipMatcher::push(int16_t v){++sample_count_;pcm_.push_back(v);if(pcm_.size()>kLength+kQuery*kHop-1)pcm_.pop_front();}
void KnownClipMatcher::reset(){pcm_.clear();sample_count_=0;previous_matched_=false;consistent_matches_=0;previous_sample_=0;previous_position_=0;}
ClipMatch KnownClipMatcher::match(){
  std::vector<ClipFeature> query;double norm=0;
  if(!prepare_query(pcm_,sample_count_,query,norm)){previous_matched_=false;consistent_matches_=0;return ClipMatch();}
  auto out=search_reference(ref_,reference_norm_,query,norm);
  accept_evidence(out,sample_count_,threshold_,margin_,previous_matched_,consistent_matches_,previous_sample_,previous_position_);
  attach_cue(out,cues_);
  return out;
}
KnownClipBank::KnownClipBank(const std::vector<ClipReference>& refs,double threshold,double margin,double cross_clip_margin)
    :threshold_(threshold),margin_(margin),cross_clip_margin_(cross_clip_margin){
  if(!valid_setting(threshold)||!valid_setting(margin)||!valid_setting(cross_clip_margin))
    throw std::runtime_error("bank thresholds invalid");
  validate_references(refs);
  refs_=refs;reference_norms_.reserve(refs_.size());
  for(const auto& ref:refs_)reference_norms_.push_back(reference_norms(ref.features));
}
void KnownClipBank::push(int16_t value){
  ++sample_count_;pcm_.push_back(value);
  if(pcm_.size()>kLength+kQuery*kHop-1)pcm_.pop_front();
}
void KnownClipBank::clear_history(){
  previous_clip_.clear();previous_matched_=false;consistent_matches_=0;previous_sample_=0;previous_position_=0;
}
void KnownClipBank::reset(){pcm_.clear();sample_count_=0;clear_history();}
ClipMatch KnownClipBank::match(){
  // One PCM buffer and one query FFT/centering pass for the whole bank.
  std::vector<ClipFeature> query;double norm=0;
  if(!prepare_query(pcm_,sample_count_,query,norm)){clear_history();return ClipMatch();}
  std::vector<ClipMatch> candidates;candidates.reserve(refs_.size());
  for(std::size_t i=0;i<refs_.size();++i)candidates.push_back(refs_.size()==1?search_reference(refs_[i].features,reference_norms_[i],query,norm):search_bank_reference_exact(refs_[i].features,reference_norms_[i],query,norm));
  std::size_t winner=0;
  for(std::size_t i=1;i<candidates.size();++i)if(candidates[i].score>candidates[winner].score)winner=i;
  auto out=candidates[winner];double runner_up=-1;
  for(std::size_t i=0;i<candidates.size();++i)if(i!=winner)runner_up=std::max(runner_up,candidates[i].score);
  // A rival's acoustic score counts even if its own temporal margin is poor.
  // Exact ties are ambiguous even with an explicitly zero cross-clip margin.
  if(candidates.size()>1&&(out.score<=runner_up||out.score-runner_up<cross_clip_margin_)){
    clear_history();return out;
  }
  if(previous_clip_!=refs_[winner].id)clear_history();
  accept_evidence(out,sample_count_,threshold_,margin_,previous_matched_,consistent_matches_,previous_sample_,previous_position_);
  if(out.matched){
    previous_clip_=refs_[winner].id;out.clip_id=refs_[winner].id;out.scene_profile=refs_[winner].profile;
    attach_cue(out,refs_[winner].cues);
  }else clear_history();
  return out;
}
}

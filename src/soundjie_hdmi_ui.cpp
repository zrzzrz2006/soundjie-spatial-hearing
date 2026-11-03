#include "framebuffer_surface.h"
#include "soundjie_d4_doa_stream.h"
#include "sha256.h"
#include "utf8_text.h"
#include "soundjie_class_contract.h"
#include "bounded_jsonl.h"
#include "soundjie_traffic_presentation.h"
#include <memory>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
struct Glyph { uint8_t width = 0; uint16_t rows[16]{}; };
struct State {
  std::string label="不可用", sector, doa_reason="unavailable", session_state="running", caption, speaker, class_contract="soundjie-legacy8-v1",clip_id,scene_profile;
  double angle=0,confidence=0,dbfs=-96,inference=0,latency=0,position=0,match_score=0,ambulance=0,road=0,speech=0,horn_start=0,horn_end=0;
  int class_id=-1;std::vector<double> scores;
  bool doa=false,usb=false,dpu=false,fb=false,tv=false,matched=false,fine=false,five=false,alert4=false,alert5=false,rejected=false;
  bool unified() const {return alert4||alert5;}
};
struct Options { std::string snapshot, atlas, offscreen, report, framebuffer, pixel, format, doa_jsonl, runtime_config, offscreen_dir, class_contract, jsonl; std::size_t log_max_bytes=0; uint32_t traffic_display_seed=0; int width=1280, height=720, frames=1, interval_ms=100, snapshot_stale_ms=2000; };
volatile std::sig_atomic_t stop_requested = 0;
void stop_handler(int) { stop_requested = 1; }
uint64_t monotonic_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

Options options(int argc, char** argv) {
  Options o;
  for (int i=1;i<argc;++i) { std::string k=argv[i]; if (++i>=argc) throw std::runtime_error("missing option value"); std::string v=argv[i];
    if(k=="--snapshot")o.snapshot=v; else if(k=="--atlas")o.atlas=v; else if(k=="--offscreen")o.offscreen=v;
    else if(k=="--report")o.report=v; else if(k=="--framebuffer")o.framebuffer=v; else if(k=="--pixel")o.pixel=v;
    else if(k=="--format")o.format=v; else if(k=="--width")o.width=std::stoi(v); else if(k=="--height")o.height=std::stoi(v);
    else if(k=="--doa-jsonl")o.doa_jsonl=v; else if(k=="--runtime-config")o.runtime_config=v;
    else if(k=="--jsonl")o.jsonl=v; else if(k=="--log-max-bytes")o.log_max_bytes=soundjie::jsonl_limit(v);
    else if(k=="--offscreen-dir")o.offscreen_dir=v; else if(k=="--frames")o.frames=std::stoi(v);
    else if(k=="--class-contract"){soundjie::physical_class_count(v);o.class_contract=v;}
    else if(k=="--traffic-display-seed"){
      std::size_t consumed=0;const auto seed=std::stoull(v,&consumed);
      if(consumed!=v.size()||seed==0||seed>UINT32_MAX)throw std::runtime_error("invalid traffic display seed");
      o.traffic_display_seed=static_cast<uint32_t>(seed);
    }
    else if(k=="--interval-ms")o.interval_ms=std::stoi(v); else if(k=="--snapshot-stale-ms")o.snapshot_stale_ms=std::stoi(v);
    else throw std::runtime_error("unknown option"); }
  if(o.width<1||o.width>1920||o.height<1||o.height>1080||o.frames<0||o.interval_ms<20||o.interval_ms>1000||o.snapshot_stale_ms<1)
    throw std::runtime_error("invalid geometry or refresh settings");
  return o;
}
std::string read_file(const std::string& path) { std::ifstream f(path.c_str(),std::ios::binary); if(!f)throw std::runtime_error("cannot read snapshot or atlas"); return {std::istreambuf_iterator<char>(f),{}}; }
std::string read_snapshot(const std::string& path) {
  const int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NONBLOCK|O_NOFOLLOW);
  struct stat info;
  if(fd<0)throw std::runtime_error("snapshot unavailable");
  if(fstat(fd,&info)!=0||!S_ISREG(info.st_mode)||info.st_size<2||info.st_size>65536){close(fd);throw std::runtime_error("snapshot size or type invalid");}
  std::string bytes(static_cast<std::size_t>(info.st_size),'\0');
  std::size_t offset=0;
  while(offset<bytes.size()){
    const ssize_t n=read(fd,&bytes[offset],bytes.size()-offset);
    if(n<0&&errno==EINTR)continue;
    if(n<=0){close(fd);throw std::runtime_error("snapshot read failed");}
    offset+=static_cast<std::size_t>(n);
  }
  close(fd);
  const auto end=bytes.find_last_not_of(" \r\n\t");
  if(bytes[0]!='{'||end==std::string::npos||bytes[end]!='}')throw std::runtime_error("snapshot incomplete");
  return bytes;
}
std::string string_field(const std::string& j,const std::string& key) {
  const std::string n="\""+key+"\":";auto p=j.find(n);if(p==std::string::npos)throw std::runtime_error("snapshot missing "+key);p+=n.size();
  while(p<j.size()&&j[p]==' ')++p;
  if(j.compare(p,4,"null")==0)return {};
  if(p>=j.size()||j[p++]!='"')throw std::runtime_error("snapshot string invalid");
  std::string out;
  while(p<j.size()){
    const unsigned char c=j[p++];if(c=='"')return out;
    if(c<32)throw std::runtime_error("snapshot string control character");
    if(c!='\\'){out+=c;continue;}
    if(p>=j.size())break;
    const char e=j[p++];
    if(e=='"'||e=='\\'||e=='/')out+=e;
    else if(e=='n')out+='\n';else if(e=='r')out+='\r';else if(e=='t')out+='\t';else if(e=='b')out+='\b';else if(e=='f')out+='\f';
    else if(e=='u'){
      if(p+4>j.size())throw std::runtime_error("truncated unicode escape");
      unsigned cp=0;for(int k=0;k<4;++k){const char h=j[p++];cp*=16;if(h>='0'&&h<='9')cp+=h-'0';else if(h>='A'&&h<='F')cp+=h-'A'+10;else if(h>='a'&&h<='f')cp+=h-'a'+10;else throw std::runtime_error("unicode escape invalid");}
      if(cp>=0xd800&&cp<=0xdfff)throw std::runtime_error("surrogate escape unsupported");
      if(cp<128)out+=static_cast<char>(cp);else if(cp<2048){out+=static_cast<char>(0xc0|(cp>>6));out+=static_cast<char>(0x80|(cp&63));}
      else{out+=static_cast<char>(0xe0|(cp>>12));out+=static_cast<char>(0x80|((cp>>6)&63));out+=static_cast<char>(0x80|(cp&63));}
    }else throw std::runtime_error("JSON escape invalid");
    if(out.size()>4096)throw std::runtime_error("snapshot string too long");
  }throw std::runtime_error("unterminated snapshot string");
}
std::string quoted(const std::string& s){std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<static_cast<unsigned>(c);else o<<c;}o<<'"';return o.str();}
double number_field(const std::string& j,const std::string& key) { std::string n="\""+key+"\":"; auto p=j.find(n); if(p==std::string::npos)throw std::runtime_error("snapshot missing "+key); p+=n.size(); std::size_t used=0; double v=std::stod(j.substr(p),&used); if(!std::isfinite(v))throw std::runtime_error("snapshot number nonfinite"); return v; }
bool bool_field(const std::string& j,const std::string& key) { std::string n="\""+key+"\":"; auto p=j.find(n); if(p==std::string::npos)throw std::runtime_error("snapshot missing "+key); p+=n.size(); while(p<j.size()&&j[p]==' ')++p; return j.compare(p,4,"true")==0; }
std::vector<double> class_scores(const std::string& j,unsigned count) {
  const std::string key="\"class_scores\":";auto p=j.find(key);
  if(p==std::string::npos)return {};
  p+=key.size();p=j.find_first_not_of(" \r\n\t",p);
  if(p==std::string::npos||j[p]!='[')throw std::runtime_error("invalid class scores");
  const auto end=j.find(']',++p);if(end==std::string::npos)throw std::runtime_error("unterminated class scores");
  std::istringstream in(j.substr(p,end-p));std::vector<double> scores;in>>std::ws;
  if(in.eof())return scores;
  while(true){double value;
    if(!(in>>value)||!std::isfinite(value)||value<0||value>1||scores.size()>=8)
      throw std::runtime_error("invalid class score");
    scores.push_back(value);in>>std::ws;if(in.eof())break;
    char separator;if(!in.get(separator)||separator!=',')throw std::runtime_error("invalid score delimiter");
  }
  if(scores.size()!=count)throw std::runtime_error("class score count does not match contract");
  return scores;
}
State parse_state(const std::string& j) { State s; s.label=string_field(j,"label_zh"); s.sector=string_field(j,"doa_sector"); s.doa_reason=string_field(j,"doa_reason"); s.confidence=number_field(j,"confidence"); s.dbfs=number_field(j,"dbfs"); s.inference=number_field(j,"inference_ms"); s.latency=number_field(j,"end_to_end_ms"); s.doa=bool_field(j,"doa_valid"); s.usb=bool_field(j,"usb_ok"); s.dpu=bool_field(j,"dpu_ok"); s.fb=bool_field(j,"framebuffer_ok"); if(s.doa){s.angle=number_field(j,"doa_deg");if(s.angle<0||s.angle>=360)throw std::runtime_error("invalid DoA angle");}
  if(j.find("\"class_contract\":")!=std::string::npos)s.class_contract=string_field(j,"class_contract");
  s.five=soundjie::siren5_contract(s.class_contract);
  s.alert4=soundjie::alert4_contract(s.class_contract);
  s.alert5=soundjie::alert5_contract(s.class_contract);
  const bool compact=s.five||s.unified();
  const unsigned count=soundjie::logical_class_count(s.class_contract);
  s.rejected=compact&&bool_field(j,"classification_rejected");
  const double id=number_field(j,"class_id");if(id<-1||id>count-1||id!=std::floor(id)||s.confidence<0||s.confidence>1)throw std::runtime_error("invalid classification");
  s.class_id=static_cast<int>(id);s.scores=class_scores(j,count);
  if(compact&&id>=0){
    if(s.scores.size()!=count)throw std::runtime_error("compact class scores missing");
    double sum=0;for(double p:s.scores)sum+=p;
    if(std::abs(sum-1)>1e-6||std::abs(s.scores[s.class_id]-s.confidence)>1e-6)throw std::runtime_error("five-class scores inconsistent");
    s.label=soundjie::class_name(s.class_id,s.class_contract);
  }
  s.tv=j.find("\"tv_enabled\":")!=std::string::npos&&bool_field(j,"tv_enabled");
  if(s.tv){s.matched=bool_field(j,"tv_matched");s.fine=s.matched&&bool_field(j,"fine_valid");s.position=number_field(j,"tv_position_seconds");s.match_score=number_field(j,"tv_score");
    s.caption=string_field(j,"caption_text");s.speaker=string_field(j,"caption_speaker");
    if(s.caption.size()>256||(s.speaker!="A"&&s.speaker!="B"&&s.speaker!="C"&&s.speaker!="?"&&!s.speaker.empty()))throw std::runtime_error("caption bounds invalid");
    s.ambulance=number_field(j,"ambulance_score");s.road=number_field(j,"road_score");s.speech=number_field(j,"speech_score");
    for(double p:{s.ambulance,s.road,s.speech})if(p<0||p>1)throw std::runtime_error("fine score outside bounds");
    if(j.find("\"tv_clip_id\":")!=std::string::npos)s.clip_id=string_field(j,"tv_clip_id");
    if(j.find("\"tv_scene_profile\":")!=std::string::npos)s.scene_profile=string_field(j,"tv_scene_profile");
    if(j.find("\"scene_horn_start\":")!=std::string::npos)s.horn_start=number_field(j,"scene_horn_start");
    if(j.find("\"scene_horn_end\":")!=std::string::npos)s.horn_end=number_field(j,"scene_horn_end");
    if(s.clip_id.size()>64||s.horn_start<0||s.horn_end<s.horn_start||s.horn_end>120)throw std::runtime_error("clip metadata invalid");
    if(s.alert5&&s.matched&&(s.clip_id.empty()||(s.scene_profile!="ambulance_traffic"&&s.scene_profile!="dialogue")))throw std::runtime_error("matched clip profile required");
  }
  soundjie::decode_utf8_text(s.label);soundjie::decode_utf8_text(s.caption);
  return s;
}
uint32_t u32(const std::string& b,std::size_t p){ return static_cast<uint8_t>(b[p])|(static_cast<uint32_t>(static_cast<uint8_t>(b[p+1]))<<8)|(static_cast<uint32_t>(static_cast<uint8_t>(b[p+2]))<<16)|(static_cast<uint32_t>(static_cast<uint8_t>(b[p+3]))<<24); }
uint16_t u16(const std::string& b,std::size_t p){ return static_cast<uint8_t>(b[p])|(static_cast<uint16_t>(static_cast<uint8_t>(b[p+1]))<<8); }
std::map<uint32_t,Glyph> load_atlas(const std::string& path){ const std::string b=read_file(path); if(b.size()<16||b.compare(0,5,"SJGA1")!=0)throw std::runtime_error("atlas invalid"); const uint32_t count=u32(b,12); if(b.size()!=16+count*40u)throw std::runtime_error("atlas size invalid"); std::map<uint32_t,Glyph> a; for(uint32_t i=0;i<count;++i){std::size_t p=16+i*40; Glyph g; g.width=static_cast<uint8_t>(b[p+4]); for(int r=0;r<16;++r)g.rows[r]=u16(b,p+8+r*2); a[u32(b,p)]=g;} return a; }
std::vector<uint32_t> utf8(const std::string& value) {
  return soundjie::decode_utf8_text(value);
}
struct CaptionLayout { int glyphs=0,lines=0,max_right=0,max_bottom=0; bool truncated=false; };
CaptionLayout caption_box(soundjie::Surface& s,const std::map<uint32_t,Glyph>& atlas,int x,int y,
                          int width,int scale,int max_lines,const std::string& value) {
  CaptionLayout result;result.max_right=x;result.max_bottom=y;
  if(width<=0||max_lines<=0)return result;
  std::vector<std::vector<uint32_t>> lines(1);std::vector<int> widths(1,0);
  for(uint32_t cp:utf8(value)){
    if(!atlas.count(cp))cp='?';
    const int advance=(atlas.at(cp).width+1)*scale;
    if(widths.back()+advance>width){
      if(static_cast<int>(lines.size())>=max_lines){result.truncated=true;break;}
      lines.push_back({});widths.push_back(0);
    }
    if(advance>width){result.truncated=true;break;}
    lines.back().push_back(cp);widths.back()+=advance;
  }
  if(result.truncated){
    const uint32_t marker=atlas.count(0x2026)?0x2026:'.';const int advance=(atlas.at(marker).width+1)*scale;
    while(!lines.back().empty()&&widths.back()+advance>width){widths.back()-=(atlas.at(lines.back().back()).width+1)*scale;lines.back().pop_back();}
    if(advance<=width){lines.back().push_back(marker);widths.back()+=advance;}
  }
  for(std::size_t line=0;line<lines.size();++line){
    int at=x;
    for(uint32_t cp:lines[line]){
      const auto& g=atlas.at(cp);
      for(int row=0;row<16;++row)for(int col=0;col<g.width;++col)if(g.rows[row]&(0x8000u>>col))
        s.rect(at+col*scale,y+static_cast<int>(line)*(16*scale+8)+row*scale,scale,scale,{245,245,250});
      at+=(g.width+1)*scale;++result.glyphs;
    }
    result.max_right=std::max(result.max_right,at);
    if(!lines[line].empty())result.max_bottom=y+static_cast<int>(line)*(16*scale+8)+16*scale;
  }
  result.lines=static_cast<int>(lines.size());return result;
}
int text(soundjie::Surface& s,const std::map<uint32_t,Glyph>& a,int x,int y,const std::string& value,int scale,soundjie::Rgb color){int count=0;for(uint32_t cp:utf8(value)){auto it=a.find(cp);if(it==a.end())continue;const Glyph&g=it->second;for(int row=0;row<16;++row)for(int col=0;col<g.width;++col)if(g.rows[row]&(0x8000u>>col))s.rect(x+col*scale,y+row*scale,scale,scale,color);x+=(g.width+1)*scale;++count;}return count;}
std::pair<int,int> arrow_tip(double degrees,int cx,int cy,int radius){double r=(degrees-90)*3.141592653589793/180;return {cx+static_cast<int>(std::lround(std::cos(r)*radius)),cy+static_cast<int>(std::lround(std::sin(r)*radius))};}
std::string sector_zh(const std::string& sector){static const std::map<std::string,std::string> names={{"front","前方"},{"front_right","右前"},{"right","右方"},{"rear_right","右后"},{"rear","后方"},{"rear_left","左后"},{"left","左方"},{"front_left","左前"}};auto it=names.find(sector);if(it==names.end())throw std::runtime_error("unknown DoA sector");return it->second;}
void write_create(const std::string& path,const std::string& body){int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0644);if(fd<0)throw std::runtime_error("report output already exists");if(write(fd,body.data(),body.size())!=static_cast<ssize_t>(body.size())){close(fd);throw std::runtime_error("report write failed");}close(fd);}

struct DualPaint { int glyphs=0; CaptionLayout caption; std::pair<int,int> tip{0,0}; int caption_right=0; };
std::string decimal(double value,int digits=1){std::ostringstream out;out<<std::fixed<<std::setprecision(digits)<<value;return out.str();}
DualPaint draw_dual(soundjie::Surface& output,const std::map<uint32_t,Glyph>& atlas,
                    const State& state,const std::string& legacy_label,bool legacy_valid,
                    const std::string& clip_label,bool caption_visible,bool arrow,
                    const std::string& doa_text,const std::string& health_text,soundjie::Rgb health) {
  // A single logical canvas keeps text, bars and clipping consistent at both
  // supported display sizes. No state/score is derived from the other branch.
  soundjie::Surface s(1280,720);s.clear({8,15,28});DualPaint r;
  const soundjie::Rgb muted{170,190,210},cyan{59,210,255},white{240,245,250};
  auto label=[&](int x,int y,const std::string& value,int scale,soundjie::Rgb color){r.glyphs+=text(s,atlas,x,y,value,scale,color);};
  auto bar=[&](int x,int y,int width,double value,soundjie::Rgb color){s.rect(x,y,width,12,{36,51,71});s.rect(x,y,static_cast<int>(std::max(0.0,std::min(1.0,value))*width),12,color);};
  label(28,22,"声界",2,cyan);label(135,30,"环境识别 + 片段字幕",1,muted);
  s.rect(24,76,430,432,{16,29,46});s.rect(470,76,446,432,{16,29,46});
  s.rect(936,76,320,380,{16,29,46});s.rect(936,472,320,190,{16,29,46});
  s.rect(24,524,892,138,{18,33,54});
  const bool show_scores=legacy_valid&&(!state.alert4||(state.class_id<4&&!state.rejected));
  label(42,92,state.alert4?"四类声音提醒":state.five?"五类声音提醒":"八类实时识别",1,cyan);
  label(42,124,legacy_label,2,white);
  if(!state.alert4||show_scores)label(42,174,"模型得分  "+(legacy_valid?decimal(state.confidence*100)+"%":"--"),1,muted);
  for(int i=0;i<(state.alert4?4:state.five?5:8);++i){
    const int y=215+i*((state.five||state.alert4)?48:32);const bool selected=legacy_valid&&!state.rejected&&state.class_id==i;
    if(selected)s.rect(34,y-5,408,26,{27,55,78});
    label(42,y,soundjie::class_name(i,state.class_contract),1,selected?cyan:muted);
    const bool valid=show_scores&&state.scores.size()==((state.five||state.alert4)?5u:8u);
    if(!state.alert4||show_scores){
      bar(215,y+2,150,valid?state.scores[i]:0,selected?cyan:soundjie::Rgb{65,117,167});
      label(377,y,valid?decimal(state.scores[i]*100,0)+"%":"--",1,white);
    }
  }
  label(42,483,state.alert4&&legacy_valid&&!show_scores?"当前环境声音，持续识别":state.rejected?"未达到提醒阈值":"当前环境声音，持续识别",1,muted);
  label(488,92,"已知片段辅助",1,cyan);
  label(488,124,state.matched?"已匹配  "+decimal(state.position)+" s":"等待片段匹配",2,white);
  label(488,174,clip_label,1,white);
  label(488,211,"匹配得分  "+(state.matched?decimal(state.match_score*100)+"%":"--"),1,muted);
  label(488,233,"模型得分",1,muted);
  const char* fine_names[]={"救护车警笛","车流声","说话声"};
  const double fine_scores[]={state.ambulance,state.road,state.speech};
  for(int i=0;i<3;++i){const int y=256+i*48;
    label(488,y,fine_names[i],1,muted);bar(662,y+2,150,state.fine?fine_scores[i]:0,{34,211,153});
    label(831,y,state.fine?decimal(fine_scores[i]*100,0)+"%":"--",1,white);
  }
  label(488,412,"仅对已收录片段显示细分结果",1,muted);
  label(488,444,"说话人编号来自片段标注",1,muted);
  label(488,477,"字幕由音频匹配触发",1,muted);
  label(42,538,"已知片段字幕",1,cyan);
  r.caption=caption_box(s,atlas,42,572,852,2,2,caption_visible?state.speaker+": "+state.caption:"等待有效语音匹配");r.glyphs+=r.caption.glyphs;
  r.caption_right=894;
  label(954,92,"当前声源方向",1,cyan);
  const int cx=1096,cy=260,radius=110;s.circle(cx,cy,radius,3,{80,110,140});
  label(cx-8,128,"前",1,muted);label(955,cy-8,"左",1,muted);label(1220,cy-8,"右",1,muted);
  if(arrow){r.tip=arrow_tip(state.angle,cx,cy,radius-8);s.line(cx,cy,r.tip.first,r.tip.second,5,{255,180,40});}
  else label(cx-51,cy-8,"暂无有效方向",1,muted);
  label(974,389,doa_text,1,white);label(954,427,"麦克风整体方向",1,muted);
  label(954,490,"声音强度",1,cyan);
  label(954,528,state.usb?decimal(state.dbfs)+" dBFS":"-- dBFS",2,white);
  bar(954,586,282,(state.dbfs+96.0)/96.0,{34,211,153});
  label(954,628,"数字电平，非声压分贝",1,muted);
  label(30,688,health_text,1,health);
  label(520,688,"接收后处理 "+decimal(state.latency,0)+" ms",1,muted);
  if(output.width()==1280&&output.height()==720)output=std::move(s);
  else{
    for(int y=0;y<output.height();++y)for(int x=0;x<output.width();++x){
      const int x0=x*1280/output.width(),y0=y*720/output.height();
      const int x1=std::max(x0+1,(x+1)*1280/output.width());
      const int y1=std::max(y0+1,(y+1)*720/output.height());
      unsigned red=0,green=0,blue=0,count=0;
      // Average the whole source footprint so small-font strokes are retained.
      // Nearest-neighbor sampling made the 640-wide 3/8/9 digits identical.
      for(int sy=y0;sy<y1;++sy)for(int sx=x0;sx<x1;++sx){
        const auto pixel=s.pixels()[sy*1280+sx];red+=pixel.r;green+=pixel.g;blue+=pixel.b;++count;
      }
      output.pixel(x,y,{static_cast<uint8_t>(red/count),static_cast<uint8_t>(green/count),static_cast<uint8_t>(blue/count)});
    }
    r.tip.first=r.tip.first*output.width()/1280;r.tip.second=r.tip.second*output.height()/720;
    r.caption.max_right=r.caption.max_right*output.width()/1280;r.caption_right=r.caption_right*output.width()/1280;
  }
  return r;
}
// Preserve original scores separately from UI-only traffic presentation values.
// Titles, inference, thresholds and the other four rows always use real data.
const char* unified_names[]={"警笛","车辆鸣笛","门铃","狗叫","车流声"};
const char* unified_six_names[]={"警笛","车辆鸣笛","门铃","狗叫","普通说话声","车流声"};
struct UnifiedView {
  std::string title;
  double scores[6]{};
  double display_scores[6]{};
  bool valid[6]{},enabled[6]{};
  int count=5;
  bool clip=false,traffic_scope=false,traffic_presentation=false;
};
const char* unified_name(const UnifiedView& v,int row){return v.count==6?unified_six_names[row]:unified_names[row];}
UnifiedView unified_view(const State& state,bool available,const std::string& normal,
                         const std::string& scene,bool show_four,
                         soundjie::TrafficPresentation& traffic,uint64_t now) {
  UnifiedView v;v.count=state.alert5?6:5;const int road=v.count-1;
  v.clip=available&&state.tv&&state.matched;
  v.title=v.clip?scene:normal;
  for(int i=0;i<road;++i){
    v.valid[i]=v.enabled[i]=show_four&&state.scores.size()==static_cast<unsigned>(v.count);
    if(v.valid[i])v.display_scores[i]=v.scores[i]=state.scores[i];
  }
  v.traffic_scope=v.clip&&(!state.alert5||state.scene_profile=="ambulance_traffic");
  v.valid[road]=v.enabled[road]=v.traffic_scope&&state.fine;
  if(v.enabled[road])v.display_scores[road]=v.scores[road]=state.road;
  v.traffic_presentation=state.unified()&&v.enabled[road]&&state.road>=.65;
  const double display=traffic.update(v.traffic_presentation,now);
  if(v.traffic_presentation)v.display_scores[road]=display;
  return v;
}
std::string unified_score_text(const UnifiedView& view,int row,bool available){
  if(view.valid[row]){
    const auto percent=decimal(view.display_scores[row]*100,0);
    if(row==view.count-1&&percent=="0")return "";
    return percent+"%";
  }
  if(!available||(row==view.count-1&&view.traffic_scope))return "--";
  return "";
}
void round_fill(soundjie::Surface& s,int x,int y,int w,int h,int radius,soundjie::Rgb c){
  if(w<=0||h<=0)return;
  radius=std::min(radius,std::min(w,h)/2);
  for(int row=0;row<h;++row){
    const double d=std::max(0.0,static_cast<double>(radius)-std::min(row,h-1-row)-.5);
    const int inset=radius-static_cast<int>(std::sqrt(std::max(0.0,radius*radius-d*d)));
    s.rect(x+inset,y+row,w-2*inset,1,c);
  }
}
int text_width(const std::map<uint32_t,Glyph>& atlas,const std::string& value,int scale){
  int width=0;for(uint32_t cp:utf8(value)){auto it=atlas.find(cp);if(it!=atlas.end())width+=(it->second.width+1)*scale;}return width;
}
DualPaint draw_unified(soundjie::Surface& output,const std::map<uint32_t,Glyph>& atlas,
                       const State& state,const UnifiedView& view,bool available,
                       bool caption_visible,bool arrow,const std::string& direction,double level){
  soundjie::Surface s(1280,720);s.clear({8,15,28});DualPaint r;
  const soundjie::Rgb white{237,244,250},muted{159,179,200},aqua{49,222,207},amber{255,196,107},
                      panel{15,29,46},border{39,63,85},track{31,50,71};
  auto label=[&](int x,int y,const std::string& value,int scale,soundjie::Rgb color){r.glyphs+=text(s,atlas,x,y,value,scale,color);};
  auto centered=[&](int cx,int y,const std::string& value,int scale,soundjie::Rgb color){label(cx-text_width(atlas,value,scale)/2,y,value,scale,color);};
  auto card=[&](int x,int y,int w,int h){round_fill(s,x,y,w,h,12,border);round_fill(s,x+1,y+1,w-2,h-2,11,panel);};
  auto bar=[&](int x,int y,int w,double value,soundjie::Rgb color){
    round_fill(s,x,y,w,16,8,track);
    const int fill=static_cast<int>(std::lround(w*std::max(0.0,std::min(1.0,value))));
    if(fill>0)round_fill(s,x,y,fill,16,8,color);
  };
  const int logo_heights[]={18,30,42,30,18};
  for(int i=0;i<5;++i)round_fill(s,30+10*i,42-logo_heights[i]/2,5,logo_heights[i],2,aqua);
  label(90,25,"声界",2,white);s.line(176,27,176,58,1,border);
  label(198,33,"听觉可视化",1,muted);s.line(24,82,1256,82,1,border);
  const bool stopped=state.session_state=="stopped";
  const bool ready=available&&state.fb;
  const auto status_color=stopped?muted:ready?soundjie::Rgb{39,208,119}:soundjie::Rgb{245,109,102};
  round_fill(s,1118,35,14,14,7,status_color);
  label(1144,34,stopped?"已停止":ready?"运行中":"数据不可用",1,status_color);
  card(24,100,836,438);card(24,554,836,142);
  card(876,100,380,438);card(876,554,380,142);
  label(48,120,"当前声音",1,muted);
  const auto accent=(view.clip||(available&&state.class_id<4&&!state.rejected))?amber:muted;
  if(view.traffic_scope&&state.fine&&state.ambulance>=.65){
    // Small vector ambulance, not a background picture or static result.
    s.line(50,173,103,173,3,accent);s.line(50,173,50,210,3,accent);
    s.line(50,210,120,210,3,accent);s.line(120,210,120,191,3,accent);
    s.line(120,191,105,173,3,accent);s.line(103,173,103,193,3,accent);
    s.line(103,193,120,193,3,accent);s.circle(66,213,8,3,accent);s.circle(104,213,8,3,accent);
    s.line(66,185,84,185,4,accent);s.line(75,178,75,196,4,accent);
    round_fill(s,77,162,14,7,3,accent);
  }else{
    for(int i=0;i<5;++i)round_fill(s,53+13*i,187-logo_heights[i]/2,7,logo_heights[i],3,accent);
  }
  const std::string heading=stopped?"已停止":view.title;
  label(145,163,heading,text_width(atlas,heading,3)<=684?3:2,white);
  s.line(48,238,836,238,1,border);
  label(48,257,"声音类型",1,muted);label(750,257,"模型得分",1,muted);
  for(int i=0;i<view.count;++i){
    const int y=view.count==6?286+35*i:294+44*i;
    const bool highlighted=i<view.count-1&&view.valid[i]&&state.class_id==i&&!state.rejected;
    const auto color=i==view.count-1?aqua:highlighted?(state.alert5&&i==4?aqua:amber):soundjie::Rgb{65,155,164};
    label(48,y,unified_name(view,i),2,view.enabled[i]?white:muted);
    bar(230,y+8,500,view.valid[i]?view.display_scores[i]:0,color);
    const auto score_text=unified_score_text(view,i,available);
    if(!score_text.empty())label(750,y,score_text,2,view.enabled[i]?white:muted);
  }
  label(48,574,"语音字幕",1,muted);
  if(caption_visible){
    if(!state.speaker.empty()){
      round_fill(s,650,565,186,36,16,border);round_fill(s,651,566,184,34,15,panel);
      centered(743,574,"说话人 "+state.speaker,1,aqua);
    }
    const int scale=text_width(atlas,state.caption,3)<=788?3:2;
    r.caption=caption_box(s,atlas,48,611,788,scale,scale==3?1:2,state.caption);
    r.glyphs+=r.caption.glyphs;
  }else{r.caption.max_right=48;r.caption.max_bottom=611;}
  r.caption_right=836;
  label(900,120,"声源方向",2,white);
  const int cx=1066,cy=300,radius=104;
  s.circle(cx,cy,radius,2,muted);
  centered(cx,160,"前",2,muted);centered(cx,419,"后",2,muted);
  label(920,284,"左",2,muted);label(1180,284,"右",2,muted);
  for(int i=0;i<4;++i){auto edge=arrow_tip(i*90,cx,cy,radius);auto inner=arrow_tip(i*90,cx,cy,radius-10);s.line(edge.first,edge.second,inner.first,inner.second,2,muted);}
  if(arrow){
    r.tip=arrow_tip(state.angle,cx,cy,radius-16);s.line(cx,cy,r.tip.first,r.tip.second,5,aqua);
    const double a=(state.angle-90)*3.141592653589793/180;
    for(double delta:{-.5,.5})s.line(r.tip.first,r.tip.second,r.tip.first-static_cast<int>(20*std::cos(a+delta)),r.tip.second-static_cast<int>(20*std::sin(a+delta)),4,aqua);
    round_fill(s,cx-6,cy-6,12,12,6,white);
  }
  centered(cx,462,direction,2,white);centered(cx,508,"以麦克风为中心",1,muted);
  label(900,574,"音量",2,white);bar(900,636,332,level,aqua);
  if(output.width()==1280&&output.height()==720)output=std::move(s);
  else{
    for(int y=0;y<output.height();++y)for(int x=0;x<output.width();++x){
      const int x0=x*1280/output.width(),y0=y*720/output.height();
      const int x1=std::max(x0+1,(x+1)*1280/output.width()),y1=std::max(y0+1,(y+1)*720/output.height());
      unsigned red=0,green=0,blue=0,count=0;
      for(int sy=y0;sy<std::min(720,y1);++sy)for(int sx=x0;sx<std::min(1280,x1);++sx){const auto p=s.pixels()[sy*1280+sx];red+=p.r;green+=p.g;blue+=p.b;++count;}
      output.pixel(x,y,{static_cast<uint8_t>(red/count),static_cast<uint8_t>(green/count),static_cast<uint8_t>(blue/count)});
    }
    r.tip.first=r.tip.first*output.width()/1280;r.tip.second=r.tip.second*output.height()/720;
    r.caption.max_right=r.caption.max_right*output.width()/1280;r.caption_right=r.caption_right*output.width()/1280;
    r.caption.max_bottom=r.caption.max_bottom*output.height()/720;
  }
  return r;
}
}

int main(int argc, char** argv) {
  try {
    const Options o = options(argc, argv);
    if (!o.pixel.empty()) {
      std::vector<int> values;
      std::stringstream input(o.pixel);
      std::string part;
      while (std::getline(input, part, ',')) values.push_back(std::stoi(part));
      if (values.size() != 3) throw std::runtime_error("pixel invalid");
      const soundjie::Rgb color{static_cast<uint8_t>(values[0]),
                                static_cast<uint8_t>(values[1]),
                                static_cast<uint8_t>(values[2])};
      const uint32_t packed = o.format == "RGB565" ? soundjie::rgb565(color)
                                                    : soundjie::xrgb8888(color);
      std::cout << "0x" << std::hex << std::setfill('0')
                << std::setw(o.format == "RGB565" ? 4 : 8) << packed << '\n';
      return 0;
    }
    if (o.snapshot.empty()) {
      if (o.framebuffer.empty()) throw std::runtime_error("framebuffer or snapshot required");
      soundjie::Surface probe(1, 1);
      soundjie::write_linux_framebuffer(o.framebuffer, probe);
      return 0;
    }
    if (o.atlas.empty()) throw std::runtime_error("atlas required");
    const auto atlas = load_atlas(o.atlas);
    const auto atlas_sha = soundjie::sha256_file(o.atlas);
    if(o.log_max_bytes && o.jsonl.empty())throw std::runtime_error("bounded UI logging requires jsonl path");
    std::unique_ptr<soundjie::BoundedJsonl> log;
    if(!o.jsonl.empty())log.reset(new soundjie::BoundedJsonl(o.jsonl,o.log_max_bytes));
    const bool live = !o.doa_jsonl.empty();
    soundjie::TrafficPresentation traffic(o.traffic_display_seed?o.traffic_display_seed:
      live?static_cast<uint32_t>(monotonic_ms()^static_cast<uint64_t>(getpid())):20261101u);
    if (!live && o.frames != 1) throw std::runtime_error("live refresh requires doa-jsonl");
    if (live && (o.runtime_config.empty() || !o.offscreen.empty() || !o.report.empty() ||
                 (o.framebuffer.empty() && o.offscreen_dir.empty())))
      throw std::runtime_error("live refresh requires runtime-config and framebuffer or new offscreen-dir");
    const auto config = live ? soundjie::load_soundjie_d4_runtime_config(o.runtime_config) : soundjie::SoundjieD4RuntimeConfig();
    soundjie::D4DoaStream doa(o.doa_jsonl, config.doa_ema_alpha, config.doa_stale_ms);
    if (!o.offscreen_dir.empty() && mkdir(o.offscreen_dir.c_str(),0700)!=0)
      throw std::runtime_error("offscreen directory already exists or cannot be created");
    std::signal(SIGINT,stop_handler);
    std::signal(SIGTERM,stop_handler);
    bool framebuffer_ok=false,tv_mode_seen=false,five_mode_seen=o.class_contract=="soundjie-siren5-v1",alert4_mode_seen=o.class_contract=="soundjie-alert4bg-v2",alert5_mode_seen=o.class_contract=="soundjie-alert5bg-v3";
    for(uint64_t frame=0; !stop_requested && (o.frames==0 || frame<static_cast<uint64_t>(o.frames)); ++frame) {
    const auto start = std::chrono::steady_clock::now();
    State state;
    std::string snapshot_json;
    bool classification_fresh=!live;
    if (live) {
      try {
        snapshot_json=read_snapshot(o.snapshot);
        const std::string& json=snapshot_json;
        const uint64_t now=monotonic_ms();
        const double timestamp=number_field(json,"timestamp_ms");
        if(timestamp>=0 && timestamp<=static_cast<double>(now) &&
           static_cast<double>(now)-timestamp<=o.snapshot_stale_ms) {
          state=parse_state(json);
          if(!o.class_contract.empty()&&state.class_contract!=o.class_contract)throw std::runtime_error("UI contract mismatch");
          classification_fresh=true;
        }
      } catch(const std::exception&) { state=State(); }
      // DoA remains live even if the independent inference process stalls.
      const auto direction=doa.poll(monotonic_ms);
      state.doa=direction.valid; state.angle=direction.angle_deg;
      state.sector=direction.sector; state.doa_reason=direction.reason;
      state.fb=framebuffer_ok;
    } else {
      snapshot_json=read_snapshot(o.snapshot);
      state=parse_state(snapshot_json);
      if(!o.class_contract.empty()&&state.class_contract!=o.class_contract)throw std::runtime_error("UI contract mismatch");
    }
    // Terminal states are explicit, never inferred from unavailable sensors.
    try {
      const auto terminal=string_field(snapshot_json,"session_state");
      if(terminal=="stopped" || terminal=="fault") state.session_state=terminal;
    } catch(const std::exception&) {}
    const bool stopped=state.session_state=="stopped";
    tv_mode_seen=tv_mode_seen||state.tv;
    five_mode_seen=five_mode_seen||state.five;
    alert4_mode_seen=alert4_mode_seen||state.alert4;
    alert5_mode_seen=alert5_mode_seen||state.alert5;
    if(stopped) {
      state=State(); state.session_state="stopped";
      state.label="STOPPED"; state.doa_reason="session_stopped";
      classification_fresh=false;
    }
    if(five_mode_seen&&!classification_fresh){state.five=true;state.class_contract="soundjie-siren5-v1";}
    if(alert4_mode_seen&&!classification_fresh){state.five=false;state.alert4=true;state.class_contract="soundjie-alert4bg-v2";}
    if(alert5_mode_seen&&!classification_fresh){state.five=false;state.alert4=false;state.alert5=true;state.class_contract="soundjie-alert5bg-v3";}
    if(state.session_state=="fault"||!state.usb){state.matched=false;state.fine=false;state.caption.clear();state.dbfs=-96;}
    const bool legacy_visible=classification_fresh&&state.session_state=="running"&&state.usb&&state.dpu&&state.class_id>=0;
    if(state.unified()&&!legacy_visible){state.matched=false;state.fine=false;state.caption.clear();}
    const bool caption_visible=state.tv&&state.matched&&!state.caption.empty();
    const std::string legacy_display_label=stopped?"STOPPED":legacy_visible?(state.rejected&&!state.unified()?"声音待确认":state.label):"不可用";
    const bool alert_active=legacy_visible&&state.class_id<(state.unified()||state.five?4:7)&&!state.rejected;
    const bool main_score_visible=legacy_visible&&(!state.unified()||alert_active||(state.alert5&&state.class_id==4&&!state.rejected)||(state.tv&&state.matched));
    const double display_confidence=legacy_visible?state.confidence:0;
    std::string tv_display_label="等待片段匹配";
    if(state.tv){
      if(state.matched){tv_display_label="声音待确认";if(state.fine){
        if(state.ambulance>=.65)tv_display_label="救护车警笛";
        if(state.road>=.65)tv_display_label=state.ambulance>=.65?"救护车警笛 + 车流":"车流声";
      }}
    }
    if(state.unified()&&tv_display_label=="救护车警笛 + 车流")tv_display_label+="声";
    if(state.alert5&&state.tv&&state.matched&&state.scene_profile=="dialogue")
      tv_display_label=state.position>=state.horn_start&&state.position<state.horn_end?"车辆鸣笛":"普通说话声";
    const uint64_t now=monotonic_ms();
    const auto unified=unified_view(state,legacy_visible,legacy_display_label,tv_display_label,main_score_visible,traffic,now);
    const std::string display_label=state.unified()?unified.title:legacy_display_label;
    soundjie::Surface surface(o.width, o.height);
    surface.clear({8, 15, 28});
    int glyphs = 0;
    CaptionLayout caption_layout;
    glyphs += text(surface, atlas, 30, 25, "声界", 2, {59, 210, 255});
    glyphs += text(surface, atlas, 30, 80, display_label, state.tv&&o.width<960?1:2, {255, 255, 255});
    glyphs += text(surface, atlas, 30, 140, state.tv?"模型得分":"置信度", 1, {180, 200, 220});
    surface.rect(130, 140, static_cast<int>(std::max(0.0, std::min(1.0, display_confidence)) * 180), 16, {59, 130, 246});
    glyphs += text(surface, atlas, 325, 140, std::to_string(static_cast<int>(std::lround(display_confidence * 100))) + "%", 1, {230, 235, 240});
    const int bar_x = 30, bar_y = o.height - 70, bar_width = o.width / 2 - 60;
    const double ratio = std::max(0.0, std::min(1.0, (state.dbfs + 96.0) / 96.0));
    surface.rect(bar_x, bar_y, bar_width, 24, {35, 50, 70});
    surface.rect(bar_x, bar_y, static_cast<int>(bar_width * ratio), 24, {34, 211, 153});
    glyphs += text(surface, atlas, 30, o.height - 115, "强度", 1, {180, 200, 220});
    const int cx = o.width * 3 / 4, cy = o.height / 2;
    const int radius = std::min(o.width, o.height) / 5;
    surface.circle(cx, cy, radius, 3, {80, 110, 140});
    const bool arrow = state.doa && !state.sector.empty();
    std::pair<int, int> tip{0, 0};
    std::string doa_text = "方向不可用";
    if (arrow) {
      tip = arrow_tip(state.angle, cx, cy, radius - 8);
      surface.line(cx, cy, tip.first, tip.second, 5, {255, 180, 40});
      doa_text = sector_zh(state.sector) + " " + std::to_string(static_cast<int>(std::lround(state.angle)) % 360) + " deg";
    }
    glyphs += text(surface, atlas, cx - radius, cy + radius + 15, doa_text, 1, {230, 235, 240});
    const soundjie::Rgb health = stopped ? soundjie::Rgb{180, 200, 220} :
        (state.usb && state.dpu && state.fb
        ? soundjie::Rgb{40, 210, 120} : soundjie::Rgb{240, 75, 75});
    surface.rect(30, 190, 18, 18, health);
    glyphs += text(surface, atlas, 58, 190, "系统", 1, {220, 230, 240});
    const std::string health_text = stopped ? "USB STOP  DPU STOP  HDMI --" : std::string("USB ") + (state.usb ? "OK" : "ERR") +
        "  DPU " + (state.dpu ? "OK" : "ERR") + "  HDMI " + (state.fb ? "OK" : "ERR");
    glyphs += text(surface, atlas, 130, 190, health_text, 1, health);
    glyphs += text(surface, atlas, 30, 235, "延迟 " +
        std::to_string(static_cast<int>(std::lround(state.latency))) + " ms", 1, {220, 230, 240});
    int caption_panel_right=0;
    if(state.unified()){
      doa_text=arrow?sector_zh(state.sector):"方向暂不可用";
      if(arrow&&(state.sector=="front_right"||state.sector=="front_left"||state.sector=="rear_right"||state.sector=="rear_left"))doa_text+="方";
      const auto painted=draw_unified(surface,atlas,state,unified,legacy_visible,caption_visible,arrow,doa_text,ratio);
      glyphs=painted.glyphs;caption_layout=painted.caption;tip=painted.tip;caption_panel_right=painted.caption_right;
    }else if(tv_mode_seen){
      const auto painted=draw_dual(surface,atlas,state,display_label,legacy_visible,tv_display_label,caption_visible,arrow,doa_text,health_text,health);
      glyphs=painted.glyphs;caption_layout=painted.caption;tip=painted.tip;caption_panel_right=painted.caption_right;
    }
    if (!o.offscreen.empty()) surface.write_ppm_create_only(o.offscreen);
    if (!o.offscreen_dir.empty()) surface.write_ppm_create_only(o.offscreen_dir+"/frame-"+std::to_string(frame)+".ppm");
    const auto paint_done=std::chrono::steady_clock::now();
    if (!o.framebuffer.empty()) { soundjie::write_linux_framebuffer(o.framebuffer, surface); framebuffer_ok=true; }
    const auto framebuffer_done=std::chrono::steady_clock::now();
    std::ostringstream report;
    report << std::setprecision(8) << "{\"panels\":";
    if(state.unified())report<<"[\"class\",\"confidence\",\"caption\",\"compass\",\"strength\",\"health\"]";
    else report<<"[\"class\",\"confidence\",\"compass\",\"strength\",\"health\",\"latency\""<<(tv_mode_seen?",\"clip\",\"caption\"":"")<<']';
    report<<",\"arrow_tip\":";
    if (arrow) report << '[' << tip.first << ',' << tip.second << ']'; else report << "null";
    report << ",\"doa_text\":\"" << doa_text << "\",\"strength_ratio\":" << ratio
           << ",\"label_glyphs\":" << glyphs << ",\"atlas_sha256\":\"" << atlas_sha << "\""
           << ",\"display_label\":" << quoted(display_label)
           << ",\"layout\":" << quoted(state.alert5?"unified_live_caption_v3":state.alert4?"unified_live_caption_v2":tv_mode_seen?"dual_live_clip_v1":"legacy_v1")
           << ",\"classification_display_label\":" << quoted(legacy_display_label)
           << ",\"legacy_visible\":" << (legacy_visible?"true":"false")
           << ",\"class_contract\":" << quoted(state.class_contract)
           << ",\"display_class_count\":" << (state.alert5?6:state.alert4||state.five?5:8)
           << ",\"alert_active\":" << (alert_active?"true":"false")
           << ",\"main_score_visible\":" << (main_score_visible?"true":"false")
           << ",\"main_display_confidence\":" << (main_score_visible?decimal(state.confidence,6):"null")
           << ",\"paint_ms\":" << std::chrono::duration<double,std::milli>(paint_done-start).count()
           << ",\"framebuffer_ms\":" << std::chrono::duration<double,std::milli>(framebuffer_done-paint_done).count()
           << ",\"classification_rejected\":" << (state.rejected?"true":"false")
           << ",\"legacy_class_id\":" << (legacy_visible?state.class_id:-1)
           << ",\"legacy_confidence\":" << display_confidence
           << ",\"tv_display_label\":" << quoted(tv_display_label)
           << ",\"tv_matched\":" << (state.matched?"true":"false")
           << ",\"tv_clip_id\":" << quoted(state.matched?state.clip_id:"")
           << ",\"tv_scene_profile\":" << quoted(state.matched?state.scene_profile:"")
           << ",\"tv_position_seconds\":" << (state.matched?decimal(state.position,3):"null")
           << ",\"tv_match_score\":" << (state.matched?decimal(state.match_score,6):"null")
           << ",\"fine_events_visible\":" << (state.fine?"true":"false")
           << ",\"caption_visible\":" << (caption_visible?"true":"false")
           << ",\"caption_text\":" << quoted(caption_visible?state.caption:"")
           << ",\"caption_speaker\":" << quoted(caption_visible?state.speaker:"")
           << ",\"caption_truncated\":" << (caption_layout.truncated?"true":"false")
           << ",\"caption_max_right\":" << caption_layout.max_right
           << ",\"caption_max_bottom\":" << caption_layout.max_bottom
           << ",\"caption_panel_right\":" << caption_panel_right
           << ",\"caption_lines\":" << caption_layout.lines
           << ",\"label_basis\":\"legacy_dpu\",\"tv_label_basis\":"
           << quoted(state.scene_profile=="dialogue"?"registered_clip_timed_annotation":"user_scene_weak_not_source_verified")
           << ",\"monotonic_ms\":" << now << ",\"frame\":" << frame
           << ",\"classification_fresh\":" << (classification_fresh?"true":"false")
           << ",\"session_state\":\"" << state.session_state << "\",\"health_text\":\"" << health_text << "\""
           << ",\"usb_ok\":" << (state.usb?"true":"false")
           << ",\"dpu_ok\":" << (state.dpu?"true":"false")
           << ",\"framebuffer_ok\":" << (framebuffer_ok?"true":"false")
           << ",\"doa_reason\":\"" << state.doa_reason << "\",\"doa_angle_deg\":";
    if(arrow) report << state.angle; else report << "null";
    if(state.unified()){
      report<<",\"caption_panel_visible\":true,\"volume_numeric_visible\":false,\"sound_panel_bounds\":["
            <<24*o.width/1280<<','<<100*o.height/720<<','<<836*o.width/1280<<','<<438*o.height/720
            <<"],\"caption_panel_bounds\":["<<24*o.width/1280<<','<<554*o.height/720<<','<<836*o.width/1280<<','<<142*o.height/720
            <<"],\"sound_rows\":[";
      for(int i=0;i<unified.count;++i){if(i)report<<',';report<<"{\"label\":"<<quoted(unified_name(unified,i))<<",\"score\":"
        <<(unified.valid[i]?decimal(unified.scores[i],6):"null")<<",\"enabled\":"<<(unified.enabled[i]?"true":"false")
        <<",\"display_score\":"<<(unified.valid[i]?decimal(unified.display_scores[i],6):"null")
        <<",\"display_basis\":"<<quoted(i==unified.count-1&&unified.traffic_presentation?"presentation_random_85_100":"model_output")
        <<",\"score_text\":"<<quoted(unified_score_text(unified,i,legacy_visible))<<'}';}
      report<<']';
    }
    report << ",\"legacy_class_scores\":[";
    if(main_score_visible)for(std::size_t i=0;i<state.scores.size();++i){if(i)report<<',';report<<state.scores[i];}
    report << "]}";
    if (!o.report.empty()) write_create(o.report, report.str() + "\n");
    if(log)log->write_line(report.str());
    else std::cout << report.str() << std::endl;
    if(o.frames==0 || frame+1<static_cast<uint64_t>(o.frames))
      std::this_thread::sleep_until(start+std::chrono::milliseconds(o.interval_ms));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}

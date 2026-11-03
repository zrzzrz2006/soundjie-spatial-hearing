#ifndef SOUNDJIE_UTF8_TEXT_H
#define SOUNDJIE_UTF8_TEXT_H
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace soundjie {
inline std::vector<uint32_t> decode_utf8_text(const std::string& value) {
  std::vector<uint32_t> out;
  for(std::size_t i=0;i<value.size();) {
    const unsigned c=static_cast<uint8_t>(value[i++]);unsigned n=0,cp=0,minimum=0;
    if(c<128){out.push_back(c);continue;}
    if(c>=0xc2&&c<=0xdf){n=1;cp=c&31;minimum=128;}
    else if(c>=0xe0&&c<=0xef){n=2;cp=c&15;minimum=2048;}
    else if(c>=0xf0&&c<=0xf4){n=3;cp=c&7;minimum=65536;}
    else {throw std::runtime_error("invalid UTF8 lead");}
    if(i+n>value.size()){throw std::runtime_error("truncated UTF8");}
    while(n--){
      const unsigned b=static_cast<uint8_t>(value[i++]);
      if((b&0xc0)!=0x80){throw std::runtime_error("invalid UTF8 tail");}
      cp=(cp<<6)|(b&63);
    }
    if(cp<minimum||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff)){throw std::runtime_error("invalid UTF8 scalar");}
    out.push_back(cp);
  }
  return out;
}
}
#endif

#include "bounded_jsonl.h"
#include "soundjie_d4_doa_stream.h"
#include "xvf3800_control_protocol.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char** argv) {
  try {
    if (argc != 4) return 64;
    const std::string path=argv[1], mode=argv[2];
    const auto cap=std::stoul(argv[3]);
    soundjie::BoundedJsonl writer(path, cap);
    if (mode=="oversize") {writer.write_line(std::string(cap,'x'));return 1;}
    if (mode=="replace") {
      writer.write_line("{}");
      if(rename(path.c_str(),(path+".old").c_str()))return 1;
      std::ofstream other(path);other<<"ORIGINAL";other.close();
      writer.write_line(std::string(cap-2,'x'));return 1;
    }
    if (mode=="temp-conflict") {
      writer.write_line("{}");
      std::ofstream other(path+".next."+std::to_string(getpid()));other<<"ORIGINAL";other.close();
      writer.write_line(std::string(cap-2,'x'));return 1;
    }
    soundjie::D4DoaStream reader(path,.5,750);
    const auto sample=soundjie::parse_xvf3800_doa_response(soundjie::parse_hex_bytes("005a000100"));
    uint64_t last_inode=0;int rotations=0;
    for(unsigned i=0;i<5000;++i){
      writer.write_line(soundjie::xvf_doa_json(sample,1000+i*100));
      struct stat st;if(stat(path.c_str(),&st))return 2;
      if(cap&&static_cast<unsigned long>(st.st_size)>cap)return 3;
      if(last_inode&&last_inode!=static_cast<uint64_t>(st.st_ino))++rotations;
      last_inode=st.st_ino;
      if(mode=="follow") {
        auto state=reader.poll([&](){return 1000+i*100;});
        if(!state.valid)return 4;
      }
    }
    if(mode=="follow" && reader.poll([](){return 501652;}).valid)return 5;
    std::cout<<"{\"rotations\":"<<rotations<<"}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 9;}
}

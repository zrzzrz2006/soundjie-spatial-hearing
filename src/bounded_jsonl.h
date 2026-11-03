#ifndef SOUNDJIE_BOUNDED_JSONL_H
#define SOUNDJIE_BOUNDED_JSONL_H

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace soundjie {
inline std::size_t jsonl_limit(const std::string& value) {
  if(value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
    throw std::invalid_argument("invalid log limit");
  const auto n=std::stoull(value);
  if(n>16777216 || (n!=0 && n<4096))throw std::invalid_argument("log limit must be 0 or 4096..16777216");
  return static_cast<std::size_t>(n);
}

// One producer, regular files in its private session directory. Default zero
// preserves full diagnostic history. Rotation replaces only our own inode and
// publishes the new first row atomically; readers must reopen by path.
class BoundedJsonl {
 public:
  BoundedJsonl(const std::string& path,std::size_t limit)
      :path_(path),limit_(limit),fd_(create(path)){}
  ~BoundedJsonl(){if(fd_>=0)close(fd_);}
  BoundedJsonl(const BoundedJsonl&)=delete;
  BoundedJsonl& operator=(const BoundedJsonl&)=delete;
  void write_line(const std::string& row) {
    if(row.empty() || row.find_first_of("\r\n")!=std::string::npos)
      throw std::invalid_argument("JSONL requires one nonempty line");
    const std::string bytes=row+"\n";
    if(limit_ && bytes.size()>limit_)throw std::runtime_error("JSONL row exceeds log limit");
    if(limit_ && bytes.size()>limit_-used_) {
      struct stat owned,current;
      if(fstat(fd_,&owned)!=0 || lstat(path_.c_str(),&current)!=0 ||
         !S_ISREG(current.st_mode) || current.st_dev!=owned.st_dev || current.st_ino!=owned.st_ino)
        throw std::runtime_error("JSONL output identity changed");
      const std::string temporary=path_+".next."+std::to_string(getpid());
      int next=create(temporary); // Never remove a pre-existing temporary.
      try {
        write_all(next,bytes);
        // Close before renaming: DrvFS cannot fstat a descriptor whose original
        // path has been renamed. The published old path remains readable until
        // the atomic replacement, and no reader holds our writer descriptor.
        const int close_result=close(next);next=-1;
        if(close_result!=0)throw std::runtime_error("JSONL close failed");
        close(fd_);fd_=-1;
        if(rename(temporary.c_str(),path_.c_str())!=0)throw std::runtime_error("JSONL rotation failed");
        fd_=open(path_.c_str(),O_WRONLY|O_APPEND|O_CLOEXEC|O_NOFOLLOW);
        if(fd_<0)throw std::runtime_error("JSONL reopen failed");
      }catch(...){if(next>=0)close(next);unlink(temporary.c_str());throw;}
      used_=bytes.size();
    }else{
      write_all(fd_,bytes);used_+=bytes.size();
    }
  }
 private:
  static int create(const std::string& path) {
    const int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0644);
    if(fd<0){
      if(errno==EEXIST)throw std::runtime_error("output already exists");
      throw std::runtime_error("cannot create JSONL: "+std::to_string(errno));
    }
    return fd;
  }
  static void write_all(int fd,const std::string& bytes){
    std::size_t done=0;
    while(done<bytes.size()){
      const ssize_t n=write(fd,bytes.data()+done,bytes.size()-done);
      if(n<0 && errno==EINTR)continue;
      if(n<=0)throw std::runtime_error("JSONL write failed");
      done+=static_cast<std::size_t>(n);
    }
  }
  std::string path_;
  std::size_t limit_,used_=0;
  int fd_;
};
} // namespace soundjie
#endif

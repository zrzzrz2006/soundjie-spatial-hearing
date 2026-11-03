#include "framebuffer_surface.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace soundjie {

uint16_t rgb565(Rgb c) {
  return static_cast<uint16_t>(((c.r >> 3) << 11) | ((c.g >> 2) << 5) | (c.b >> 3));
}
uint32_t xrgb8888(Rgb c) {
  return 0xff000000u | (static_cast<uint32_t>(c.r) << 16) |
         (static_cast<uint32_t>(c.g) << 8) | c.b;
}

Surface::Surface(int width, int height)
    : width_(width), height_(height), pixels_(static_cast<std::size_t>(width) * height) {
  if (width <= 0 || height <= 0) throw std::invalid_argument("invalid surface geometry");
}
void Surface::clear(Rgb color) { std::fill(pixels_.begin(), pixels_.end(), color); }
void Surface::pixel(int x, int y, Rgb color) {
  if (x >= 0 && y >= 0 && x < width_ && y < height_)
    pixels_[static_cast<std::size_t>(y) * width_ + x] = color;
}
void Surface::rect(int x, int y, int width, int height, Rgb color) {
  for (int row = std::max(0, y); row < std::min(height_, y + height); ++row)
    for (int column = std::max(0, x); column < std::min(width_, x + width); ++column)
      pixel(column, row, color);
}
void Surface::line(int x0, int y0, int x1, int y1, int thickness, Rgb color) {
  const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int error = dx + dy;
  for (;;) {
    rect(x0 - thickness / 2, y0 - thickness / 2, thickness, thickness, color);
    if (x0 == x1 && y0 == y1) break;
    const int twice = 2 * error;
    if (twice >= dy) { error += dy; x0 += sx; }
    if (twice <= dx) { error += dx; y0 += sy; }
  }
}
void Surface::circle(int cx, int cy, int radius, int thickness, Rgb color) {
  for (int y = cy - radius - thickness; y <= cy + radius + thickness; ++y) {
    for (int x = cx - radius - thickness; x <= cx + radius + thickness; ++x) {
      const double distance = std::sqrt(static_cast<double>((x-cx)*(x-cx)+(y-cy)*(y-cy)));
      if (std::abs(distance - radius) <= thickness / 2.0) pixel(x, y, color);
    }
  }
}
void Surface::write_ppm_create_only(const std::string& path) const {
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) throw std::runtime_error("offscreen output already exists or cannot be created");
  std::string header = "P6\n" + std::to_string(width_) + " " + std::to_string(height_) + "\n255\n";
  std::string payload;
  payload.reserve(header.size() + pixels_.size() * 3);
  payload += header;
  for (const Rgb color : pixels_) {
    payload.push_back(static_cast<char>(color.r));
    payload.push_back(static_cast<char>(color.g));
    payload.push_back(static_cast<char>(color.b));
  }
  std::size_t written = 0;
  while (written < payload.size()) {
    const ssize_t count = write(fd, payload.data() + written, payload.size() - written);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) { close(fd); throw std::runtime_error("PPM write failed"); }
    written += static_cast<std::size_t>(count);
  }
  if (close(fd) != 0) throw std::runtime_error("PPM close failed");
}

void write_linux_framebuffer(const std::string& path, const Surface& surface) {
  const int fd = open(path.c_str(), O_RDWR);
  if (fd < 0) throw std::runtime_error("framebuffer open failed");
  fb_fix_screeninfo fixed{}; fb_var_screeninfo variable{};
  if (ioctl(fd, FBIOGET_FSCREENINFO, &fixed) != 0 ||
      ioctl(fd, FBIOGET_VSCREENINFO, &variable) != 0) {
    close(fd); throw std::runtime_error("framebuffer ioctl validation failed");
  }
  if (!((variable.bits_per_pixel == 16 && variable.red.length == 5 && variable.green.length == 6) ||
        (variable.bits_per_pixel == 32 && variable.red.length == 8 && variable.green.length == 8))) {
    close(fd); throw std::runtime_error("framebuffer pixel format unsupported");
  }
  const std::size_t length = fixed.smem_len;
  const unsigned bytes_per_pixel = variable.bits_per_pixel / 8;
  if(variable.xres==0||variable.yres==0||variable.xres>8192||variable.yres>8192||
      (static_cast<uint64_t>(variable.xoffset)+variable.xres)*bytes_per_pixel>fixed.line_length||
      (static_cast<uint64_t>(variable.yoffset)+variable.yres)*fixed.line_length>length){
    close(fd);throw std::runtime_error("framebuffer geometry outside mapped memory");
  }
  void* mapped = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) { close(fd); throw std::runtime_error("framebuffer mmap failed"); }
  // ARMv7 integer division is expensive. The horizontal map is identical on every row.
  std::vector<unsigned> source_x(variable.xres);
  for(unsigned x=0;x<variable.xres;++x)source_x[x]=(static_cast<uint64_t>(x)*surface.width())/variable.xres;
  std::vector<uint8_t> packed(variable.xres*bytes_per_pixel);
  for (unsigned y = 0; y < variable.yres; ++y) {
    const int sy = static_cast<int>((static_cast<uint64_t>(y) * surface.height()) / variable.yres);
    const Rgb* row=surface.pixels().data()+static_cast<std::size_t>(sy)*surface.width();
    for(unsigned x=0;x<variable.xres;++x){
      const Rgb c=row[source_x[x]];
      if(bytes_per_pixel==2){const uint16_t v=rgb565(c);std::memcpy(packed.data()+x*2,&v,2);}
      else{const uint32_t v=xrgb8888(c);std::memcpy(packed.data()+x*4,&v,4);}
    }
    uint8_t* target=static_cast<uint8_t*>(mapped)+(y+variable.yoffset)*fixed.line_length+variable.xoffset*bytes_per_pixel;
    std::memcpy(target,packed.data(),packed.size());
  }
  msync(mapped, length, MS_SYNC); munmap(mapped, length); close(fd);
}

}  // namespace soundjie

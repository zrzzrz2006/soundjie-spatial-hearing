#ifndef SOUNDJIE_FRAMEBUFFER_SURFACE_H
#define SOUNDJIE_FRAMEBUFFER_SURFACE_H

#include <cstdint>
#include <string>
#include <vector>

namespace soundjie {

struct Rgb { uint8_t r, g, b; };

uint16_t rgb565(Rgb color);
uint32_t xrgb8888(Rgb color);

class Surface {
 public:
  Surface(int width, int height);
  int width() const { return width_; }
  int height() const { return height_; }
  void clear(Rgb color);
  void pixel(int x, int y, Rgb color);
  void rect(int x, int y, int width, int height, Rgb color);
  void line(int x0, int y0, int x1, int y1, int thickness, Rgb color);
  void circle(int cx, int cy, int radius, int thickness, Rgb color);
  void write_ppm_create_only(const std::string& path) const;
  const std::vector<Rgb>& pixels() const { return pixels_; }

 private:
  int width_;
  int height_;
  std::vector<Rgb> pixels_;
};

void write_linux_framebuffer(const std::string& path, const Surface& surface);

}  // namespace soundjie
#endif

// HOST TEST ONLY. Never link this translation unit into an ARM release.
#include "n2cube.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#ifndef SOUNDJIE_PHYSICAL_OUTPUTS
#define SOUNDJIE_PHYSICAL_OUTPUTS 8
#endif
#ifndef SOUNDJIE_STORAGE_OUTPUTS
#define SOUNDJIE_STORAGE_OUTPUTS SOUNDJIE_PHYSICAL_OUTPUTS
#endif
static DPUKernel kernel;
static DPUTask task;
static int8_t input[6144];
static int8_t output[8] = {-2, -1, 0, 4, 1, -3, 2, 0};
static bool fail(const char* name) {
  const char* value = std::getenv("SJ_TEST_FAIL");
  return value && std::strcmp(value, name) == 0;
}
static void trace(const char* event) {
  std::ofstream(std::getenv("SJ_TEST_TRACE"), std::ios::app) << event << '\n';
}
void dpuSetExceptionMode(int) {}
int dpuOpen() { trace("open"); std::printf("library diagnostic\n"); return fail("open") ? -1 : 0; }
int dpuClose() { trace("close"); return 0; }
DPUKernel* dpuLoadKernel(const char* name) {
  trace("load"); return !std::strcmp(name, "soundjie_d3") && !fail("load") ? &kernel : nullptr;
}
int dpuDestroyKernel(DPUKernel*) { trace("destroy_kernel"); return 0; }
DPUTask* dpuCreateTask(DPUKernel*, int) { trace("create"); return fail("create") ? nullptr : &task; }
int dpuDestroyTask(DPUTask*) { trace("destroy_task"); return 0; }
int dpuGetInputTensorSize(DPUTask*, const char* node, int) {
  return !std::strcmp(node, "conv1") && !fail("input_size") ? 6144 : 0;
}
int8_t* dpuGetInputTensorAddress(DPUTask*, const char*, int) { return fail("input_null") ? nullptr : input; }
float dpuGetInputTensorScale(DPUTask*, const char*, int) { return std::strtof(std::getenv("SJ_TEST_INPUT_SCALE"), nullptr); }
float dpuGetOutputTensorScale(DPUTask*, const char*, int) { return std::strtof(std::getenv("SJ_TEST_OUTPUT_SCALE"), nullptr); }
int dpuRunTask(DPUTask*) {
  trace("run");
  std::ofstream(std::getenv("SJ_TEST_TENSOR"), std::ios::binary).write(reinterpret_cast<char*>(input), sizeof(input));
  if (fail("zero")) std::memset(output, 0, sizeof(output));
  if (SOUNDJIE_STORAGE_OUTPUTS > SOUNDJIE_PHYSICAL_OUTPUTS)
    output[SOUNDJIE_PHYSICAL_OUTPUTS] = fail("padding") ? 127 : 0;
  return fail("run") ? -1 : 0;
}
int dpuGetOutputTensorSize(DPUTask*, const char* node, int) {
  return !std::strcmp(node, "fc8") && !fail("output_size") ? SOUNDJIE_STORAGE_OUTPUTS : 9;
}
int8_t* dpuGetOutputTensorAddress(DPUTask*, const char*, int) { return fail("output_null") ? nullptr : output; }

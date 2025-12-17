#pragma once

namespace GLBackend {
static void* glHandle = nullptr;
void runBenchmark();

struct GLresult {
  float totalElapsed;
  float penalty; // Number of frames not computed
};

GLresult runTriangleBenchmark(int width, int height, int frames, const char** fragShaderSrc);
GLresult runMemBenchmark(int width, int height, int frames, int texWidth, int texHeight);
}; // namespace GLBackend
#pragma once

namespace GLBackend {
static void* glHandle = nullptr;
void runBenchmark();

float runTriangleBenchmark(int width, int height, int frames);
}; // namespace GLBackend
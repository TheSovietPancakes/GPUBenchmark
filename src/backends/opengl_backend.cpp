#include "opengl_backend.hpp"
#include "../shared/shared.hpp"
#include "modules/gl_backends/aluHeavyFrag.hpp"
#include "modules/gl_backends/memHeavyFrag.hpp"
#include "modules/gl_backends/triangleFrag.hpp"
#include "modules/gl_backends/triangleVer.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <GL/gl.h>
#include <GLES3/gl3.h>
#include <GLFW/glfw3.h>
#include <math.h>

GLBackend::GLresult GLBackend::runTriangleBenchmark(int width, int height, int frames, const char** fragShaderSrc) {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow* window = glfwCreateWindow(width, height, "OpenGL Benchmark", nullptr, nullptr);
  if (!window) {
    const char* err;
    glfwGetError(&err);
    std::cerr << "Failed to create GLFW window: " << err << "\n";
    glfwTerminate();
    return {.totalElapsed = 0.0f, .penalty = -1.0f};
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);
  glfwSetWindowSize(glfwGetCurrentContext(), width, height);

  glViewport(0, 0, width, height);

  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &triangleVert_src, NULL);
  glCompileShader(vs);

  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, fragShaderSrc, NULL);
  glCompileShader(fs);

  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  glUseProgram(prog);

  auto checkShader = [](GLuint s) {
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
      char log[4096];
      glGetShaderInfoLog(s, sizeof(log), nullptr, log);
      std::cerr << log << "\n";
      std::abort();
    }
  };

  auto checkProgram = [](GLuint p) {
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
      char log[4096];
      glGetProgramInfoLog(p, sizeof(log), nullptr, log);
      std::cerr << log << "\n";
      std::abort();
    }
  };

  checkShader(vs);
  checkShader(fs);
  checkProgram(prog);

  float verts[] = {-0.6f, -0.4f, 0.6f, -0.4f, 0.0f, 0.6f};
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

  GLuint query;
  glGenQueries(1, &query);
  glBeginQuery(GL_TIME_ELAPSED, query);
  int framesPassed = 0;
  while (framesPassed < frames && !glfwWindowShouldClose(window)) {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glfwSwapBuffers(window);
    // glfwPollEvents();
    glFinish();
    // Spin vertices
    float angle = 0.01f;
    float cosA = cosf(angle);
    float sinA = sinf(angle);
    for (int j = 0; j < std::size(verts); j += 2) {
      float x = verts[j];
      float y = verts[j + 1];
      float newX = x * cosA - y * sinA;
      float newY = x * sinA + y * cosA;
      verts[j] = newX;
      verts[j + 1] = newY;
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);
    ++framesPassed;
  }

  glEndQuery(GL_TIME_ELAPSED);
  GLuint timeElapsed = 0;
  glGetQueryObjectuiv(query, GL_QUERY_RESULT, &timeElapsed);
  // Tell the window it is time to close
  glfwSetWindowShouldClose(window, GLFW_TRUE);
  // Cleanup
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);
  glDeleteShader(vs);
  glDeleteShader(fs);
  glDeleteProgram(prog);
  glfwDestroyWindow(window);

  // TODO: Pick either time or pps as the main determinant of score. I am unsure which is better right now LOL
  size_t pixelTotal = width * height * framesPassed;
  float pixelsPerSecond = static_cast<float>(pixelTotal) / (static_cast<float>(timeElapsed) / 1e9f);

  float penalty = static_cast<float>(frames - framesPassed);
  float totalTimeSec = static_cast<float>(timeElapsed) / 1e9f;
  return {.totalElapsed = totalTimeSec, .penalty = penalty};
}

GLBackend::GLresult GLBackend::runMemBenchmark(int width, int height, int frames, int texWidth, int texHeight) {
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  GLFWwindow* window = glfwCreateWindow(width, height, "Memory Benchmark", nullptr, nullptr);
  if (!window) {
    const char* err;
    glfwGetError(&err);
    std::cerr << "Failed to create GLFW window: " << err << "\n";
    glfwTerminate();
    return {.totalElapsed = 0.0f, .penalty = -1.0f};
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(0); // uncouple from refresh rate
  glfwSetWindowSize(glfwGetCurrentContext(), width, height);
  glViewport(0, 0, width, height);

  // --- Compile shaders ---
  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &triangleVert_src, nullptr);
  glCompileShader(vs);

  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &memHeavyFrag_src, nullptr);
  glCompileShader(fs);

  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glLinkProgram(prog);
  glUseProgram(prog);

  // --- Setup triangle ---
  float verts[] = {-0.6f, -0.4f, 0.6f, -0.4f, 0.0f, 0.6f};
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

  // --- Setup texture ---
  GLuint tex;
  glGenTextures(1, &tex);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  // Allocate texture
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texWidth, texHeight, 0, GL_RGBA, GL_FLOAT, nullptr);

  // Initialize texture with non-zero data
  std::vector<float> initData(texWidth * texHeight * 4);
  for (size_t i = 0; i < initData.size(); ++i)
    initData[i] = float(i % 256) / 255.0f;
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texWidth, texHeight, GL_RGBA, GL_FLOAT, initData.data());

  // Bind uniforms
  glUseProgram(prog);
  glUniform1i(glGetUniformLocation(prog, "uTex"), 0);
  glUniform2i(glGetUniformLocation(prog, "uTexSize"), texWidth, texHeight);

  // --- GPU timing ---
  GLuint query;
  glGenQueries(1, &query);
  glBeginQuery(GL_TIME_ELAPSED, query);

  int framesPassed = 0;
  while (framesPassed < frames && !glfwWindowShouldClose(window)) {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glfwSwapBuffers(window);
    glFinish(); // ensure timing includes GPU work
    ++framesPassed;
  }

  glEndQuery(GL_TIME_ELAPSED);
  GLuint timeElapsed = 0;
  glGetQueryObjectuiv(query, GL_QUERY_RESULT, &timeElapsed);
  glfwSetWindowShouldClose(window, GLFW_TRUE);
  // --- Cleanup ---
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);
  glDeleteShader(vs);
  glDeleteShader(fs);
  glDeleteProgram(prog);
  glDeleteTextures(1, &tex);
  glfwDestroyWindow(window);

  // --- Compute metrics ---
  size_t pixelTotal = width * height * framesPassed;
  float pixelsPerSecond = static_cast<float>(pixelTotal) / (static_cast<float>(timeElapsed) / 1e9f);
  float penalty = static_cast<float>(frames - framesPassed);
  float totalTimeSec = static_cast<float>(timeElapsed) / 1e9f;

  return {.totalElapsed = totalTimeSec, .penalty = penalty};
}

void GLBackend::runBenchmark() {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW\n";
    return;
  }
  int FRAMES = 500;
  int WIDTH = 1920;
  int HEIGHT = 1080;
  std::cout << OPENGL << "Running triangle benchmark (" << WIDTH << "x" << HEIGHT << ", " << FRAMES << " frames)...\n";
  GLresult result = runTriangleBenchmark(WIDTH, HEIGHT, FRAMES, &triangleFrag_src);
  if (result.totalElapsed < 0.0f) {
    std::cerr << OPENGL << "Benchmark failed.\n";
    glfwTerminate();
    return;
  }
  if (result.penalty > 0.0f) {
    std::cout << OPENGL << "Triangle benchmark incomplete, finished in " << std::fixed << std::setprecision(5) << result.totalElapsed
              << " seconds with penalty of " << result.penalty << " frames.\n";
    std::string text = "Would you like to retry the benchmark to get \na complete result, without penalty?";
    wrapped_print(std::string(OPENGL), text);
    // TODO
  }
  std::cout << OPENGL << "Triangle benchmark completed in " << std::fixed << std::setprecision(5) << result.totalElapsed << " seconds.\n";

  GLresult aluResult = runTriangleBenchmark(WIDTH, HEIGHT, FRAMES, &aluHeavyFrag_src);
  if (aluResult.totalElapsed < 0.0f) {
    std::cerr << OPENGL << "ALU-heavy benchmark failed.\n";
    glfwTerminate();
    return;
  }
  if (aluResult.penalty > 0.0f) {
    std::cout << OPENGL << "ALU-heavy benchmark incomplete, finished in " << std::fixed << std::setprecision(5) << aluResult.totalElapsed
              << " seconds with penalty of " << aluResult.penalty << " frames.\n";
    std::string text = "Would you like to retry the benchmark to get \na complete result, without penalty?";
    wrapped_print(std::string(OPENGL), text);
    // TODO
  }
  std::cout << OPENGL << "ALU-heavy benchmark completed in " << std::fixed << std::setprecision(5) << aluResult.totalElapsed << " seconds.\n";

  int TEX_WIDTH = 4096;
  GLresult memResult = runMemBenchmark(WIDTH, HEIGHT, FRAMES, TEX_WIDTH, TEX_WIDTH);
  if (memResult.penalty < 0.0f) {
    std::cerr << OPENGL << "Memory-heavy benchmark failed.\n";
    glfwTerminate();
    return;
  }
  if (memResult.penalty > 0.0f) {
    std::cout << OPENGL << "Memory-heavy benchmark incomplete, finished in " << std::fixed << std::setprecision(5) << memResult.totalElapsed
              << " seconds with penalty of " << memResult.penalty << " frames.\n";
    std::string text = "Would you like to retry the benchmark to get \na complete result, without penalty?";
    wrapped_print(std::string(OPENGL), text);
    // TODO
  }
  std::cout << OPENGL << "Memory-heavy benchmark completed in " << std::fixed << std::setprecision(5) << memResult.totalElapsed << " seconds.\n";
  glfwTerminate();
}

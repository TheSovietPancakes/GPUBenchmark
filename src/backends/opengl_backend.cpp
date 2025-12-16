#include "opengl_backend.hpp"
#include "../shared/shared.hpp"
#include "modules/gl_backends/triangleFrag.hpp"
#include "modules/gl_backends/triangleVer.hpp"

#include <cstring>
#include <iomanip>
#include <iostream>

#include <EGL/egl.h>
#include <GL/gl.h>
#include <GLES3/gl3.h>

float GLBackend::runTriangleBenchmark(int width, int height, int frames) {
  // ===== EGL pbuffer setup =====
  EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (eglDisplay == EGL_NO_DISPLAY) {
    std::cerr << "Failed to get EGL display\n";
    return -1.0f;
  }
  if (!eglInitialize(eglDisplay, nullptr, nullptr)) {
    std::cerr << "Failed to initialize EGL\n";
    return -1.0f;
  }

  EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                            EGL_PBUFFER_BIT,
                            EGL_RENDERABLE_TYPE,
                            EGL_OPENGL_ES2_BIT,
                            EGL_RED_SIZE,
                            8,
                            EGL_GREEN_SIZE,
                            8,
                            EGL_BLUE_SIZE,
                            8,
                            EGL_ALPHA_SIZE,
                            8,
                            EGL_NONE};
  EGLConfig config;
  EGLint numConfigs;
  if (!eglChooseConfig(eglDisplay, configAttribs, &config, 1, &numConfigs) || numConfigs == 0) {
    std::cerr << "Failed to choose EGL config\n";
    return -1.0f;
  }

  EGLint pbufferAttribs[] = {
      EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE,
  };
  EGLSurface eglSurface = eglCreatePbufferSurface(eglDisplay, config, pbufferAttribs);
  if (eglSurface == EGL_NO_SURFACE) {
    std::cerr << "Failed to create EGL pbuffer surface\n";
    return -1.0f;
  }

  if (!eglBindAPI(EGL_OPENGL_ES_API)) {
    std::cerr << "Failed to bind OpenGL ES API\n";
    return -1.0f;
  }

  EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
  EGLContext eglContext = eglCreateContext(eglDisplay, config, EGL_NO_CONTEXT, ctxAttribs);
  if (eglContext == EGL_NO_CONTEXT) {
    std::cerr << "Failed to create EGL context\n";
    return -1.0f;
  }

  if (!eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
    std::cerr << "Failed to make EGL context current\n";
    return -1.0f;
  }

  // ===== GL setup =====
  glViewport(0, 0, width, height);

  const char* vsSrc = R"(attribute vec2 pos; void main() { gl_Position = vec4(pos,0.0,1.0); })";
  const char* fsSrc = R"(void main() { gl_FragColor = vec4(1.0,0.5,0.2,1.0); })";

  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vsSrc, NULL);
  glCompileShader(vs);

  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fsSrc, NULL);
  glCompileShader(fs);

  GLuint prog = glCreateProgram();
  glAttachShader(prog, vs);
  glAttachShader(prog, fs);
  glBindAttribLocation(prog, 0, "pos");
  glLinkProgram(prog);
  glUseProgram(prog);

  float verts[] = {-0.6f, -0.4f, 0.6f, -0.4f, 0.0f, 0.6f};
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

  // ===== Benchmark loop =====
  GLuint query;
  glGenQueries(1, &query);
  glBeginQuery(GL_TIME_ELAPSED, query);
  unsigned char output[width * height * 4];
  for (int i = 0; i < frames; ++i) {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, output); // Ensure rendering is done
    eglSwapBuffers(eglDisplay, eglSurface);
  }
  glEndQuery(GL_TIME_ELAPSED);
  GLuint timeElapsed;
  glGetQueryObjectuiv(query, GL_QUERY_RESULT, &timeElapsed);
  // Cleanup
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);
  glDeleteShader(vs);
  glDeleteShader(fs);
  glDeleteProgram(prog);

  eglMakeCurrent(eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroyContext(eglDisplay, eglContext);
  eglDestroySurface(eglDisplay, eglSurface);
  eglTerminate(eglDisplay);

  return timeElapsed / 1e9f; // Convert nanoseconds to seconds
}

void GLBackend::runBenchmark() {
  int FRAMES = 500;
  int WIDTH = 1920;
  int HEIGHT = 1080;
  std::cout << OPENGL << "Running triangle benchmark (" << WIDTH << "x" << HEIGHT << ", " << FRAMES << " frames)...\n";
  float timeTaken = runTriangleBenchmark(WIDTH, HEIGHT, FRAMES);
  std::cout << OPENGL << "Triangle benchmark completed in " << std::fixed << std::setprecision(5) << timeTaken << " seconds.\n";
}

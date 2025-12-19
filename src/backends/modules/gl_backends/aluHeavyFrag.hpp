// Auto-generated
static const char* aluHeavyFrag_src = R"ptx(
#version 330 core
out vec4 color;
void main() {
  // Perform heavy ALU operations to stress the GPU
  float x = gl_FragCoord.x;
  float y = gl_FragCoord.y;
  float result = 0.0;
  const float ITER = 2000.0; // more iterations = more ALU stress
  for (float i = 0; i < ITER; ++i) {
    result += sin(x * 0.01 + i) * cos(y * 0.01 - i) - tanh(i * 0.001) + exp(-i * 0.0001);
    result = sqrt(abs(result)) + log(abs(result) + 1.0);
  }
  color = vec4(fract(result), fract(result * 1.3), fract(result * 1.7), 1.0);
}
)ptx";

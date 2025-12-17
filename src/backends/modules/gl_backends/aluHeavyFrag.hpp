// Auto-generated
static const char* aluHeavyFrag_src = R"ptx(
#version 330 core
out vec4 color;
void main() {
  // Perform heavy ALU operations to stress the GPU
  float x = gl_FragCoord.x;
  float y = gl_FragCoord.y;
  float result = 0.0;
  for (int i = 0; i < 1000; ++i) {
    result += sin(x * 0.01 + float(i)) * cos(y * 0.01 + float(i));
    result = sqrt(abs(result)) + log(abs(result) + 1.0);
  }
  color = vec4(fract(result), fract(result * 1.3), fract(result * 1.7), 1.0);
}
)ptx";

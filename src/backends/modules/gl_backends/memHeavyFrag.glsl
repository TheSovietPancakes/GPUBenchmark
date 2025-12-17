#version 330 core
uniform sampler2D uTex;
uniform ivec2 uTexSize;
out vec4 color;

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

void main() {
    ivec2 frag = ivec2(gl_FragCoord.xy);
    float seed = float(frag.x) + float(frag.y) * 4096.0;
    vec4 acc = vec4(0.0);
    const int ITER = 10000; // more iterations = more memory stress

    for (int i = 0; i < ITER; i++) {
      acc += texture(uTex, (vec2(frag) + 0.5) / vec2(uTexSize));
    }
    color = acc / float(ITER);
}

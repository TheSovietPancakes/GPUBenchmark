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
    const float ITER = 1000.0; // more iterations = more memory stress

    for (float i = 0.0; i < ITER; i++) {
      // Look up the texture multiple times to stress memory bandwidth
      // in the GPU (NOT over the bus)
      acc += vec4(sin(texture(uTex, (vec2(frag) + 0.5) / vec2(uTexSize)).r + seed + i), 
            cos(texture(uTex, (vec2(frag) + 0.5) / vec2(uTexSize)).g + seed + i), 
            sin(texture(uTex, (vec2(frag) + 0.5) / vec2(uTexSize)).b + seed * 0.5 + i), 
            cos(texture(uTex, (vec2(frag) + 0.5) / vec2(uTexSize)).a + seed * 0.5 + i)) * 0.01;
    }
    color = acc / float(ITER);
}

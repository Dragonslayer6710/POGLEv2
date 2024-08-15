// block.vert
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aScale;
layout (location = 2) in vec3 aColor;
layout (location = 3) in float aAlpha;
out vec4 vColor;

void main(){
    gl_Position = vec4(aScale * aPos, 0.0, 1.0);
    vColor = vec4(aColor, aAlpha);
}

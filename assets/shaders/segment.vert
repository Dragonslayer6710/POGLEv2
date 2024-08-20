// default.vert
#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aAlpha;

out vec4 vColour;

uniform mat4 uView;
uniform mat4 uProjection;

void main(){
    gl_Position = uProjection * uView * vec4(aPos, 1.0);
    vColour = vec4(aColor, aAlpha);
}
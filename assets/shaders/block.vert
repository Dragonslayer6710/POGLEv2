// block.vert
#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexUV;
layout (location = 2) in vec2 aTexPos;
layout (location = 3) in vec2 aTexSize;
layout (location = 4) in mat4 aModelMatrix;

out vec2 vTexUV;

uniform mat4 uView;
uniform mat4 uProjection;

void main(){
    gl_Position = uProjection * uView * aModelMatrix * vec4(aPos, 1.0);
    vTexUV = aTexUV * aTexSize + aTexPos;
}

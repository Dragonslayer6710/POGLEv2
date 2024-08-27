// block.vert
#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexUV;
layout (location = 2) in int aSideID;
layout (location = 3) in int aTexSizeID;
layout (location = 4) in int aTexOffsetID;

out vec2 vTexUV;

layout (std140) uniform Matrices
{
    mat4 uProjection;
    mat4 uView;
};

uniform mat4 uSideMats[6];
uniform mat4 uTexSizes[1];
uniform mat4 uTexOffsets[3];



void main(){
    gl_Position = uProjection * uView * aModelMatrix * vec4(aPos, 1.0);
    vTexUV = aTexUV * aTexSize + aTexPos;
}

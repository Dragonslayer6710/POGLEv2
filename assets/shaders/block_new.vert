// block.vert
#version 450 core
// Per Vertex Data
layout (location = 0) in vec2 aLocalPosXY;
layout (location = 1) in vec2 aTexUV;
// Per instance data
// - Changes every instance (side of cube)
layout (location = 2) in vec2 aTexPos;
layout (location = 3) in vec2 aTexSize;
layout (location = 4) in int aSideID;
// - Changes every 6 instances (every cube)
layout (location = 5) in vec3 aWorldPos;

out vec2 vTexUV;

layout (std140) uniform Matrices
{
    mat4 uProjection;
    mat4 uView;
};

layout (std140) uniform BlockSides
{
    mat4 uBlockSides[6];
};

void main(){
    // First, transform the local position by the block transformation matrix.
    vec4 localBlockPos = uBlockSides[aSideID] * vec4(aLocalPosXY, 0.0, 1.0);

    // Then, add the world position (with w = 1.0 to ensure proper translation).
    vec4 worldPos = localBlockPos + vec4(aWorldPos, 0.0);

    // Finally, apply the view and projection transformations.
    gl_Position = uProjection * uView * worldPos;
    vTexUV = aTexUV * aTexSize + aTexPos;
}

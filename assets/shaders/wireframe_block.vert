// default.vert
#version 450 core
layout (location = 0) in vec3 aLocalPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aAlpha;
layout (location = 3) in vec3 aWorldPos;

out vec4 vColour;

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
    vec4 localBlockPos = uBlockSides[gl_InstanceID] * vec4(aLocalPos, 1.0);

    // Then, add the world position (with w = 1.0 to ensure proper translation).
    vec4 worldPos = localBlockPos + vec4(aWorldPos, 0.0);

    // Finally, apply the view and projection transformations.
    gl_Position = uProjection * uView * worldPos;

    vColour = vec4(aColor, aAlpha);
}

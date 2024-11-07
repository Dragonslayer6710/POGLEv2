// block.vert
#version 450 core
// Per Vertex
layout (location = 0) in vec3 a_Position;
layout (location = 1) in float a_Alpha;
layout (location = 2) in vec2 a_TexUV;

// Per Block
layout (location = 3) in mat4 a_Model;

// Per Face
layout (location = 7) in int a_FaceID;
layout (location = 8) in int a_TexPosID;
layout (location = 9) in int a_TexSizeID;

out vec2 vTexUV;

layout (std140) uniform ub_Matrices
{
    mat4 u_Projection;
    mat4 u_View;
};

#define NUM_TEXTURES 4
#define NUM_SIZES 1

layout (std140) uniform ub_FaceData
{
    mat4 u_FaceTransforms[6];
    vec2 u_TexPositions[NUM_TEXTURES];
    vec2 u_TexSizes[NUM_SIZES];
};

void main(){
    // Finally, apply the view and projection transformations.
    // gl_Position = vec4(a_Position, 1.0);
    gl_Position = u_Projection * u_View  * u_FaceTransforms[0] * vec4(a_Position, 1.0);

}

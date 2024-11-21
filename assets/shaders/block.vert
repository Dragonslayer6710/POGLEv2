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
layout (location = 8) in int a_FaceTexID;
layout (location = 9) in int a_FaceTexSizeID;

out vec2 vTexUV;

layout (std140) uniform ub_Matrices
{
    mat4 u_Projection;
    mat4 u_View;
};

#define NUM_SUB_TEXTURES 4
#define NUM_SIZES 1
#define USE_SIZES true
#define USE_SUB_TEXTURES true

layout (std140) uniform ub_FaceTransforms
{
    mat4 u_FaceTransforms[6];
};
layout (std140) uniform ub_FaceTexPositions
{
    vec2 u_TexPositions[NUM_SUB_TEXTURES];
};
layout (std140) uniform ub_FaceTexSizes
{
    vec2 u_TexSizes[NUM_SIZES];
};

void main(){
    // Finally, apply the view and projection transformations.
    // gl_Position = vec4(a_Position, 1.0);
    mat4 face_transform;
    face_transform = a_Model * u_FaceTransforms[a_FaceID];

    vec2 tex_size;
    tex_size = u_TexSizes[a_FaceTexSizeID];
    vec2 tex_pos;
    tex_pos = u_TexPositions[a_FaceTexID];

    if ((-1 < a_FaceTexID) && (a_FaceTexID < NUM_SUB_TEXTURES)) {
        gl_Position = u_Projection * u_View * face_transform * vec4(a_Position, 1.0);
        vTexUV = a_TexUV;
        if (USE_SIZES)
            vTexUV *= tex_size;
        if (USE_SUB_TEXTURES)
            vTexUV += tex_pos;
    }
    else {
        gl_Position = vec4(0);
    }
}

// block.vert
#version 450 core
// Per Vertex
layout (location = 0) in vec3 a_Position;
layout (location = 1) in float a_Alpha;

// Per Block
layout (location = 2) in mat4 a_Model;


layout (std140) uniform ub_Matrices
{
    mat4 u_Projection;
    mat4 u_View;
};

out vec4 v_Color;

void main(){
    gl_Position = u_Projection * u_View * a_Model * vec4(a_Position, 1.0);
    v_Color = vec4(vec3(1), a_Alpha);
}

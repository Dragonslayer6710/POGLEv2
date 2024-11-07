// default.frag
#version 450 core

in vec2 vTexUV;

out vec4 FragColor;

uniform sampler2D tex0;

void main(){
    FragColor = texture(tex0, vTexUV);
    if (FragColor == vec4(0,0,0,1) || vTexUV == vec2(99))
        FragColor = vec4(1,1,1,1);

}
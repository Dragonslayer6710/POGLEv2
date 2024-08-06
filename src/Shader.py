from Buffer import *

_shaderExt = {
    GL_VERTEX_SHADER: "vert",
    GL_FRAGMENT_SHADER: "frag"
}
_shaderType = {
    GL_VERTEX_SHADER: "VERTEX",
    GL_FRAGMENT_SHADER: "FRAGMENT"
}
_linetab = "\n\t"
_line = "\n"

class Shader:
    def __init__(self, shaderName: str, shaderType: GLenum):
        self.ID = glCreateShader(shaderType)

        self.shaderPath = f"../assets/shader/{shaderName}.{_shaderExt[shaderType]}"
        with open(self.shaderPath) as f:
            self.source = f.read()
        glShaderSource(self.ID, self.source,)

        glCompileShader(self.ID)
        if not glGetShaderiv(self.ID, GL_COMPILE_STATUS):
            print(
                f"ERROR::SHADER::{_shaderType[shaderType]}::COMPILATION_FAILED:{_linetab + _linetab.join(glGetShaderInfoLog(self.ID).decode().split(_line))}\n")


class VertexShader(Shader):
    def __init__(self, shaderName: str = "default"):
        super().__init__(shaderName, GL_VERTEX_SHADER)


class FragmentShader(Shader):
    def __init__(self, shaderName: str = "default"):
        super().__init__(shaderName, GL_FRAGMENT_SHADER)


class ShaderProgram:
    def __init__(self, vertex: VertexShader = None, fragment: FragmentShader = None):
        if not vertex: vertex = VertexShader()
        if not fragment: fragment = FragmentShader()

        self.ID = glCreateProgram()

        glAttachShader(self.ID, vertex.ID)
        glAttachShader(self.ID, fragment.ID)

        glLinkProgram(self.ID)
        if not glGetProgramiv(self.ID, GL_LINK_STATUS):
            print(
                f"ERROR::SHADER::PROGRAM::LINKING_FAILED:{_linetab + _linetab.join(glGetProgramInfoLog(self.ID).decode().split(_line))}\n")

    def use(self):
        glUseProgram(self.ID)
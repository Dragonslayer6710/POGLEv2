from POGLE.Buffer import *

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
    def __init__(self, vsName: str = "default", fsName: str = "default"):
        vertexShader = VertexShader(vsName)
        fragmentShader = FragmentShader(fsName)

        self.ID = glCreateProgram()

        self.uniLocations = {}

        glAttachShader(self.ID, vertexShader.ID)
        glAttachShader(self.ID, fragmentShader.ID)

        glLinkProgram(self.ID)
        if not glGetProgramiv(self.ID, GL_LINK_STATUS):
            print(
                f"ERROR::SHADER::PROGRAM::LINKING_FAILED:{_linetab + _linetab.join(glGetProgramInfoLog(self.ID).decode().split(_line))}\n")

    def use(self):
        glUseProgram(self.ID)

    def _cache_uniform(self, name: str):
        self.uniLocations[name] = glGetUniformLocation(self.ID, name)
        return self.uniLocations[name]

    def _gul(self, name: str): # Get uniform location
        return self.uniLocations.get(name, self._cache_uniform(name))

    def setMat4(self, name:str, value: glm.mat4):
        glUniformMatrix4fv(self._gul(name), 1, GL_FALSE, glm.value_ptr(value))

    def setMat4Array(self, name: str, arr: glm.array):
        glUniformMatrix4fv(self._gul(name), arr.length, GL_FALSE, arr)

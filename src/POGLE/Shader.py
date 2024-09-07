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

        self.shaderPath = f"{cwd}/../assets/shaders/{shaderName}.{_shaderExt[shaderType]}"
        with open(self.shaderPath) as f:
            self.source = f.read()
        glShaderSource(self.ID, self.source,)

        glCompileShader(self.ID)
        if not glGetShaderiv(self.ID, GL_COMPILE_STATUS):
            print(
                f"ERROR::SHADER::{_shaderType[shaderType]}::COMPILATION_FAILED:{_linetab + _linetab.join(glGetShaderInfoLog(self.ID).decode().split(_line))}\n")

    def __del__(self):
        if glIsShader(self.ID):
            try:
                # glDeleteShader expects a single integer, not a list
                glDeleteShader(self.ID)
            except Exception as e:
                print(f"Exception during shader deletion: {e}")
            finally:
                self.ID = 0  # Avoid double deletion and mark as deleted

class VertexShader(Shader):
    def __init__(self, shaderName: str = "default"):
        super().__init__(shaderName, GL_VERTEX_SHADER)


class FragmentShader(Shader):
    def __init__(self, shaderName: str = "default"):
        super().__init__(shaderName, GL_FRAGMENT_SHADER)


defaultUniformBlockLayout = DUBL("Matrices", [
    UBA.Float().Mat4(attrName="uProjection"),  # Projection Matrix
    UBA.Float().Mat4(attrName="uView")  # View Matrix
])



class UniformBlock:
    __create_key = object()
    layout: DUBL
    _UBCache: dict = {}
    _next_block_binding: int = 0

    def __init__(self, create_key, uniformBlockElements, layout: DUBL = defaultUniformBlockLayout):
        super().__init__(uniformBlockElements, layout)
        assert (create_key == UniformBlock.__create_key), \
            "UniformBlock objects must be created using UniformBlock.create"

        self.binding: int = UniformBlock.next_block_binding()
        UniformBlock._UBCache[self.name]: UniformBlock = self

    @classmethod
    def create(cls, uniformBlockElements, layout: DUBL = defaultUniformBlockLayout):
        block: UniformBlock | None = UniformBlock.get_uniform_block(layout.name)
        if block:
            return block
        return UniformBlock(cls.__create_key, uniformBlockElements, layout)

    @staticmethod
    def next_block_binding() -> int:
        next_block_binding = UniformBlock._next_block_binding
        UniformBlock._next_block_binding += 1
        return next_block_binding

    @staticmethod
    def get_uniform_block(block_name: str):
        return UniformBlock._UBCache.get(block_name)

    @property
    def name(self) -> str:
        return self.layout.name


class ShaderProgram:
    def __init__(self, vsName: str = "default", fsName: str = "default"):
        self.vertexShader = VertexShader(vsName)
        self.fragmentShader = FragmentShader(fsName)

        self.ID = glCreateProgram()

        self.uniLocations = {}
        self.uniBlockIndices = {}
        self.boundBlocks = []

        glAttachShader(self.ID, self.vertexShader.ID)
        glAttachShader(self.ID, self.fragmentShader.ID)

        glLinkProgram(self.ID)
        if not glGetProgramiv(self.ID, GL_LINK_STATUS):
            print(
                f"ERROR::SHADER::PROGRAM::LINKING_FAILED:{_linetab + _linetab.join(glGetProgramInfoLog(self.ID).decode().split(_line))}\n")

    def __del__(self):
        if glIsProgram(self.ID):
            try:
                # glDeleteProgram expects a single integer, not a list
                glDeleteProgram(self.ID)
            except Exception as e:
                print(f"Exception during shader program deletion: {e}")
            finally:
                self.ID = 0  # Avoid double deletion and mark as deleted

    def use(self):
        glUseProgram(self.ID)

    def _cache_uniform(self, uniform_name: str):
        self.uniLocations[uniform_name] = glGetUniformLocation(self.ID, uniform_name)
        return self.uniLocations[uniform_name]

    def _gul(self, uniform_name: str): # Get uniform location
        return self.uniLocations.get(uniform_name, self._cache_uniform(uniform_name))

    def _cache_uniform_block(self, block_name: str):
        self.uniBlockIndices[block_name] = glGetUniformBlockIndex(self.ID, block_name)
        return self.uniBlockIndices[block_name]

    def _gubi(self, block_name: str):
        return self.uniBlockIndices.get(block_name, self._cache_uniform_block(block_name))

    def _gub(self, block_name: str) -> UniformBlock | None:
        return UniformBlock.get_uniform_block(block_name)

    def bind_uniform_block(self, block_name: str):
        block: UniformBlock | None = self._gub(block_name)
        if not block:
            return None
        block_index = self._gubi(block_name)
        if GL_INVALID_INDEX == block_index:
            return GL_INVALID_INDEX
        glUniformBlockBinding(self.ID, block_index, block.binding)

    def setMat4(self, name:str, value: glm.mat4):
        glUniformMatrix4fv(self._gul(name), 1, GL_FALSE, glm.value_ptr(value))

    def setMat4Array(self, name: str, arr: glm.array):
        glUniformMatrix4fv(self._gul(name), arr.length, GL_FALSE, arr)

    def setInt(self, name: str, value: GLint):
        glUniform1i(self._gul(name), value)

class UniformBuffer(Buffer):
    def __init__(self):
        super().__init__(GL_UNIFORM_BUFFER)
        self.blockLocations = {}

    def bind_block(self, block_binding: int = 0):
        glBindBufferBase(self.target, block_binding, self.ID)
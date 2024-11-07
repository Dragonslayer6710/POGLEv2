from POGLE.Core.Core import os, proj_dir
from POGLE.Geometry.Data import *

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

        self.shaderPath = f"{proj_dir}\\assets\\shaders\\{shaderName}.{_shaderExt[shaderType]}"
        with open(self.shaderPath) as f:
            self.source = f.read()
        glShaderSource(self.ID, self.source, )

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


class UniformBlockLayout(DataLayout):
    def __init__(self, name: str, attributes: List[VertexAttribute]):
        super().__init__(attributes)
        self.name = name


UBL = UniformBlockLayout
#
# defaultUniformBlockLayout = UBL("ub_Matrices", [
#     VA.Float.Mat4(aName="uProjection"),  # Projection Matrix
#     VA.Float.Mat4(aName="uView")  # View Matrix
# ])


class UniformBlock:
    __create_key = object()
    layout: UBL
    _UBCache: dict = {}
    _next_block_binding: int = 0

    def __init__(self, create_key, layout: UBL):
        self.layout: layout = layout
        self.data: np.ndarray | None = None
        self.bytes: int | None = None
        #self.setData(layout.get_data())
        assert (create_key == UniformBlock.__create_key), \
            "UniformBlock objects must be created using UniformBlock.create"

        self.binding: int = UniformBlock.next_block_binding()
        UniformBlock._UBCache[self.name]: UniformBlock = self

    def setData(self, dataElements):
        self.data = []
        self.bytes = 0
        for i in range(len(dataElements)):
            dataElement = dataElements[i]
            vertAttrib = self.layout.attributes[i]

            self.bytes += vertAttrib.num_bytes
            #dtype = _typeDict[vertAttrib.dtype]
            # Check the type and handle padding if it's a vec2
            if isinstance(dataElement, glm.vec2):
                padded_element = np.array([dataElement.x, dataElement.y, 0.0, 0.0], np.float32)
                self.data = np.concatenate((self.data, padded_element.flatten()), dtype=np.float32)
            elif isinstance(dataElement, glm.mat4):
                # Handle mat4 elements (reshape and transpose as in std140 layout)
                dataElement = np.array(dataElement, np.float32).reshape(4, 4).T
                self.data = np.concatenate((self.data, dataElement.flatten()), dtype=np.float32)
            else:
                # Default case for other types
                dataElement = np.array(dataElement, np.float32)
                self.data = np.concatenate((self.data, dataElement.flatten()), dtype=np.float32)

    @classmethod
    def create(cls, layout: UBL):
        block: UniformBlock | None = UniformBlock.get_uniform_block(layout.name)
        if block:
            return block
        return UniformBlock(cls.__create_key, layout)

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

    def _gul(self, uniform_name: str):  # Get uniform location
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

    def setMat4(self, name: str, value: glm.mat4):
        glUniformMatrix4fv(self._gul(name), 1, GL_FALSE, glm.value_ptr(value))

    def setMat4Array(self, name: str, arr: glm.array):
        glUniformMatrix4fv(self._gul(name), arr.length, GL_FALSE, arr)

    def setInt(self, name: str, value: GLint):
        glUniform1i(self._gul(name), value)

    def setTexture(self, name: str, texture_index: int):
        self.setInt(name, texture_index)


class UniformBuffer(Buffer):
    def __init__(self):
        super().__init__(GL_UNIFORM_BUFFER, GL_STATIC_DRAW)
        self.blockLocations = {}

    def bind_block(self, block_binding: int = 0):
        glBindBufferBase(self.target, block_binding, self.ID)

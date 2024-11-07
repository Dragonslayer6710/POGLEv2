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
        self.data: Optional[bytes] = None
        self.bytes: Optional[int] = None
        self.set_data(layout.attributes)
        assert (create_key == UniformBlock.__create_key), \
            "UniformBlock objects must be created using UniformBlock.create"

        self.binding: int = UniformBlock.next_block_binding()
        UniformBlock._UBCache[self.name]: UniformBlock = self

    def _pad_data(self, data: Collection) -> np.ndarray:
        if isinstance(data[0], Collection):  # If this is a 3D array
            raise TypeError("UniformBlock accepts a maximum depth of 2")
        if len(data) < 4:  # Pad with 0s if necessary
            data = *data , *((0,) * (4 - len(data)))
        return np.array(data, dtype=type(data[0]))

    def _process_2d_array(self, data: Collection) -> np.ndarray:
        temp_data = []
        for sub_data in data:
            if len(sub_data) < 4:  # Pad with 0s if necessary
                print()
            sub_data = self._pad_data(sub_data)
            temp_data = np.concatenate((temp_data, sub_data), dtype=sub_data.dtype)
        return temp_data

    def _process_array(self, data: Collection) -> np.ndarray:
        if isinstance(data[0], np.ndarray):  # If this is a 2D array
            return self._process_2d_array(data)
        else:  # If this is a 1D array
            return self._pad_data(data)

    def set_data(self, data: Union[List[Any], bytes]):
        self.data = []
        if isinstance(data, bytes):  # Direct bytes
            self.data = data
            return
        elif isinstance(data, List):  # List of Data Points
            for data_point in data:
                if isinstance(data_point, VertexAttribute):
                    # Data Point is a Vertex Attribute
                    # Get the numpy array from the Data Point
                    np_data: np.ndarray = data_point.data
                    data_point = np.concatenate([
                        self._process_array(np_data[i]) for i in range(np_data.shape[0])
                    ]) if len(np_data.shape) == 3 else self._process_array(np_data)
                elif isinstance(data_point, Collection):
                    # Data Points is a Collection of values (an array of ints/floats/vecs/mats)
                    if isinstance(data_point, np.ndarray):
                        data_point = self._process_array(np_data)  # Process the array
                    elif isinstance(data_point, (glm.vec2, glm.vec3, glm.vec4, glm.mat2, glm.mat3, glm.mat4)):
                        if isinstance(data_point, (glm.mat2, glm.mat3, glm.mat4)):
                            data_point = np.array(data_point, np.float32).T
                        data_point = self._process_array(data_point)
                    elif isinstance(data_point, (glm.ivec2, glm.ivec3, glm.ivec4, glm.imat2, glm.imat3, glm.imat4)):
                        if isinstance(data_point, (glm.imat2, glm.imat3, glm.imat4)):
                            data_point = np.array(data_point, np.int32).T
                        data_point = self._process_array(data_point)
                    elif isinstance(data_point, (int, np.int32, float, np.float32)):
                        data_point = self._pad_data(data_point)
                    else:
                        raise TypeError(f"Invalid data type: {type(data_point)}")
                self.data = np.concatenate((self.data, data_point.flatten()), dtype=data_point.dtype)
            self.bytes = self.data.nbytes
            self.data = self.data.tobytes()
            return
        raise TypeError(f"Invalid data type: {type(data)}")

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

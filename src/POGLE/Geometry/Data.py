from POGLE.OGL.OpenGLContext import *
import glfw
import numpy as np
import ctypes
from ctypes import c_float, c_double, c_uint, c_short, c_ushort
import glm
from dataclasses import dataclass
from typing import List, Collection, Union, Any, Tuple, Dict, Optional


def NMM(t: glm.vec3, r: glm.vec3 = glm.vec3(), s: glm.vec3 = glm.vec3(1), glr: bool = False) -> Union[glm.mat4, np.ndarray]:
    return NewModelMatrix(t, r, s, glr)


def NewModelMatrix(translation: glm.vec3 = glm.vec3(),
                   rotation: glm.vec3 = glm.vec3(),
                   scale: glm.vec3 = glm.vec3(1.0),
                   gl_ready = False) -> Union[glm.mat4, np.ndarray]:
    # Create the rotation matrix using Euler angles
    rotation_matrix = glm.mat4_cast(glm.quat(glm.vec3(glm.radians(rotation))))

    # Create the scale and translation matrices
    scale_matrix = glm.scale(glm.mat4(), scale)
    translation_matrix = glm.translate(glm.mat4(), translation)

    # Combine the matrices: translation * (rotation * scale)
    model_matrix = translation_matrix * rotation_matrix * scale_matrix
    if gl_ready:
        return np.array(model_matrix.to_list())
    return model_matrix


class Buffer:
    _bound_buffers: dict = {}

    def __init__(self, target, usage, dtype: np.dtype = np.float32):
        self.target = target
        self.usage = usage
        self.ID = glGenBuffers(1)
        if self.ID == 0:
            raise RuntimeError("Failed to generate buffer ID.")
        self.size = 0
        self._cleaned = False  # Flag to track if cleanup has been done
        if dtype is not None:
            self.dtype = dtype

    @staticmethod
    def getBoundBuffer(target):
        return Buffer._bound_buffers.get(target) or 0

    @staticmethod
    def isBufferBound(target, bufferID):
        return Buffer.getBoundBuffer(target) == bufferID

    @staticmethod
    def recBoundBuffer(target, bufferID):
        Buffer._bound_buffers[target] = bufferID

    @property
    def bound(self):
        return Buffer.isBufferBound(self.target, self.ID)

    def recBind(self, clear=False):
        bufferID = 0 if clear else self.ID
        Buffer.recBoundBuffer(self.target, bufferID)

    def bind(self):
        if self._cleaned:
            raise RuntimeError("Cannot bind a cleaned up buffer.")
        if not self.bound:
            glBindBuffer(self.target, self.ID)
            self.recBind()

    def unbind(self):
        if self._cleaned:
            raise RuntimeError("Cannot unbind a cleaned up buffer.")
        if self.bound:
            glBindBuffer(self.target, 0)
            self.recBind(True)

    def buffer_data(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            raise TypeError("Buffer can only receive an np array or bytes object when buffering data")
        if self._cleaned:
            raise RuntimeError("Cannot buffer data to a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot buffer data to an unbound buffer")
        self.size = len(data)
        glBufferData(self.target, self.size, data, self.usage)

    def buffer_sub_data(self, offset, data: bytes):
        if self._cleaned:
            raise RuntimeError("Cannot buffer sub-data to a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot buffer sub-data to an unbound buffer")
        glBufferSubData(self.target, offset, len(data), data)

    def allocate(self, size):
        if self._cleaned:
            raise RuntimeError("Cannot allocate a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot allocate an unbound buffer")
        self.size = size
        glBufferData(self.target, self.size, None, self.usage)

    def get_data(self, offset=0, size=None):
        current_buffer = self._bound_buffers[self.target]
        if current_buffer != self.ID:
            self.bind()
        if self._cleaned:
            raise RuntimeError("Cannot read a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot read an unbound buffer")

        # Get buffer size if not specified
        if size is None:
            buffer_size = glGetBufferParameteriv(self.target, GL_BUFFER_SIZE)
        else:
            buffer_size = size

        # Read the data from the buffer
        data = glGetBufferSubData(self.target, offset, buffer_size)
        if current_buffer != self.ID:
            self.unbind()
            glBindBuffer(self.target, current_buffer)

        # Convert to numpy array for easier inspection
        # Assuming the data is in float format for simplicity
        data_array = np.frombuffer(data, dtype=self.dtype)
        return data_array, buffer_size

    def print_data(self, offset=0, size=None):
        data_array, buffer_size = self.get_data()
        print(f"Buffer Data (offset: {offset}, size: {buffer_size}):")
        print(data_array)

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        if glIsBuffer(self.ID):
            if not self._cleaned and self.ID and self.ID != 0:
                try:
                    glDeleteBuffers(1, [self.ID])
                except TypeError as e:
                    print(f"TypeError during buffer deletion: {e}")
                except OpenGL.GL.error.GLError as e:
                    print(f"OpenGL error during buffer deletion: {e}")
                except Exception as e:
                    print(f"Exception during buffer deletion: {e}")
                finally:
                    self.ID = 0  # Mark as deleted
                    self._cleaned = True  # Set the flag to indicate cleanup is done
            else:
                if not self._cleaned:
                    print(f"Buffer ID is invalid or already deleted: {self.ID}")
        else:
            print(f"Buffer ID {self.ID} does not exist.")


class VertexBuffer(Buffer):
    def __init__(self, usage, dtype=np.float32):
        super().__init__(GL_ARRAY_BUFFER, usage, dtype)


class ElementBuffer(Buffer):
    dtype = GL_UNSIGNED_SHORT

    def __init__(self, usage):
        super().__init__(GL_ELEMENT_ARRAY_BUFFER, usage, np.ushort)


@dataclass
class VertexAttribute:
    name: str
    data: Union[Any, Collection[Any], List[Collection[Any]], np.ndarray]
    normalized: bool = False
    divisor: GLsizei = 0

    def __post_init__(self):
        if isinstance(self.data, np.ndarray):
            self.data = list(self.data)
        if not isinstance(self.data, List):
            self.data = [self.data]

        self._component_count = 1
        self._elem_count = 1
        data_point: Union[Any, Collection[Any]] = self.data[0]
        if isinstance(data_point, Collection):  # Indicates Vector or Matrix
            self._elem_count = len(data_point)
            data_point = data_point[0]
            if isinstance(data_point, Collection):  # Indicates this is a Matrix
                self._component_count, self._elem_count = self._elem_count, len(data_point)
                data_point = data_point[0]
                if isinstance(data_point, Collection):
                    raise TypeError("Maximum Vertex Attribute Depth is 2 for Matrix (WxH)")
        self.num_elements = self._component_count * self._elem_count

        self.gl_type: GLenum = None
        self.dtype: np.dtype = None
        elem_size: int = 4  # int32 and float32 are 4 bytes in size
        if isinstance(data_point, (int, np.int32)):
            self.gl_type = GL_INT
            self.dtype = np.int32
        elif isinstance(data_point, (float, np.float32)):
            self.gl_type = GL_FLOAT
            self.dtype = np.float32
        else:
            raise TypeError("Only Base Data Types of int/np.int32 or float/np.float32 are allowed")
        self.data = np.array(self.data, dtype=self.dtype)
        self.base_size: int = elem_size * self.num_elements
        self._component_size = self.base_size // self._component_count

    @property
    def size(self):
        return len(self.data)

    def set_attribute_pointer(self, start_index: GLsizei, stride: GLsizei, pointer: GLsizei,
                              print_attribute=False) -> int:
        for i in range(self._component_count):
            if stride == 0 and self._component_count > 1:
                stride = self.base_size
            attr_index = start_index + i
            glEnableVertexAttribArray(attr_index)

            component_pointer = pointer + self._component_size * i
            if self.gl_type in (GL_INT, GL_UNSIGNED_INT, GL_SHORT, GL_UNSIGNED_SHORT):
                glVertexAttribIPointer(attr_index, self._elem_count, self.gl_type, stride,
                                       ctypes.c_void_p(component_pointer))
            else:
                glVertexAttribPointer(attr_index, self._elem_count, self.gl_type, self.normalized, stride,
                                      ctypes.c_void_p(component_pointer))
            glVertexAttribDivisor(attr_index, self.divisor)
            if print_attribute:
                print(
                    f"Pointer Set: {self.name}{('['+str(i)+']') if self._component_count > 1 else ''}:\t"
                    f"{{index: {attr_index} | elements: {self._elem_count} | bytes: {self._component_size} |"
                    f" dtype: {self.gl_type} | normalized: {self.normalized} | stride: {stride} |"
                    f" offset: {component_pointer} | divisor: {self.divisor}}}")
        return attr_index + 1


@dataclass
class DataLayout:
    attributes: List[VertexAttribute]

    def __post_init__(self):
        self._divisors: List[int] = []
        self._strides: Dict[int, int] = {}
        self._attributes: Dict[int, List[VertexAttribute]] = {}
        self._dtypes: Dict[int, List] = {}
        for attribute in self.attributes:
            divisor = attribute.divisor
            if divisor not in self._divisors:
                self._divisors.append(divisor)
                self._strides[divisor] = 0
                self._attributes[divisor] = []
                self._dtypes[divisor] = []
            else:
                if self._strides[divisor] == 0:
                    self._strides[divisor] += self._attributes[divisor][-1].base_size
                self._strides[divisor] += attribute.base_size
            self._attributes[divisor].append(attribute)

            self._dtypes[divisor].append((
                attribute.name,
                attribute.dtype,
                attribute._elem_count
                if not attribute._component_count -1 else (
                    attribute._component_count,
                    attribute._elem_count
                )
            ))


    def set_pointers(self, print_attributes=False):
        index: int = 0
        division_pointer: int = 0 # Points to the start of each division
        for i, divisor in enumerate(self._divisors):
            attribute_pointer: int = 0 # Points to the start of each attribute irrespective of divisor
            for attribute in self._attributes[divisor]:
                pointer: int = division_pointer + attribute_pointer
                index = attribute.set_attribute_pointer(
                    index, self._strides[divisor], pointer, print_attributes
                )
                attribute_pointer += attribute.base_size
            # Multiple attribute pointer by number of "instances" of the attributes to get the next division pointer
            division_pointer += attribute_pointer * self._attributes[divisor][0].size

    def get_data(self) -> bytes:
        data: Optional[bytes] = None
        offset: int = 0
        for divisor in self._divisors:
            vertex_dtype = np.dtype(self._dtypes[divisor])
            vertex_data = np.zeros(self._attributes[divisor][0].size, dtype=vertex_dtype)
            for attribute in self._attributes[divisor]:
                vertex_data[attribute.name] = attribute.data
            if data is None:
                data = vertex_data.tobytes()
            else:
                data += vertex_data.tobytes()
            offset += vertex_data.nbytes
        return data


class BufferManager:
    def __init__(self):
        self.buffers: dict[int, Buffer] = {}

    def _register_buffer(self, buffer):
        self.buffers[buffer.ID] = buffer

    def create_vbo(self, usage) -> VertexBuffer:
        buffer = VertexBuffer(usage)
        self._register_buffer(buffer)
        return buffer

    def create_ebo(self, usage) -> ElementBuffer:
        buffer = ElementBuffer(usage)
        self._register_buffer(buffer)
        return buffer

    def delete_buffer(self, buffer_id):
        self.buffers.pop(buffer_id, None)

    def _get_buffer(self, buffer_id, buffer_type=Buffer):
        buffer = self.buffers.get(buffer_id)
        if not isinstance(buffer, buffer_type):
            raise TypeError("Buffer Requested is not of Requested Type")
        return buffer

    def get_vbo(self, buffer_id) -> VertexBuffer:
        return self._get_buffer(buffer_id, VertexBuffer)

    def get_ebo(self, buffer_id) -> ElementBuffer:
        return self._get_buffer(buffer_id, ElementBuffer)


class VertexArray:
    def __init__(self):
        self.ID = glGenVertexArrays(1)  # Ensure you get the ID correctly
        self.ebo = None
        self.buffer_manager = BufferManager()
        self.attribute_index = 0

    def bind(self):
        glBindVertexArray(self.ID)
        if self.ebo:
            self.ebo.bind()

    def unbind(self):
        glBindVertexArray(0)
        if self.ebo:
            self.ebo.unbind()

    def add_vbo(self, layout: DataLayout, usage=GL_STATIC_DRAW, print_buffer=True) -> VertexBuffer:
        # Create and bind the vertex buffer object
        buffer: VertexBuffer = self.buffer_manager.create_vbo(usage)

        # Bind the VAO
        self.bind()

        # Bind the VBO
        buffer.bind()
        data = layout.get_data()
        # buffer.allocate(np.sum([attr.num_bytes for attr in layout.attributes]))
        #[buffer.buffer_sub_data(*data[1:]) for data in None]
        buffer.buffer_data(data)
        try:
            # Set vertex attribute pointers
            layout.set_pointers(print_buffer)
        finally:
            # Unbind the VBO
            if print_buffer:
                buffer.print_data()
            buffer.unbind()

        # Unbind the VAO
        self.unbind()

        return buffer

    def set_ebo(self, usage=GL_STATIC_DRAW, data: np.ndarray = None, print_buffer=True) -> ElementBuffer:
        ebo: ElementBuffer = self.buffer_manager.create_ebo(usage)
        ebo.bind()
        ebo.buffer_data(data.tobytes())
        try:
            self.ebo = ebo
        finally:
            if print_buffer:
                ebo.print_data()
            ebo.unbind()
        return ebo


    def _cleanup(self):
        if self.ID and self.ID != 0:  # Check if ID is not None and valid
            try:
                glDeleteVertexArrays(1, [self.ID])
                self.ID = 0  # Mark as deleted
            except OpenGL.GL.error.GLError as e:
                print(f"OpenGL error during VAO deletion: {e}")
            except Exception as e:
                print(f"Exception during VAO deletion: {e}")
            finally:
                self.ID = 0

    def __del__(self):
        self._cleanup()
        # The buffer_manager's buffers should be deleted by their own __del__ methods
        self.buffer_manager = None

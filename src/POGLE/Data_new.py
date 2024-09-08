from OpenGL.GL import *
import glfw
import numpy as np
import ctypes
from ctypes import c_float, c_double, c_uint, c_short, c_ushort


class Buffer:
    _bound_buffers: dict = {}

    def __init__(self, target, usage):
        self.target = target
        self.usage = usage
        self.ID = glGenBuffers(1)
        if self.ID == 0:
            raise RuntimeError("Failed to generate buffer ID.")
        self.size = 0
        self._cleaned = False  # Flag to track if cleanup has been done

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

    def buffer_data(self, data: bytes):
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

    def read_data(self, offset=0, size=None):
        if self._cleaned:
            raise RuntimeError("Cannot read a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot read an unbound buffer")

        # Get buffer size if not specified
        if size is None:
            buffer_size = glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)
        else:
            buffer_size = size

        # Read the data from the buffer
        data = glGetBufferSubData(GL_ARRAY_BUFFER, offset, buffer_size)
        self.unbind()

        # Convert to numpy array for easier inspection
        # Assuming the data is in float format for simplicity
        data_array = np.frombuffer(data, dtype=np.float32)

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
    def __init__(self, usage):
        super().__init__(GL_ARRAY_BUFFER, usage)


class ElementBuffer(Buffer):
    def __init__(self, usage):
        super().__init__(GL_ELEMENT_ARRAY_BUFFER, usage)


class VertexAttribute:
    _type_bytes = {
        GL_FLOAT: ctypes.sizeof(c_float),
        GL_DOUBLE: ctypes.sizeof(c_double),
        GL_INT: ctypes.sizeof(c_int),
        GL_UNSIGNED_INT: ctypes.sizeof(c_uint),
        GL_SHORT: ctypes.sizeof(c_short),
        GL_UNSIGNED_SHORT: ctypes.sizeof(c_ushort),
    }

    class _Base:
        __create_key = object()

        gl_type = None

        def __init__(self, create_key, aName: str, size: GLsizei, normalized=False, divisor: GLsizei = 0):
            if create_key != self.__class__.__create_key:
                raise ValueError("VertexAttribute objects must be created using VertexAttribute._new")
            self._num_bytes: GLsizei = None
            self.aName = aName
            self.num_elements = size
            self.normalized = normalized
            self.divisor = divisor

            self._num_components = 4 if self.num_elements > 4 else self.num_elements
            self._num_attributes = (self.num_elements + self._num_components - 1) // self._num_components

        @property
        def num_bytes(self):
            if self._num_bytes is None:
                self._num_bytes = VertexAttribute._type_bytes.get(self.gl_type)
                if self._num_bytes is None:
                    raise ValueError(f"Type bytes for OpenGL type {self.gl_type} is not defined.")

                if self.num_elements - 1:
                    # For matrices, calculate the total bytes based on matrix dimensions
                    self._num_bytes *= self.num_elements
            return self._num_bytes

        def set_attribute_pointer(self, index: GLsizei, stride: GLsizei, offset: GLsizei, print_attribute=False):
            for i in range(self._num_attributes):
                attr_index = index + i
                glEnableVertexAttribArray(attr_index)

                component_size = min(self._num_components, self.num_elements - i * self._num_components)
                component_bytes = (self.num_bytes // self.num_elements) * component_size
                component_offset = offset + component_bytes * i

                if self.gl_type in (GL_INT, GL_UNSIGNED_INT, GL_SHORT, GL_UNSIGNED_SHORT):
                    glVertexAttribIPointer(attr_index, component_size, self.gl_type, stride,
                                           ctypes.c_void_p(component_offset))
                else:
                    glVertexAttribPointer(attr_index, component_size, self.gl_type, self.normalized, stride,
                                          ctypes.c_void_p(component_offset))
                glVertexAttribDivisor(attr_index, self.divisor)
                if print_attribute:
                    print(
                        f"Pointer Set:\t{{index: {attr_index} | elements: {component_size} | bytes: {component_bytes} | dtype: {self.gl_type} | normalized: {self.normalized} | stride: {stride} | offset: {component_offset} | divisor: {self.divisor}}}")
            return self._num_attributes

        def __repr__(self):
            # Determine if the attribute represents a matrix and construct the appropriate description
            if self._num_attributes > 1:
                mat_info = f"matrix ({self._num_attributes}x{self._num_components})"
            else:
                mat_info = f"vector ({self.num_elements})" if self.num_elements > 1 else "scalar"

            # Represent the normalized and divisor properties clearly
            normalized_info = "normalized" if self.normalized else "not normalized"
            divisor_info = f"divisor={self.divisor}" if self.divisor else "no divisor"

            return (f"VertexAttribute(name={self.aName}, type={self.gl_type}, "
                    f"{mat_info}, {normalized_info}, {divisor_info})")

        @classmethod
        def _new(cls, aName: str, size: GLsizei, normalized=False, divisor: GLsizei = 0):
            return cls(cls.__create_key, aName, size, normalized, divisor)

        @classmethod
        def Scalar(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._new(aName, 1, normalized, divisor)

        @classmethod
        def Array(cls, aName: str, arr_length: GLsizei, normalized=False, divisor: GLsizei = 0):
            return cls._new(aName, arr_length, normalized, divisor)

        @classmethod
        def _Vec(cls, aName: str, size: GLsizei, normalized=False, divisor: GLsizei = 0):
            return cls.Array(aName, size, normalized, divisor)

        @classmethod
        def Vec2(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Vec(aName, 2, normalized, divisor)

        @classmethod
        def Vec3(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Vec(aName, 3, normalized, divisor)

        @classmethod
        def Vec4(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Vec(aName, 4, normalized, divisor)

        @classmethod
        def _Mat(cls, aName: str, rows: GLsizei, cols: GLsizei, normalized=False, divisor: GLsizei = 0):
            return cls.Array(aName, rows * cols, normalized, divisor)

        @classmethod
        def _Mat2R(cls, aName: str, cols: GLsizei, normalized=False, divisor: GLsizei = 0):
            return cls._Mat(aName, 2, cols, normalized, divisor)

        @classmethod
        def Mat2(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat2R(aName, 2, normalized, divisor)

        @classmethod
        def Mat2x3(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat2R(aName, 3, normalized, divisor)

        @classmethod
        def Mat2x4(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat2R(aName, 4, normalized, divisor)

        @classmethod
        def _Mat3R(cls, aName: str, cols: GLsizei, normalized=False, divisor: GLsizei = 0):
            return cls._Mat(aName, 3, cols, normalized, divisor)

        @classmethod
        def Mat3x2(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat3R(aName, 2, normalized, divisor)

        @classmethod
        def Mat3(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat3R(aName, 3, normalized, divisor)

        @classmethod
        def Mat3x4(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat3R(aName, 4, normalized, divisor)

        @classmethod
        def _Mat4R(cls, aName: str, cols: GLsizei, normalized=False, divisor: GLsizei = 0):
            return cls._Mat(aName, 4, cols, normalized, divisor)

        @classmethod
        def Mat4x2(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat4R(aName, 2, normalized, divisor)

        @classmethod
        def Mat4x3(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat4R(aName, 3, normalized, divisor)

        @classmethod
        def Mat4(cls, aName: str, normalized=False, divisor: GLsizei = 0):
            return cls._Mat4R(aName, 4, normalized, divisor)

    class Float(_Base):
        gl_type = GL_FLOAT

    class Int(_Base):
        gl_type = GL_INT

    class UInt(_Base):
        gl_type = GL_UNSIGNED_INT

    class Short(_Base):
        gl_type = GL_SHORT

    class UShort(_Base):
        gl_type = GL_UNSIGNED_SHORT

VA = VertexAttribute

class BufferLayout:
    def __init__(self, attributes):
        self.attributes: list[VertexAttribute] = attributes
        self.stride = sum(attr.num_bytes for attr in attributes)

    def get_offset(self, index):
        offset = 0
        for i, attr in enumerate(self.attributes):
            if i == index:
                return offset
            offset += attr.num_bytes
        return offset


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

    def add_vbo(self, layout, usage=GL_STATIC_DRAW, data=None) -> VertexBuffer:
        # Create and bind the vertex buffer object
        buffer: VertexBuffer = self.buffer_manager.create_vbo(usage)

        # Bind the VAO
        self.bind()

        # Bind the VBO
        buffer.bind()

        # Upload data to the buffer if provided
        if data is not None:
            buffer.buffer_data(data)

        try:
            # Set vertex attribute pointers
            self._set_vertex_attributes(layout)
        finally:
            # Unbind the VBO
            buffer.unbind()

        # Unbind the VAO
        self.unbind()

        return buffer

    def set_ebo(self, usage=GL_STATIC_DRAW, data=None) -> ElementBuffer:
        ebo: ElementBuffer = self.buffer_manager.create_ebo(usage)
        ebo.bind()
        ebo.buffer_data(data)
        try:
            self.ebo = ebo
        finally:
            ebo.unbind()
        return ebo

    def _set_vertex_attributes(self, layout, print_attributes=False):
        stride = layout.stride
        for i, attr in enumerate(layout.attributes):
            index = self.attribute_index + i
            offset = layout.get_offset(i)
            self.attribute_index += attr.set_attribute_pointer(index, stride, offset, print_attributes)

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


def create_shader_program():
    vertex_shader_code = """
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 color;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    out vec3 fragColor;
    void main()
    {
        gl_Position = projection * view * model * vec4(position, 1.0);
        fragColor = color;
    }
    """

    fragment_shader_code = """
    #version 330 core
    in vec3 fragColor;
    out vec4 color;
    void main()
    {
        color = vec4(fragColor, 1.0);
    }
    """

    def compile_shader(source_code, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source_code)
        glCompileShader(shader)
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(f"Shader compilation failed: {glGetShaderInfoLog(shader).decode()}")
        return shader

    def link_program(vertex_shader, fragment_shader):
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(f"Program linking failed: {glGetProgramInfoLog(program).decode()}")
        return program

    vertex_shader = compile_shader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_shader_code, GL_FRAGMENT_SHADER)
    shader_program = link_program(vertex_shader, fragment_shader)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program


def setup_cube():
    # Define vertices and indices for a cube
    vertices = np.array([
        # Positions         # Colors
        -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
        0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
        0.5, 0.5, -0.5, 0.0, 0.0, 1.0,
        -0.5, 0.5, -0.5, 1.0, 1.0, 0.0,
        -0.5, -0.5, 0.5, 1.0, 0.0, 1.0,
        0.5, -0.5, 0.5, 0.0, 1.0, 1.0,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        -0.5, 0.5, 0.5, 1.0, 1.0, 1.0
    ], dtype=np.float32)

    indices = np.array([
        0, 1, 2, 2, 3, 0,  # Front face
        4, 5, 6, 6, 7, 4,  # Back face
        0, 1, 5, 5, 4, 0,  # Bottom face
        2, 3, 7, 7, 6, 2,  # Top face
        0, 3, 7, 7, 5, 1,  # Left face
        1, 2, 6, 6, 5, 1  # Right face
    ], dtype=np.uint32)

    # Create Vertex Array Object
    vao = VertexArray()

    # Define vertex buffer and layout
    vbo = vao.add_vbo(BufferLayout([
        VA.Float.Vec3('position'),
        VA.Float.Vec3('color')
    ]), data=vertices.tobytes())

    # Set up element buffer (indices)
    ebo = vao.set_ebo(data=indices.tobytes())

    return vao, len(indices)


def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Test Buffer Management", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    shader_program = create_shader_program()

    if not shader_program:
        print("Failed to create shader program.")
        glfw.terminate()
        return

    vao, index_count = setup_cube()

    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Compute rotation
        time = glfw.get_time()
        model = np.array([
            np.cos(time), np.sin(time), 0.0, 0.0,
            -np.sin(time), np.cos(time), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ], dtype=np.float32)

        view = np.array([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, -5.0,
            0.0, 0.0, 0.0, 1.0
        ], dtype=np.float32)

        projection = np.array([
            1.0 / (800.0 / 600.0), 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, -2.0 / (1000.0 - 0.1), -1.0,
            0.0, 0.0, -1.0, 0.0
        ], dtype=np.float32)

        glUseProgram(shader_program)

        # Get uniform locations
        model_loc = glGetUniformLocation(shader_program, "model")
        view_loc = glGetUniformLocation(shader_program, "view")
        projection_loc = glGetUniformLocation(shader_program, "projection")

        if model_loc == -1 or view_loc == -1 or projection_loc == -1:
            print("One or more uniform locations are invalid.")
            glfw.terminate()
            return

        # Set uniforms
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

        vao.bind()
        glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, None)
        vao.unbind()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()

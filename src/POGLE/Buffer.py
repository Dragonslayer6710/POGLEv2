from POGLE.Geometry.Data import *
class Buffer:
    def __init__(self, buffers = GL_ARRAY_BUFFER, usage = GL_STATIC_DRAW, dtype = GLfloat):
        self.target: GLenum = buffers
        self.ID: GLuint = glGenBuffers(1)
        self.usage = usage
        self.dtype = dtype
        self.stored_size: int = 0
        self.data_len: int = 0

    def bind(self):
        glBindBuffer(self.target, self.ID)

    def unbind(self):
        glBindBuffer(self.target, 0)

    def buffer_data(self, size: GLsizeiptr, data: np.ndarray):
        self.data_len = len(data)
        self.stored_size = size
        glBufferData(self.target, size, data, self.usage)

    def get_data(self):
        # Feedback is for debugging, may be removed or disabled in production
        feedback = (self.dtype * self.data_len)()
        glGetBufferSubData(self.target, 0, self.stored_size, feedback)
        return feedback

    def print_data(self):
        print(f"Buffering {self.stored_size} bytes of data: {list(self.get_data())}")  # Debugging info


    def __del__(self):
        if glIsBuffer(self.ID):
            try:
                # glDeleteBuffers expects a list or tuple
                glDeleteBuffers(1, [self.ID])
            except Exception as e:
                print(f"Exception during buffer deletion: {e}")
            finally:
                self.ID = 0  # Avoid double deletion and mark as deleted


class VertexBuffer(Buffer):
    def __init__(self):
        super().__init__()

    def buffer_data(self, vertices: Vertices, instances: Instances = None):
        if not instances:
            super().buffer_data(vertices.bytes, vertices.data)
        else:
            super().buffer_data(vertices.bytes + instances.bytes, np.concatenate([vertices.data, instances.data], axis=0))

class ElementBuffer(Buffer):
    def __init__(self):
        super().__init__(GL_ELEMENT_ARRAY_BUFFER, dtype=GLushort)

class VertexArray:
    def __init__(self, vertices: Vertices, indices: list[int], instances: Instances = None):
        self.EBO = None

        indices = np.array(indices, np.uint16)

        self.ID = glGenVertexArrays(1)
        self.bind()

        self.VBO = VertexBuffer()
        self.VBO.bind()
        self.VBO.buffer_data(vertices, instances)
        vertices.set_vertex_attrib_pointers()
        if instances:
            instances.set_vertex_attrib_pointers(vertices.nextID(), vertices.bytes)
        self.VBO.unbind()

        self.EBO = ElementBuffer()
        self.EBO.bind()
        self.EBO.buffer_data(indices.nbytes, indices)

        self.unbind()


    def __del__(self):
        if self.ID != 0:
            try:
                glDeleteVertexArrays(1, [self.ID])
            except Exception as e:
                print(f"Exception during VAO deletion: {e}")
            finally:
                self.ID = 0  # Avoid double deletion

    def bind(self):
        glBindVertexArray(self.ID)
        if self.EBO:
            self.EBO.bind()

    def unbind(self):
        glBindVertexArray(0)
        self.EBO.unbind()
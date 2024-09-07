import numpy as np

from POGLE.Geometry.Data import *
_typeDict = Data._typeDict

_defaultVBOLayout = DVL(VA.Float().Single())
_EBOLayout = DVL(VA.UShort().Single())

class Buffer:
    def __init__(self, layout: Data._Layout, target = GL_ARRAY_BUFFER, usage = GL_STATIC_DRAW):
        self.target: GLenum = target
        self.ID: GLuint = glGenBuffers(1)
        self.layout = layout
        self.usage = usage
        self.stored_size: int = 0
        self.data_len: int = 0

    def bind(self):
        glBindBuffer(self.target, self.ID)

    def unbind(self):
        glBindBuffer(self.target, 0)

    def buffer_data(self, size: GLsizeiptr, data: bytes):
        self.data_len = len(data)
        self.stored_size = size
        glBufferData(self.target, size, data, self.usage)

    def get_data(self):
        feedback = []
        # Feedback is for debugging, may be removed or disabled in production
        offset = 0
        for attribute in self.layout.attributes:
            bytes = attribute.numBytes
            proportion = bytes // self.layout.stride
            sub_feedback = (_typeDict[attribute.ctype_base] * self.data_len * proportion)()
            glGetBufferSubData(self.target, offset, bytes, sub_feedback)
            feedback += sub_feedback
            offset += bytes
        return feedback

    def print_data(self):
        print(f"Buffering {self.stored_size} bytes of data:\n{list(self.get_data())}")  # Debugging info


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
    def __init__(self, layout: Data._Layout = _defaultVBOLayout):
        super().__init__(layout)

    def buffer_data(self, vertices: Vector):
        super().buffer_data(vertices.bytes, vertices.data)

class ElementBuffer(Buffer):
    def __init__(self):
        super().__init__(_EBOLayout, GL_ELEMENT_ARRAY_BUFFER)

class VertexArray:
    def __init__(self, vertices: Vector, indices: list[int], instances: Optional[Union[List[Vector], Vector]] = None):
        self.EBO = None

        indices = np.array(indices, np.ushort)

        self.ID = glGenVertexArrays(1)
        self.bind()

        self.IBOs: Optional[List[VertexBuffer]] = None
        if isinstance(instances, List):
            self.IBOs = []
            for i, instDataSet in enumerate(instances):
                instDataSet.set_attribute_array_pointers()
                self.IBOs.append(VertexBuffer())
                self.IBOs[i].bind()
                self.IBOs[i].buffer_data(instDataSet)
                self.IBOs[i].unbind()

        self.VBO = VertexBuffer()
        self.VBO.bind()
        vertices.set_attribute_array_pointers()
        if not instances or isinstance(instances, List):
            self.VBO.buffer_data(vertices)
        else:
            self.VBO.buffer_data(vertices, instances)
            instances.set_attribute_array_pointers(vertices.bytes)
        self.VBO.unbind()

        self.EBO = ElementBuffer()
        self.EBO.bind()
        self.EBO.buffer_data(indices.nbytes, indices.tobytes())

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
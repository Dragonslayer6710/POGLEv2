import numpy as np

from MineClone.Block import BlockShape
from POGLE.Shader import *
from POGLE.Geometry.Texture import *
from POGLE.Geometry.Shape import *

from POGLE.Geometry.Shape import _BaseShape


class Mesh:
    def __init__(self, vertex_data: List[np.ndarray], vertex_buffer_layouts: List[DataLayout],
                 indices: np.ndarray, shader: ShaderProgram, primitive: GLenum = GL_TRIANGLES,
                 textures: Optional[Dict[str, Texture]] = None,
                 instance_data: Optional[List[np.ndarray]] = None,
                 instance_buffer_layout: Optional[List[DataLayout]] = None):
        self.vao: VertexArray = VertexArray()
        self.vbos: List[VertexBuffer] = self._set_vertex_buffers(vertex_data, vertex_buffer_layouts)
        self.num_instances: int = 1

        self.vao.set_ebo(data=indices)
        self.num_indices = len(indices)
        self.shader = shader
        self.primitive = primitive

        self.textures: Dict[str, Texture] = textures if textures is not None else {}

        self.instance_vbos: List[VertexBuffer] = []
        if instance_data is not None:
            self.set_instance_buffers(instance_data, instance_buffer_layout)
        self.shader.use()
        self.bind_textures()

    def _set_vertex_buffers(self, vertex_data: List[np.ndarray], vertex_buffer_layouts: List[DataLayout]):
        if not isinstance(vertex_data, List):
            vertex_data = [vertex_data]
        if not isinstance(vertex_buffer_layouts, List):
            vertex_buffer_layouts = [vertex_buffer_layouts]
        return [self.vao.add_vbo(layout, data=data) for data, layout
                in zip(vertex_data, vertex_buffer_layouts)]

    def add_texture(self, name: str, texture: Texture):
        self.textures[name] = texture

    def bind_textures(self):
        for name, texture in self.textures.items():
            texture.bind()
            texture_unit = texture.get_texture_slot()

            self.shader.setTexture(name, texture_unit)

    def unbind_textures(self):
        for texture in self.textures.values():
            texture.unbind()

    def set_instance_buffers(self, instance_data: List[np.ndarray], instance_buffer_layout: List[DataLayout]):
        if not isinstance(instance_data, List):
            instance_data = [instance_data]
        if not isinstance(instance_buffer_layout, List):
            instance_buffer_layout = [instance_buffer_layout]
        self.instance_vbos = self._set_vertex_buffers(instance_data, instance_buffer_layout)
        self.num_instances = len(instance_data[0]) // instance_buffer_layout[0].num_elements

    def set_instance_data(self, instance_buffer_index: int, instance_data: np.ndarray,
                          instance_buffer_layout: Optional[DataLayout] = None):
        if len(instance_data) != self.num_instances:
            raise RuntimeError("When providing instance data, the number of instances must be the same")
        num_instance_vbos = len(self.instance_vbos)
        if num_instance_vbos < instance_buffer_index:
            raise RuntimeError("Trying to set the data of instance buffer at a mesh instance buffer index that doesn't"
                               " exist")
        elif num_instance_vbos == instance_buffer_index:
            if instance_buffer_layout is None:
                raise RuntimeError("Trying to set new instance buffer in mesh without providing a buffer layout")
            self.instance_vbos.append(self.vao.add_vbo(instance_data, instance_buffer_layout))
        else:
            ibo = self.instance_vbos[instance_buffer_index]
            ibo.bind()
            ibo.buffer_data(instance_data)
            ibo.unbind()

    def bind(self):
        self.shader.use()
        self.vao.bind()

    def unbind(self):
        self.vao.unbind()

    def draw(self):
        self.bind()  # Bind the shader and VAO

        if self.num_instances > 1:
            # Draw with instancing if there are multiple instances
            glDrawElementsInstanced(self.primitive, self.num_indices, GL_UNSIGNED_SHORT, None, self.num_instances)
        else:
            # Draw normally if only one instance
            glDrawElements(self.primitive, self.num_indices, GL_UNSIGNED_SHORT, None)

        self.unbind()  # Unbind the VAO after drawing


class ShapeMesh(Mesh):
    def __init__(self, shape: _BaseShape, shader: ShaderProgram, textures: Optional[Dict[str, Texture]] = None):
        super().__init__(
            shape.vertices, shape.vertex_layout,
            shape.indices, shader, shape.primitive, textures, shape.instances, shape.instance_layout
        )


class LineMesh(ShapeMesh):
    def __init__(self, start: glm.vec3, end: glm.vec3, shader: ShaderProgram, colour: glm.vec3 = Color.BLACK,
                 alpha: float = 1.0, thickness=1.0):
        super().__init__(Line(start, end, thickness, colour, alpha), shader)


class ColQuadMesh(ShapeMesh):
    def __init__(self, shader: ShaderProgram, colours: Optional[Union[glm.vec3, List[glm.vec3]]] = None,
                 alphas: Optional[Union[float, List[float]]] = None, quad_model_mats: Optional[List[glm.mat4]] = None):
        super().__init__(ColQuad(colours, alphas, quad_model_mats), shader)


class TexQuadMesh(ShapeMesh):
    def __init__(self, shader: ShaderProgram, tex_uvs: Optional[Union[glm.vec3, List[glm.vec3]]] = None,
                 alphas: Optional[Union[float, List[float]]] = None, quad_model_mats: Optional[List[glm.mat4]] = None):
        super().__init__(ColQuad(tex_uvs, alphas, quad_model_mats), shader)


class BlockMesh(Mesh):
    def __init__(self, block_model_mats: Union[glm.mat4, List[glm.mat4]],
                 block_face_shape: BlockShape, shader: ShaderProgram,
                 textures: Optional[Dict[str, Texture]] = None):
        super().__init__(
            block_face_shape.vertices, block_face_shape.vertex_layout, block_face_shape.indices, shader,
            textures=textures
        )
        model_mats: np.ndarray = None
        if isinstance(block_model_mats, glm.mat4):
            block_model_mats = [block_model_mats]
        elif not isinstance(block_model_mats, List):
            raise TypeError("Block Model Mats should be of type glm.mat4 or List")
        if isinstance(block_model_mats[0], glm.mat4):
            model_mats = np.array(
                [
                    np.array(block_model_mat, np.float32).reshape(4, 4).T
                    for block_model_mat in block_model_mats
                ],
                np.float32
            ).flatten()
        else:
            raise TypeError("Block Model Mats should be a list of glm.mat4s")

        self.set_instance_buffers(
            [
                model_mats,
                block_face_shape.instances
            ],
            [
                DataLayout([VA.Float.Mat4("a_Model", divisor=1)]),
                block_face_shape.instance_layout
            ]
        )
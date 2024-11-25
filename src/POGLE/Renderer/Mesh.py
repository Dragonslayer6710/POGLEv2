import numpy as np

from POGLE.Shader import *
from POGLE.Geometry.Texture import *
from POGLE.Geometry.Shape import *

class Mesh:
    def __init__(self, indices: np.ndarray, data_layout: DataLayout,  num_instances: int, shader: ShaderProgram,
                 primitive: GLenum, textures: Optional[Dict[str, Texture]] = None,
                 print_buffers=False, print_attributes=False):
        self.vao: VertexArray = VertexArray()
        self.vao.set_ebo(data=indices, print_buffer=print_buffers)
        self.data_layout: DataLayout = data_layout
        self.vao.add_vbo(data_layout, print_buffer=print_buffers)

        self.num_instances: int = num_instances

        self.vao.set_ebo(data=indices)
        self.num_indices = len(indices)
        self.shader = shader
        self.primitive = primitive

        self.textures: Dict[str, Texture] = textures if textures is not None else {}

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

    def bind_uniform_blocks(self, uniform_block_names: Union[str, List[str]]):
        if isinstance(uniform_block_names, str):
            uniform_block_names = [uniform_block_names]
        [self.shader.bind_uniform_block(uniform_block_name) for uniform_block_name in uniform_block_names]

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
    def __init__(self, shape: Shape, num_instances: int,
                 shader: ShaderProgram, textures: Optional[Dict[str, Texture]] = None,
                 print_buffers=False, print_attributes=False):
        super().__init__(
            shape.indices, shape.data_layout, num_instances, shader, shape.primitive, textures,
            print_buffers, print_attributes
        )


class LineMesh(ShapeMesh):
    def __init__(self, start: glm.vec3, end: glm.vec3, shader: ShaderProgram, colour: glm.vec3 = Color.BLACK,
                 alpha: float = 1.0, thickness=1.0):
        super().__init__(Line(start, end, thickness, colour, alpha), 1, shader)


class ColQuadMesh(ShapeMesh):
    def __init__(self, shader: ShaderProgram, colours: Optional[Union[glm.vec3, List[glm.vec3]]] = None,
                 alphas: Optional[Union[float, List[float]]] = None, quad_model_mats: Optional[List[glm.mat4]] = None):
        num_instances = len(quad_model_mats)
        super().__init__(ColQuad(colours, alphas, quad_model_mats), num_instances, shader)


class TexQuadMesh(ShapeMesh):
    def __init__(self, shader: ShaderProgram, textures: Optional[Dict[str, Texture]] = None,
                 tex_uvs: Optional[Union[glm.vec3, List[glm.vec3]]] = None,
                 alphas: Optional[Union[float, List[float]]] = None, quad_model_mats: Optional[List[glm.mat4]] = None):
        num_instances = len(quad_model_mats)
        super().__init__(ColQuad(tex_uvs, alphas, quad_model_mats), num_instances, shader)
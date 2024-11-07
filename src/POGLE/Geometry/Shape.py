import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union, TypeVar, Type
import glm

from POGLE.Core.Core import Color, interleave_attributes, filter_out_none
from POGLE.Geometry.Data import DataLayout, VertexAttribute
from POGLE.OGL.OpenGLContext import *


@dataclass
class _Shape:
    indices: np.ndarray
    _vert_positions: np.ndarray
    _vert_colours: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    _vert_alphas: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    _vert_tex_uvs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    _inst_model_mats: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __post_init__(self):

        self.vertices: np.ndarray = interleave_attributes(
            self.get_vertex_data_lists()
        )

        self.vertex_layout: DataLayout = DataLayout(
            filter_out_none([
                        VertexAttribute("a_Position"),
                        *self.list_vert_attrs()
            ])
        )

        inst_attrs = False
        for attr_name, value in self.__dict__.items():
            if attr_name.startswith("_inst_"):
                if len(value):
                    inst_attrs = True
                    break

        self.instances: Optional[np.ndarray] = None
        self.instance_layout: Optional[DataLayout] = None
        if inst_attrs:
            self.instances = interleave_attributes(
                self.get_instance_data_lists()
            )
            self.instance_layout = DataLayout(
                filter_out_none(self.list_inst_attrs())
            )

    def get_vertex_data_lists(self) -> List[np.ndarray]:
        return [
            self._vert_positions,
            self._vert_colours if len(self._vert_colours) else None,
            self._vert_alphas if len(self._vert_alphas) else None,
            self._vert_tex_uvs if len(self._vert_tex_uvs) else None
        ]

    def list_vert_attrs(self) -> List[VertexAttribute]:
        return [
            VertexAttribute.Float.Vec3("a_Colour") if len(self._vert_colours) else None,
            VertexAttribute.Float.Scalar("a_Alpha") if len(self._vert_alphas) else None,
            VertexAttribute.Float.Vec2("a_TexUV") if len(self._vert_tex_uvs) else None
        ]

    def list_inst_attrs(self) -> List[VertexAttribute]:
        return [
            VertexAttribute.Float.Mat4("a_Model", divisor=1) if len(self._inst_model_mats) else None,
        ]

    def get_instance_data_lists(self) -> List[np.ndarray]:
        return [
            self._inst_model_mats if len(self._inst_model_mats) else None
        ]



class _BaseShape(_Shape):
    _indices: Optional[List[int]] = None
    _positions: Optional[List[glm.vec3]] = None
    _num_vertices: int = 2

    primitive: GLenum = GL_TRIANGLES

    def __init__(self, colours: Optional[List[glm.vec3]] = None, alphas: Optional[List[float]] = None,
                 tex_uvs: Optional[List[glm.vec3]] = None, model_mats: Optional[List[glm.mat4]] = None):
        super().__init__(
            indices=np.array(self.__class__._indices, np.ushort),
            _vert_positions=np.array(self._positions, dtype=np.float32),
            _vert_colours=np.array([] if colours is None else colours, dtype=np.float32),
            _vert_alphas=np.array([] if alphas is None else alphas, dtype=np.float32),
            _vert_tex_uvs=np.array([] if tex_uvs is None else tex_uvs, dtype=np.float32),
            _inst_model_mats=np.array(
                [] if model_mats is None else
                [
                    np.array(model_mat, dtype=np.float32).reshape(4, 4).T.flatten() for model_mat in model_mats
                ],
                dtype=np.float32
            ),
        )


_A = TypeVar("_A")


def _validate_attribute(dtype: Type[_A], attribute_elements: Union[_A, List[_A]],
                        num_vertices: int, attrib_name: str = "Vec3 Attribute") -> List[_A]:
    _A = dtype
    if isinstance(attribute_elements, List):
        if len(attribute_elements) == num_vertices:
            return attribute_elements
        else:
            raise ValueError(
                f"Invalid number of elements for {attrib_name}. Expected {num_vertices}, got {len(attribute_elements)}."
            )
    elif isinstance(attribute_elements, _A):
        return [attribute_elements] * num_vertices
    raise TypeError(f"Invalid type for {attrib_name}. Expected {_A} or List[{_A}].")


def _validate_floats(attribute_elements: Union[float, List[float]],
                     num_vertices: int, attrib_name: str = "Float Attribute") -> List[float]:
    return _validate_attribute(float, attribute_elements, num_vertices, attrib_name)


def _validate_alphas(alphas: Optional[Union[float, List[float]]], num_vertices: int) -> List[float]:
    return _validate_floats(alphas, num_vertices, "Alpha") if alphas else [1.0] * num_vertices


def _validate_vec2s(attribute_elements: Union[glm.vec2, List[glm.vec2]],
                    num_vertices: int, attrib_name: str = "Vec2 Attribute") -> List[glm.vec2]:
    return _validate_attribute(glm.vec2, attribute_elements, num_vertices, attrib_name)


def _validate_tex_uvs(tex_uvs: Union[glm.vec2, List[glm.vec2]], num_vertices: int) -> List[glm.vec2]:
    return _validate_vec2s(tex_uvs, num_vertices, "TexUV")


def _validate_vec3s(attribute_elements: Union[glm.vec3, List[glm.vec3]],
                    num_vertices: int, attrib_name: str = "Vec3 Attribute") -> List[glm.vec3]:
    return _validate_attribute(glm.vec3, attribute_elements, num_vertices, attrib_name)


def _validate_colours(colours: Union[glm.vec3, List[glm.vec3]], num_vertices: int) -> List[glm.vec3]:
    return _validate_vec3s(colours, num_vertices, "Color") if colours else [Color.BLACK] * num_vertices


class Triangle(_BaseShape):
    _indices = [
        0, 1, 2
    ]
    _num_vertices = 3


class Quad(_BaseShape):
    _indices = [
        0, 1, 2,
        2, 3, 0
    ]
    _positions = [
        glm.vec3(-1.0, -1.0, 0.0),  # Bottom left
        glm.vec3(-1.0,  1.0, 0.0),  # Bottom right
        glm.vec3( 1.0,  1.0, 0.0),  # Top left
        glm.vec3( 1.0, -1.0, 0.0)  # Top right
    ]
    _num_vertices = 4


class Cube(_BaseShape):
    _positions = [
        glm.vec3(-1.0, 1.0, 1.0),  # 0: Front Top Left
        glm.vec3(1.0, 1.0, 1.0),  # 1: Front Top Right
        glm.vec3(1.0, -1.0, 1.0),  # 2: Front Bottom Right
        glm.vec3(-1.0, -1.0, 1.0),  # 3: Front Bottom Left
        glm.vec3(1.0, 1.0, -1.0),  # 4: Back Top Right
        glm.vec3(-1.0, 1.0, -1.0),  # 5: Back Top Left
        glm.vec3(-1.0, -1.0, -1.0),  # 6: Back Bottom Left
        glm.vec3(1.0, -1.0, -1.0),  # 7: Back Bottom Right
    ]
    _indices = [
        # Front face (z = 1.0)
        0, 1, 2, 2, 3, 0,

        # Back face (z = -1.0)
        4, 5, 6, 6, 7, 4,

        # Top face (y = 1.0)
        0, 1, 4, 4, 5, 0,

        # Bottom face (y = -1.0)
        2, 3, 6, 6, 7, 2,

        # Right face (x = 1.0)
        1, 2, 7, 7, 4, 1,

        # Left face (x = -1.0)
        0, 5, 6, 6, 3, 0
    ]
    _num_vertices = 8


class Line(_BaseShape):
    _indices = [0, 1]
    primitive = GL_LINES

    def __init__(self, start: glm.vec3, end: glm.vec3, thickness: float = 1.0,
                 colours: Optional[List[glm.vec3]] = None, alphas: Optional[List[float]] = None,
                 model_mats: Optional[List[glm.mat4]] = None):
        self._positions = [start, end]
        self.thickness = thickness
        super().__init__(
            _validate_vec3s(colours, self._num_vertices),
            _validate_alphas(alphas, self._num_vertices),
            model_mats=model_mats
        )


class WireframeQuad(Quad):
    _indices = [
        0, 1,  # Top edge
        1, 2,  # Right edge
        2, 3,  # Bottom edge
        3, 0,  # Left edge
    ]
    primitive = GL_LINES

    def __init__(self, thickness: int = 1.0, colour: Optional[glm.vec3] = None, alphas: Optional[List[float]] = None,
                 model_mats: Optional[List[glm.mat4]] = None):
        self.thickness: int = thickness
        if colour is None:
            colour = Color.BLACK
        if not isinstance(colour, glm.vec3):
            raise ValueError("Wireframe shapes must receive only one colour")
        super().__init__(
            _validate_colours(colour, self._num_vertices),
            _validate_alphas(alphas, self._num_vertices),
            model_mats=model_mats
        )


class ColQuad(Quad):
    def __init__(self, colours: Union[glm.vec3, List[glm.vec3]], alphas: Optional[List[float]] = None,
                 model_mats: Optional[List[glm.mat4]] = None):
        super().__init__(
            _validate_vec3s(colours, self._num_vertices),
            _validate_alphas(alphas, self._num_vertices),
            model_mats=model_mats
        )


class TexQuad(Quad):
    _tex_uvs = [
        glm.vec2(0.0, 0.0),
        glm.vec2(0.0, 1.0),
        glm.vec2(1.0, 1.0),
        glm.vec2(1.0, 0.0)
    ]

    def __init__(self, alphas: Optional[List[float]] = None,
                 model_mats: Optional[List[glm.mat4]] = None):
        super().__init__(tex_uvs=self._tex_uvs, alphas=_validate_alphas(alphas, self._num_vertices), model_mats=model_mats)


class WireframeCube(Cube):
    _indices = [
        # Front face edges
        0, 1,  # Top edge
        1, 2,  # Right edge
        2, 3,  # Bottom edge
        3, 0,  # Left edge

        # Back face edges
        4, 5,  # Top edge
        5, 6,  # Left edge
        6, 7,  # Bottom edge
        7, 4,  # Right edge

        # Connecting edges
        0, 5,  # Front top left to back top left
        1, 4,  # Front top right to back top right
        2, 7,  # Front bottom right to back bottom right
        3, 6  # Front bottom left to back bottom left
    ]
    primitive = GL_LINES

    def __init__(self, thickness: int = 1.0, colour: Optional[glm.vec3] = None, alphas: Optional[List[float]] = None,
                 model_mats: Optional[List[glm.mat4]] = None):
        self.thickness: int = thickness
        if colour is None:
            colour = Color.BLACK
        if not isinstance(colour, glm.vec3):
            raise ValueError("Wireframe shapes must receive only one colour")
        super().__init__(
            _validate_colours(colour, self._num_vertices),
            _validate_alphas(alphas, self._num_vertices),
            model_mats=model_mats
        )


class ColCube(Cube):
    def __init__(self, colours: Union[glm.vec3, List[glm.vec3]], alphas: Optional[List[float]] = None,
                 model_mats: Optional[List[glm.mat4]] = None):
        super().__init__(
            _validate_vec3s(colours, self._num_vertices),
            _validate_alphas(alphas, self._num_vertices),
            model_mats=model_mats
        )


class TexCube(Cube):
    def __init__(self, tex_uvs: List[glm.vec3], alphas: Optional[List[float]] = None,
                 model_mats: Optional[List[glm.mat4]] = None):
        super().__init__(tex_uvs=tex_uvs, alphas=_validate_alphas(alphas, self._num_vertices), model_mats=model_mats)

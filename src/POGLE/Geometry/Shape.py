import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union, TypeVar, Type
import glm

from POGLE.Core.Core import Color, interleave_attributes, filter_out_none
from POGLE.Geometry.Data import DataLayout, VertexAttribute, NewModelMatrix
from POGLE.OGL.OpenGLContext import *


class Shape:
    primitive: GLenum = GL_TRIANGLES

    def __init__(self,
                 indices: Union[List[int], np.ndarray],
                 positions: Union[List[glm.vec3], np.ndarray],
                 colours: Optional[Union[List[glm.vec3], np.ndarray]] = None,
                 tex_coords: Optional[Union[List[glm.vec2], np.ndarray]] = None,
                 alphas: Optional[Union[List[float], np.ndarray]] = None,
                 model_mats: Optional[Union[List[glm.mat4], np.ndarray]] = None,
                 model_div: int = 1,
                 extra_attrs: Optional[List[VertexAttribute]] = None
                 ):
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices, np.ushort)
        self.indices = indices
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions, np.float32)
        attributes: List[VertexAttribute] = [VertexAttribute("a_Position", positions)]
        self._num_vertices = len(positions)

        if colours is not None:
            if not isinstance(colours, np.ndarray):
                colours = np.array(colours, np.float32)
            attributes.append(VertexAttribute("a_Colour", colours))
        if tex_coords is not None:
            if not isinstance(tex_coords, np.ndarray):
                tex_coords = np.array(tex_coords, np.float32)
            attributes.append(VertexAttribute("a_TexUV", tex_coords))
        if alphas is not None:
            if not isinstance(alphas, np.ndarray):
                alphas = np.array(alphas, np.float32)
            attributes.append(VertexAttribute("a_Alpha", alphas))

        if model_mats is not None:
            if not isinstance(model_mats, np.ndarray):
                model_mats = np.array(model_mats).reshape(-1, 4, 4).transpose(0, 2, 1)
            attributes.append(VertexAttribute("a_Model", model_mats, divisor=model_div))

        if extra_attrs is not None:
            attributes.extend(extra_attrs)
        self.data_layout = DataLayout(attributes)

class Triangle(Shape):
    _indices = np.array([
        0, 1, 2
    ], np.ushort)

    def __init__(self, *args, **kwargs):
        super().__init__(indices=self.__class__._indices, *args, **kwargs)


class EqualTriangle(Triangle):
    _positions = np.array([
        glm.vec3(glm.vec2(0, 1), 0),
        glm.vec3(glm.vec2(1, 0), 0),
        glm.vec3(glm.vec2(-1, 0), 0),
    ], np.float32)

    def __init__(self, *args, **kwargs):
        super().__init__(positions=self.__class__._positions, *args, **kwargs)


class Quad(Shape):
    _indices = np.array([
        0, 1, 2,
        2, 3, 0
    ], np.ushort)
    _positions = np.array([
        glm.vec3(-1.0, -1.0, 0.0),  # Bottom left
        glm.vec3(-1.0, 1.0, 0.0),  # Bottom right
        glm.vec3(1.0, 1.0, 0.0),  # Top left
        glm.vec3(1.0, -1.0, 0.0)  # Top right
    ], np.float32)

    def __init__(self, *args, **kwargs):
        super().__init__(indices=self.__class__._indices, positions=self.__class__._positions, *args, **kwargs)


class Cube(Shape):
    _indices = np.array([
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
    ], np.ushort)
    _positions = np.array([
        glm.vec3(-1.0, 1.0, 1.0),  # 0: Front Top Left
        glm.vec3(1.0, 1.0, 1.0),  # 1: Front Top Right
        glm.vec3(1.0, -1.0, 1.0),  # 2: Front Bottom Right
        glm.vec3(-1.0, -1.0, 1.0),  # 3: Front Bottom Left
        glm.vec3(1.0, 1.0, -1.0),  # 4: Back Top Right
        glm.vec3(-1.0, 1.0, -1.0),  # 5: Back Top Left
        glm.vec3(-1.0, -1.0, -1.0),  # 6: Back Bottom Left
        glm.vec3(1.0, -1.0, -1.0),  # 7: Back Bottom Right
    ], np.float32)

    def __init__(self, *args, **kwargs):
        super().__init__(indices=self.__class__._indices, positions=self.__class__._positions, *args, **kwargs)


class Line(Shape):
    primitive = GL_LINES
    _indices = np.array([0, 1], np.ushort)

    def __init__(self, start: glm.vec3, end: glm.vec3, thickness: float = 1.0, *args, **kwargs):
        super().__init__(
            indices=self.__class__._indices,
            positions=np.array([start, end], np.float32),
            *args, **kwargs
        )
        self.thickness: float = thickness


class WireframeQuad(Quad):
    _indices = np.array([
        0, 1,  # Top edge
        1, 2,  # Right edge
        2, 3,  # Bottom edge
        3, 0,  # Left edge
    ], np.ushort)

    def __init__(self, thickness: int = 1.0, colour: glm.vec3 = Color.BLACK, *args, **kwargs):
        super().__init__(colour=[colour] * len(self._positions))
        self.thickness: int = thickness


class ColQuad(Quad):
    def __init__(self,
                 colours: Union[glm.vec3, List[glm.vec3]],
                 alphas: Optional[Union[float, List[float]]],
                 *args, **kwargs):
        super().__init__(
            colours=[colours] * len(self._positions) if isinstance(colours, glm.vec3) else colours,
            alphas=[alphas] * len(self._positions) if isinstance(alphas, glm.vec3) else alphas,
            *args, **kwargs
        )


class TexQuad(Quad):
    _tex_coords = [
        glm.vec2(0.0, 0.0),
        glm.vec2(0.0, 1.0),
        glm.vec2(1.0, 1.0),
        glm.vec2(1.0, 0.0)
    ]

    def __init__(self,
                 alphas: Optional[Union[float, List[float]]],
                 *args, **kwargs):
        super().__init__(
            tex_coords=self._tex_coords,
            alphas=[alphas] * len(self._positions) if isinstance(alphas, glm.vec3) else alphas,
            *args, **kwargs
        )


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

    def __init__(self, thickness: int = 1.0, colour: glm.vec3 = Color.BLACK, *args, **kwargs):
        super().__init__(colour=[colour] * len(self._positions))
        self.thickness: int = thickness


class ColCube(Cube):
    def __init__(self,
                 colours: Union[glm.vec3, List[glm.vec3]],
                 alphas: Optional[Union[float, List[float]]],
                 *args, **kwargs):
        super().__init__(
            colours=[colours] * len(self._positions) if isinstance(colours, glm.vec3) else colours,
            alphas=[alphas] * len(self._positions) if isinstance(alphas, glm.vec3) else alphas,
            *args, **kwargs
        )


class TexCube(Cube):
    _tex_coords = [
        glm.vec2(0.0, 0.0),
        glm.vec2(0.0, 1.0),
        glm.vec2(1.0, 1.0),
        glm.vec2(1.0, 0.0)
    ]

    def __init__(self,
                 alphas: Optional[Union[float, List[float]]],
                 *args, **kwargs):
        raise NotImplementedError("TexCube not yet implemented")
        super().__init__(
            tex_coords=self._tex_coords,
            alphas=[alphas] * len(self._positions) if isinstance(alphas, glm.vec3) else alphas,
            *args, **kwargs
        )
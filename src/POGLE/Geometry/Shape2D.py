from POGLE.Geometry.Vertex import *

class Shape:
    positions = []
    indices: list[int] = []
    def __init__(self, colors:list[glm.vec3]=None, alphas:list[float] = None):
        self.vertexCount = len(self.positions)
        if colors is None:
            colors = self.vertexCount * [glm.vec3()]
        elif type(colors) != list:
            colors = self.vertexCount * [colors]

        if alphas is None:
            alphas = self.vertexCount * [[alphas]]
        elif type(alphas) != list:
            alphas = self.vertexCount * [alphas]

        self.vertices = Vertices(interleave_arrays(
            self.positions,
            colors,
            alphas
        ))


class Quad(Shape):
    positions = [
        glm.vec3(-1.0, -1.0,  0.0),
        glm.vec3( 1.0, -1.0,  0.0),
        glm.vec3(-1.0,  1.0,  0.0),
        glm.vec3( 1.0,  1.0,  0.0)
    ]
    indices = [
        0, 1, 2,
        1, 2, 3
    ]

    def __init__(self, colors: list[glm.vec3] = Color.WHITE, alphas: list[float] = 1.0):
        super().__init__(colors, alphas)


class Pentagon(Shape):
    positions = [
        glm.vec3(    0.0,     1.0,  0.0),
        glm.vec3( 0.9511,  0.3090,  0.0),
        glm.vec3( 0.5878, -0.8090,  0.0),
        glm.vec3(-0.5878, -0.8090,  0.0),
        glm.vec3(-0.9511,  0.3090,  0.0)
    ]
    indices = [
        0, 1, 2,
        0, 2, 4,
        2, 3, 4
    ]

    def __init__(self, colors: list[glm.vec3] = Color.WHITE, alphas: list[float] = 1.0):
        super().__init__(colors, alphas)




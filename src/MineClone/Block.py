from POGLE.Core.Application import *

class Block:
    class ID(Enum):
        Grass = auto()
        Stone = auto()
        Dirt = auto()

    class Side(Enum):
        East = auto()
        South = auto()
        West = auto()
        North = auto()
        Top = auto()
        Bottom = auto()

    class TexSide(Enum):
        GrassTop    = auto()
        Stone       = auto()
        Dirt        = auto()
        GrassSide   = auto()

    _TextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))

    face_matrices = [
        NewModelMatrix(glm.vec3(-1.0, 0.0, 0.0), glm.vec3(0, - 90, 0)),
        NewModelMatrix(glm.vec3(0.0, 0.0, -1.0)),
        NewModelMatrix(glm.vec3(1.0, 0.0, 0.0), glm.vec3(0, 90, 0)),
        NewModelMatrix(glm.vec3(0.0, 0.0, 1.0), glm.vec3(0, 180, 0)),
        NewModelMatrix(glm.vec3(0.0, 1.0, 0.0), glm.vec3(90, 0, 0)),
        NewModelMatrix(glm.vec3(0.0, -1.0, 0.0), glm.vec3(- 90, 0, 0)),
    ]

    _blockInstLayout = VertexLayout([
        FloatVA.Vec2(1),  # Texture Coord
        FloatVA.Vec2(1),  # Texture Size
        FloatVA.Vec4(),  # Model Matrix
    ])

    instances = []

    blockNets: dict[ID, list[TexSide]] = {

    }

    def __init__(self, modelMatrix: glm.mat4, id: ID = ID.Grass):
        self.ID: Block.ID = id
        for i in range(6):
            texDims = self._TextureAtlas.get_sub_texture(self.blockNets[id][i])
            self.instances.append([modelMatrix * self.face_matrices[i], texDims.pos, texDims.size])

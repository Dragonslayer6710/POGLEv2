from POGLE.Core.Application import *

_QUADS_IN_BLOCK = 6
class Block:
    class ID(Enum):
        Null = 0
        Grass = auto()
        Stone = auto()
        Dirt = auto()

    transparentBlocks: list[ID] = [
        ID.Null
    ]

    class Side(Enum):
        East = 0
        South = auto()
        West = auto()
        North = auto()
        Top = auto()
        Bottom = auto()

    class TexSide(Enum):
        GrassTop    = 0
        Stone       = auto()
        Dirt        = auto()
        GrassSide   = auto()

    _TextureAtlas = None

    blockNets: dict[ID, list[TexSide]] = {
        ID.Grass : [*[TexSide.GrassSide] * 4, TexSide.GrassTop, TexSide.Dirt],
        ID.Stone: [TexSide.Stone] * 6,
        ID.Dirt: [TexSide.Dirt] * 6
    }

    vertices: Vertices = None
    indices: np.array = Quad.indices
    instanceLayout: VertexLayout = VertexLayout([
        FloatVA.Vec2(1),
        FloatVA.Vec2(1),
        FloatVA.Mat4()
    ])

    adjBlockOffsets: dict[Side, glm.vec3] = {
        Side.East   : glm.vec3(-1, 0, 0),
        Side.South  : glm.vec3( 0, 0, 1),
        Side.West   : glm.vec3( 1, 0, 0),
        Side.North  : glm.vec3( 0, 0,-1),
        Side.Top    : glm.vec3( 0, 1, 0),
        Side.Bottom : glm.vec3( 0,-1, 0)
    }

    def __init__(self, chunkBlockPos: glm.vec3, id: ID = ID.Null):
        if None != chunkBlockPos:
            self.chunkBlockPos: glm.vec3 = chunkBlockPos
            self.worldBlockPos: glm.vec3 = None
            self.adjBlocks: dict[Block.Side, Block] = None
        if not id == Block.ID.Null:
            self.set_block(id)
        else:
            self.blockID = id
        self.visibleSides: dict[Block.Side, bool] = {
            Block.Side.East: True,
            Block.Side.South: True,
            Block.Side.West: True,
            Block.Side.North: True,
            Block.Side.Top: True,
            Block.Side.Bottom: True
        }
        self.is_transparent = self.blockID in Block.transparentBlocks

    def set(self, chunk, chunkBlockID, id: ID):
        from MineClone.Chunk import Chunk
        self.chunk: Chunk = chunk
        self.chunkBlockID = chunkBlockID
        self.worldBlockPos = self.chunk.get_world_pos(self.chunkBlockPos)
        self.blockID: Block.ID = id
        self.is_transparent = self.blockID in Block.transparentBlocks

    def init(self):
        if self.is_block():
            if None == Block._TextureAtlas:
                Block._TextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))
            self.inst_data_texPos = []
            self.inst_data_texSize = []
            for i in range(6):
                texDims = Block._TextureAtlas.get_sub_texture(self.blockNets[self.blockID][i].value)
                self.inst_data_texPos.append(texDims.pos)
                self.inst_data_texSize.append(texDims.size)
            texQuadCube = TexQuadCube(NMM(self.worldBlockPos), self.inst_data_texPos, self.inst_data_texSize)
            if not Block.vertices:
                Block.vertices = texQuadCube.vertices
            self.face_instances: list[np.ndarray] = split_array(texQuadCube.instances.data, 6)

            self.adjBlocks = {
                side: self.chunk.get_block(self.worldBlockPos + offset) for side, offset in self.adjBlockOffsets.items()
            }

            self.update_side_visibility()

    def update_side_visibility(self) -> bool:
        updated = False
        if self.is_block():
            for side, adjBlock in self.adjBlocks.items():
                if self.face_visible(side):
                    if not adjBlock.is_transparent:
                        self.hide_face(side)
                        updated = True
                else:
                    if adjBlock.is_transparent:
                        self.reveal_face(side)
                        updated = True
        return updated

    def edit_side_state(self, side: Side, state: bool):
        self.visibleSides[side] = state

    def face_visible(self, side: Side) -> bool:
        return self.visibleSides[side]

    def hide_face(self, side: Side):
        self.edit_side_state(side, False)

    def reveal_face(self, side: Side):
        self.edit_side_state(side, True)

    def is_block(self) -> bool:
        return self.blockID != Block.ID.Null

    def get_instance_data(self) -> np.ndarray:
        if self.is_block():
            instance_data = np.array([])
            cnt = 0
            for side in Block.Side:
                if self.visibleSides[side]:
                    cnt+=1
                    instance_data = np.concatenate((instance_data, self.face_instances[side.value]), dtype=self.face_instances[side.value].dtype)
            if cnt == 0:
                return None
            return instance_data
        return None

BLOCK_NULL = Block(None)
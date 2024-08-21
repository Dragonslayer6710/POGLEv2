import struct

from POGLE.Core.Application import *
from POGLE.Physics.SpatialTree import *
from POGLE.Renderer.Mesh import WireframeCubeMesh

_QUADS_IN_BLOCK = 6
class Block(PhysicalBox):
    class ID(Enum):
        Null = 0
        Grass = auto()
        Stone = auto()
        Dirt = auto()

    transparentBlocks: list[ID] = [
        ID.Null
    ]

    class Side(Enum):
        West = 0
        South = auto()
        East = auto()
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
        Side.West   : glm.vec3(-1, 0, 0),
        Side.South  : glm.vec3( 0, 0, 1),
        Side.East   : glm.vec3( 1, 0, 0),
        Side.North  : glm.vec3( 0, 0,-1),
        Side.Top    : glm.vec3( 0, 1, 0),
        Side.Bottom : glm.vec3( 0,-1, 0)
    }

    def __init__(self, chunkBlockPos: glm.vec3, id: ID = ID.Null):
        super().__init__()
        if None != chunkBlockPos:
            self.chunkBlockPos: glm.vec3 = chunkBlockPos
            self.adjBlocks: dict[Block.Side, Block] = None
            self.visibleSides: dict[Block.Side, bool] = {
                Block.Side.West: True,
                Block.Side.South: True,
                Block.Side.East: True,
                Block.Side.North: True,
                Block.Side.Top: True,
                Block.Side.Bottom: True
            }
        self.blockID: Block.ID = id
        self.is_block: bool = self.blockID != Block.ID.Null
        self.is_transparent: bool = self.blockID in Block.transparentBlocks
        self.initialised: bool = False
    def init(self, chunk, chunkBlockID, id: ID):
        from MineClone.Chunk import Chunk
        self.initialised = True

        self.chunk: Chunk = chunk
        self.chunkBlockID = chunkBlockID
        self.bounds = AABB.from_pos_size(self.chunk.get_world_pos(self.chunkBlockPos))
        self.blockID: Block.ID = id
        self.is_block = self.blockID != Block.ID.Null
        self.is_transparent = self.blockID in Block.transparentBlocks
        texQuadCube = TexQuadCube(NMM(self.pos), glm.vec2(), glm.vec2())
        if self.is_block:
            if None == Block._TextureAtlas:
                Block._TextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))
                Block.vertices = texQuadCube.vertices
        self.face_instances: list[np.ndarray] = split_array(texQuadCube.instances.data, 6)
        self.update_face_textures()
        self.adjBlocks: dict[Block.Side, Block] = {
            side: self.chunk.get_block(self.pos + offset) for side, offset in self.adjBlockOffsets.items()
        }

    def set(self, id: ID):
        self.blockID: Block.ID = id
        self.is_block = self.blockID != Block.ID.Null
        was_transparent = self.is_transparent
        self.is_transparent = self.blockID in Block.transparentBlocks
        self.update_side_visibility()
        self.update_face_textures()
        if was_transparent != self.is_transparent:
            for block in self.adjBlocks.values():
                if block:
                    if block.update_side_visibility():
                        block.chunk.set_block_instance(block)
        self.chunk.set_block(self.chunkBlockID, self.blockID)


    def update_face_textures(self):
        for i in range(6):
            if self.blockID is not Block.ID.Null:
                texDims = Block._TextureAtlas.get_sub_texture(self.blockNets[self.blockID][i].value)
                self.face_instances[i][:4] = [*texDims.pos, *texDims.size]
            else:
                self.face_instances[i][:4] = [0.0 for i in range(4)]

    def update_side_visibility(self) -> bool:
        updated = False
        if self.is_block:
            for side, adjBlock in self.adjBlocks.items():
                if not adjBlock:
                    self.reveal_face(side)
                    updated = True
                    continue
                elif self.face_visible(side) and adjBlock:
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

    def get_adjblock_at_segment_intersect(self, intersect: glm.vec3):
        if intersect.x == self.max.x:
            return self.adjBlocks[Block.Side.East]
        elif intersect.x == self.min.x:
            return self.adjBlocks[Block.Side.West]
        elif intersect.y == self.max.y:
            return self.adjBlocks[Block.Side.Top]
        elif intersect.y == self.min.y:
            return self.adjBlocks[Block.Side.Bottom]
        elif intersect.z == self.min.z:
            return self.adjBlocks[Block.Side.North]
        else:
            return self.adjBlocks[Block.Side.South]

    def get_face_instance_data(self) -> np.ndarray:
        if self.is_block:
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

    def get_wireframe_cube_mesh(self) -> WireframeCubeMesh | None:
        if self.is_block:
            wfqCube = WireframeQuadCube(NMM(self.pos),Color.BLACK)
            wire_frame_faces = split_array(wfqCube.instances.data, 6)
            instance_data = np.array([])
            cnt = 0
            for side in Block.Side:
                if self.visibleSides[side]:
                    cnt+=1
                    instance_data = np.concatenate((instance_data, wire_frame_faces[side.value]), dtype=wire_frame_faces[side.value].dtype)
            if cnt == 0:
                return None
            return WireframeCubeMesh(wfqCube,
                              Instances(instance_data, wfqCube.instances.layout, True))
        return None

    def __str__(self):
        return f"Block(id: {self.blockID}, pos: {self.pos})"

    def serialize(self) -> bytes:
        # Pack the chunkBlockPos (3 floats) and blockID (int) into bytes
        return struct.pack("fffI", self.chunkBlockPos.x, self.chunkBlockPos.y, self.chunkBlockPos.z, self.blockID.value)

    @classmethod
    def deserialize(cls, binary_data: bytes):
        # Unpack 3 floats and 1 unsigned int (the block ID) from the binary data
        x, y, z, blockID = struct.unpack("fffI", binary_data)

        # Create a new instance of Block with the deserialized data
        return cls(glm.vec3(x, y, z), Block.ID(blockID))

import copy
import struct

from POGLE.Core.Application import *
from POGLE.Physics.SpatialTree import *
from POGLE.Renderer.Mesh import WireframeCubeMesh

_QUADS_IN_BLOCK = 6
class Block(PhysicalBox):
    class ID(Enum):
        Air = 0
        Grass = auto()
        Stone = auto()
        Dirt = auto()

    transparentBlocks: list[ID] = [
        ID.Air
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
        FloatDA.Vec2(1),
        FloatDA.Vec2(1),
        FloatDA.Mat4()
    ])

    adjBlockOffsets: dict[Side, glm.vec3] = {
        Side.West   : glm.vec3(-1, 0, 0),
        Side.South  : glm.vec3( 0, 0, 1),
        Side.East   : glm.vec3( 1, 0, 0),
        Side.North  : glm.vec3( 0, 0,-1),
        Side.Top    : glm.vec3( 0, 1, 0),
        Side.Bottom : glm.vec3( 0,-1, 0)
    }

    visibleSide: dict[Side, bool] = {
        Side.West: True,
        Side.South: True,
        Side.East: True,
        Side.North: True,
        Side.Top: True,
        Side.Bottom: True
    }

    oppositeSide: dict[Side, Side] = {
        Side.West: Side.East,
        Side.East: Side.West,
        Side.South: Side.North,
        Side.North: Side.South,
        Side.Top: Side.Bottom,
        Side.Bottom: Side.Top
    }

    def __init__(self, offset_from_chunk: glm.vec3, id: ID = ID.Air):
        from MineClone.Chunk import Chunk
        super().__init__()
        self.offset_from_chunk: glm.vec3 = offset_from_chunk
        self.id_from_chunk: int = Chunk.block_id_from_chunk_block_pos(offset_from_chunk)
        self._adjBlockCache: dict[Block.Side, Block | None] = {}
        self.adjBlockPositions: dict[Block.Side, glm.vec3] = {k: v + self.offset_from_chunk for k, v in Block.adjBlockOffsets.items()}
        self.visibleSide: dict[Block.Side, bool] = copy.deepcopy(Block.visibleSide)
        self.id: Block.ID = id

        self.is_solid: bool = self.id != Block.ID.Air

        self.was_transparent: bool = True
        self.is_transparent: bool = self.id in Block.transparentBlocks
        self.initialised: bool = False

        self.chunk: Chunk = None

        self.face_instances: list | None = None

    def link_chunk(self, chunk):
        self.chunk = chunk
        self.bounds = AABB.from_pos_size(self.chunk.get_world_pos(self.offset_from_chunk) + 0.5)
        if not self.face_instances:
            texQuadCube = TexQuadCube(self.pos, glm.vec2(), glm.vec2())
            if None == Block._TextureAtlas:
                Block._TextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))
                Block.vertices = texQuadCube.vertices
            self.face_instances: list[np.ndarray] = split_array(texQuadCube.instances.data, 6)
        self.update_face_textures()


    def adjBlock(self, side: Side):
        if side not in self._adjBlockCache.keys():
            self._adjBlockCache[side] = self.chunk.get_block(self.adjBlockPositions[side])
        return self._adjBlockCache[side]

    def update_transparency(self):
        self.was_transparent = self.is_transparent
        self.is_transparent = self.id in Block.transparentBlocks

    def set(self, id: ID):
        self.id: Block.ID = id
        self.is_solid = self.id != Block.ID.Air
        self.update()

    def update(self):
        self.update_transparency()
        self.update_face_textures()
        self.update_side_visibility()

        self.chunk.update_block_in_chunk(self)

    def update_face_textures(self):
        for i in range(6):
            if self.id is not Block.ID.Air:
                texDims = Block._TextureAtlas.get_sub_texture(self.blockNets[self.id][i].value)
                self.face_instances[i][:4] = [*texDims.pos, *texDims.size]
            else:
                self.face_instances[i][:4] = [0.0 for i in range(4)]

    def update_side_visibility(self) -> bool:
        updated = False
        if self.is_solid:
            for side in Block.Side:
                adjBlock = self.adjBlock(side)

                if not adjBlock:
                    # Reveal face if there is no adjacent block (exposed to outside)
                    self.reveal_face(side)
                    updated = True
                else:
                    # determine face visibility based on adjacent block's transparency
                    if self.face_visible(side):
                        if not adjBlock.is_transparent:
                            # Hide face if adjacent block is not transparent
                            self.hide_face(side)
                            updated = True
                    elif adjBlock.is_transparent:
                        # Reveal face if adjacent block is transparent
                        self.reveal_face(side)
                        updated = True
        return updated

    def edit_side_state(self, side: Side, state: bool):
        self.visibleSide[side] = state

    def face_visible(self, side: Side) -> bool:
        return self.visibleSide[side]

    def hide_face(self, side: Side):
        self.edit_side_state(side, False)

    def reveal_face(self, side: Side):
        self.edit_side_state(side, True)

    def get_adjblock_at_segment_intersect(self, intersect: glm.vec3):
        if intersect.x == self.max.x:
            return self.adjBlock(Block.Side.East)
        elif intersect.x == self.min.x:
            return self.adjBlock(Block.Side.West)
        elif intersect.y == self.max.y:
            return self.adjBlock(Block.Side.Top)
        elif intersect.y == self.min.y:
            return self.adjBlock(Block.Side.Bottom)
        elif intersect.z == self.min.z:
            return self.adjBlock(Block.Side.North)
        else:
            return self.adjBlock(Block.Side.South)

    def get_face_instance_data(self) -> np.ndarray:
        if self.is_solid:
            instance_data = np.array([])
            cnt = 0
            for side in Block.Side:
                if self.visibleSide[side]:
                    cnt+=1
                    instance_data = np.concatenate((instance_data, self.face_instances[side.value]), dtype=self.face_instances[side.value].dtype)
            if cnt == 0:
                return None
            return instance_data
        return None

    def get_wireframe_cube_mesh(self) -> WireframeCubeMesh | None:
        if self.is_solid:
            wfqCube = WireframeQuadCube(self.pos, Color.BLACK)
            wire_frame_faces = split_array(wfqCube.instances.data, 6)
            instance_data = np.array([])
            cnt = 0
            for side in Block.Side:
                if self.visibleSide[side]:
                    cnt+=1
                    instance_data = np.concatenate((instance_data, wire_frame_faces[side.value]), dtype=wire_frame_faces[side.value].dtype)
            if cnt == 0:
                return None
            return WireframeCubeMesh(wfqCube,
                              Instances(instance_data, wfqCube.instances.layout, True))
        return None

    def __str__(self):
        return f"Block(id: {self.id}, pos: {self.pos})"

    def serialize(self) -> bytes:
        # Pack the chunkBlockPos (3 floats) and blockID (int) into bytes
        header_data = struct.pack(
            "fffI",
            self.offset_from_chunk.x,
            self.offset_from_chunk.y,
            self.offset_from_chunk.z,
            self.id.value
        )

        # Flatten face_instances numpy arrays into bytes
        face_instances_data = b''.join(
            [faces.tobytes() for faces in self.face_instances]
        )

        # Combine header data and face instances data
        return header_data + face_instances_data

    @classmethod
    def deserialize(cls, binary_data: bytes):
        import numpy as np

        # Unpack 3 floats and 1 unsigned int (the block ID) from the binary data
        x, y, z, blockID = struct.unpack("fffI", binary_data[:16])

        # Create a new instance of Block with the deserialized data
        block = cls(glm.vec3(x, y, z), Block.ID(blockID))

        # Compute the size of the face instances data
        face_instance_size = 6 * 4 * 4 * np.dtype(
            np.float32).itemsize  # 6 faces, 4 vertices per face, 4 floats per vertex

        # Extract face instances data from binary_data
        face_instances_data = binary_data[16:]
        face_instances = []

        # Assuming face_instances_data is contiguous for each face
        for i in range(6):
            start = i * face_instance_size
            end = start + face_instance_size
            face_instance_array = np.frombuffer(face_instances_data[start:end], dtype=np.float32).reshape((4, 4))
            face_instances.append(face_instance_array)

        block.face_instances = face_instances

        # Ensure Block._TextureAtlas is initialized
        if Block._TextureAtlas is None:
            texQuadCube = TexQuadCube(NMM(block.pos), glm.vec2(), glm.vec2())
            Block._TextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))
            Block.vertices = texQuadCube.vertices

        return block

    @staticmethod
    def calculate_serialization_size() -> int:
        # Basic block info size
        basic_info_size = 3 * 4 + 4  # 3 floats (12 bytes) + 1 integer (4 bytes)

        # Face instances data size
        face_size = 4 * 4 * 4  # 4 vertices * 4 floats/vertex * 4 bytes/float
        total_face_size = 6 * face_size  # 6 faces

        return basic_info_size + total_face_size


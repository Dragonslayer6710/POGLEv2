import copy
import struct

from MineClone.Block import *
from MineClone.Block import _QUADS_IN_BLOCK
from MineClone.Block import _QUADS_IN_BLOCK

_CHUNK_WIDTH = 8
_CHUNK_HEIGHT = 8
_CHUNK_SIZE = glm.vec3(_CHUNK_WIDTH, _CHUNK_HEIGHT, _CHUNK_WIDTH)
_CHUNK_SIZE_HALF = _CHUNK_SIZE / 2

_BLOCKS_IN_CHUNK = _CHUNK_WIDTH * _CHUNK_WIDTH * _CHUNK_HEIGHT
_QUADS_IN_CHUNK = _QUADS_IN_BLOCK * _BLOCKS_IN_CHUNK


class Chunk(PhysicalBox):
    class Cardinal(Enum):
        West = 0
        SouthWest = auto()
        South = auto()
        SouthEast = auto()
        East = auto()
        NorthEast = auto()
        North = auto()
        NorthWest = auto()

    block_face_ids: list[np.array] = [None] * _BLOCKS_IN_CHUNK
    block_face_tex_dims: list[np.array] = [None] * _BLOCKS_IN_CHUNK
    block_positions: list[np.array] = [None] * _BLOCKS_IN_CHUNK

    neighbourOffsets: dict[Cardinal, glm.vec2] = {
        Cardinal.West: glm.vec2(-1, 0),
        Cardinal.SouthWest: glm.vec2(-1, -1),
        Cardinal.South: glm.vec2(0, -1),
        Cardinal.SouthEast: glm.vec2(-1, 1),
        Cardinal.East: glm.vec2(1, 0),
        Cardinal.NorthEast: glm.vec2(1, 1),
        Cardinal.North: glm.vec2(0, 1),
        Cardinal.NorthWest: glm.vec2(1, -1),
    }
    solid_blocks: list[int | None] = [None for i in range(_CHUNK_WIDTH ** 2 * _CHUNK_HEIGHT)]

    # World chunk position in world vec2 from position in world
    # (worldPos.x/z = -inf to inf)
    @staticmethod
    def pos_from_world_pos(worldPos: glm.vec3) -> glm.vec2:
        return worldPos.xz // _CHUNK_WIDTH

    @staticmethod
    def block_id_from_chunk_block_pos(x: int | glm.vec3, y: int = None, z: int = None) -> int:
        if type(x) == glm.vec3:
            x, y, z = x
        return int(x * _CHUNK_WIDTH * _CHUNK_HEIGHT + y * _CHUNK_WIDTH + z)

    def __init__(self, chunk_array_offset: glm.vec2, blocks: list[Block] = None):
        from MineClone.World import World
        from MineClone.World import _WORLD_CHUNK_AXIS_LENGTH
        super().__init__()
        self._blocks: list[Block] = []
        self.solid_blocks: list[int | None] = copy.deepcopy(Chunk.solid_blocks)

        self.block_face_ids = copy.deepcopy(Chunk.block_face_ids)
        self.block_face_tex_dims = copy.deepcopy(Chunk.block_face_tex_dims)
        self.block_positions = copy.deepcopy(Chunk.block_positions)

        self.num_solid_blocks = 0

        self.normal_chunk_pos: glm.vec2 = chunk_array_offset
        self.chunk_pos: glm.vec2 = chunk_array_offset - _WORLD_CHUNK_AXIS_LENGTH
        self.chunk_id: int = World.chunk_id_from_chunk_array_offset(self.normal_chunk_pos)
        self.world: World | None = None

        self._neighbourChunkCache: dict[Chunk.Cardinal, Chunk | None] = {}
        self.neighbourPositions: dict[Chunk.Cardinal, glm.vec2] = {k: v + self.normal_chunk_pos for k, v in
                                                                   Chunk.neighbourOffsets.items()}
        self.worldChunkBlockPos: glm.vec3 = glm.vec3(
            self.chunk_pos[0]* _CHUNK_WIDTH - _CHUNK_SIZE_HALF.x,
            0,
            self.chunk_pos[1]* _CHUNK_WIDTH - _CHUNK_SIZE_HALF.z)
        self.bounds = AABB.from_pos_size(self.worldChunkBlockPos + _CHUNK_SIZE_HALF, _CHUNK_SIZE)
        self.octree: SpatialTree = Octree(self.bounds)
        if not blocks:
            for x in range(_CHUNK_WIDTH):
                for y in range(_CHUNK_HEIGHT):
                    for z in range(_CHUNK_WIDTH):
                        block: Block = Block(glm.vec3(x,y,z), Block.ID(random.randrange(0, len(Block.ID))))
                        block.link_chunk(self)
                        self._blocks.append(block)
        else:
            for block in blocks:
                block.link_chunk(self)
                self._blocks.append(block)
        for block in self._blocks:
            self.update_block_in_chunk(block)

    def blocks(self, chunkBlockID) -> Block:
        return self._blocks[chunkBlockID]
    @property
    def not_empty(self) -> bool:
        return self.num_solid_blocks != 0

    def neighbourChunk(self, dir: Cardinal):
        if self.world:
            if dir not in self._neighbourChunkCache.keys():
                self._neighbourChunkCache[dir] = self.world.get_chunk(self.neighbourPositions[dir])
            return self._neighbourChunkCache[dir]
        return None

    def link_world(self, world):
        self.world = world

    def update_block_in_chunk(self, block: Block):
        if block.is_solid:
            if None is self.solid_blocks[block.block_id_in_chunk]:
                self.solid_blocks[block.block_id_in_chunk] = block.block_id_in_chunk
                self.num_solid_blocks += 1
                self.octree.insert(block)
        else:
            if self.solid_blocks[block.block_id_in_chunk] == block.block_id_in_chunk:
                self.num_solid_blocks -= 1
                self.solid_blocks[block.block_id_in_chunk] = None
                self.octree.remove(block)
        if self.solid_blocks[block.block_id_in_chunk]:
            self.set_block_instance(block)
        else:
            self.set_block_instance(block, True)
        for side in Block.Side:
            adjBlock = block.adjBlock(side)
            if adjBlock:
                if not adjBlock.is_transparent:
                    opposite = Block.oppositeSide[side]
                    visible = adjBlock.face_visible(opposite)
                    if visible and not block.is_transparent:
                        adjBlock.hide_face(opposite)
                    elif not visible and block.is_transparent:
                        adjBlock.reveal_face(opposite)
                    else:
                        continue
                    self.set_block_instance(adjBlock)

        if self.world:
            self.world.update_chunk_in_world(self)

    def set_block(self, chunkBlockID: int, blockID: Block.ID):
        block = self._blocks[chunkBlockID]
        if not block.initialised:
            block.link_chunk(self, chunkBlockID, blockID)
            self.octree.insert(block)
        else:
            self.set_block_instance(block)
        if self.solid_blocks[chunkBlockID] and not block.is_solid:
            self.num_solid_blocks -= 1
            self.solid_blocks[chunkBlockID] = None
        elif not self.solid_blocks[chunkBlockID] and block.is_solid:
            self.num_solid_blocks += 1
            self.solid_blocks[chunkBlockID] = block

    def query_aabb_blocks(self, boxRange: AABB, hitBlocks: set[Block] = None) -> set[Block]:
        return self.octree.query_aabb(boxRange, hitBlocks)

    def query_segment_blocks(self, ray: Ray, hitBlocks: set[Block] = None) -> set[Block]:
        return self.octree.query_segment(ray, hitBlocks)

    def get_world_pos(self, blockPos: glm.vec3) -> glm.vec3:
        return self.worldChunkBlockPos + blockPos  # Chunk Pos in world plus chunk rel coordinate

    def _get_block(self, x: int, y: int, z: int) -> Block:
        return self._blocks[Chunk.block_id_from_chunk_block_pos(x, y, z)]

    def get_block(self, chunkBlockPos: glm.vec3 = None) -> Block | None:
        chunkBlockX, chunkBlockY, chunkBlockZ = [int(i) for i in chunkBlockPos]

        if chunkBlockY not in range(_CHUNK_HEIGHT):
            return None
        xOutOfRange, zOutOfRange = chunkBlockX not in range(_CHUNK_WIDTH), chunkBlockZ not in range(_CHUNK_WIDTH)
        if xOutOfRange or zOutOfRange:
            if zOutOfRange:
                # adding chunkwidth brings negative to width-1 or width to 2x width, in both cases % chunkwidth after
                # will get you to an in bounds result (width-1 for the former and 0 for the latter
                chunkBlockZ = (chunkBlockZ + _CHUNK_WIDTH) % _CHUNK_WIDTH
            if xOutOfRange:
                chunkBlockX = (chunkBlockX + _CHUNK_WIDTH) % _CHUNK_WIDTH
                if chunkBlockX < 0:
                    if not zOutOfRange:
                        cardinalDir: Chunk.Cardinal = Chunk.Cardinal.West
                    else:
                        if chunkBlockX < 0:
                            cardinalDir: Chunk.Cardinal = Chunk.Cardinal.SouthWest
                        else:
                            cardinalDir: Chunk.Cardinal = Chunk.Cardinal.NorthWest
                else:
                    if not zOutOfRange:
                        cardinalDir: Chunk.Cardinal = Chunk.Cardinal.East
                    else:
                        if chunkBlockX < 0:
                            cardinalDir: Chunk.Cardinal = Chunk.Cardinal.SouthEast
                        else:
                            cardinalDir: Chunk.Cardinal = Chunk.Cardinal.NorthEast
            elif zOutOfRange:
                if chunkBlockZ < 0:
                    cardinalDir: Chunk.Cardinal = Chunk.Cardinal.South
                else:
                    cardinalDir: Chunk.Cardinal = Chunk.Cardinal.North
            else:
                raise Exception("If x or z out of range, this should not be possible to reach")
            chunk: Chunk = self.neighbourChunk(cardinalDir)
            if chunk:
                return chunk.get_block(glm.vec3(chunkBlockX, chunkBlockY, chunkBlockZ))
            return None
        else:
            return self._get_block(chunkBlockX, chunkBlockY, chunkBlockZ)

    def is_pos_in_chunk(self, worldPos: glm.vec3) -> bool:
        chunkPosX, chunkPosZ = [int(i) for i in (worldPos - self.worldChunkBlockPos).xz]
        return chunkPosX in range(_CHUNK_WIDTH) and chunkPosZ in range(_CHUNK_WIDTH)

    def get_block_from_chunk_pos(self, chunkX: int, chunkY: int, chunkZ: int):
        return self._get_block(chunkX, chunkY, chunkZ)

    def get_block_from_world_block_pos(self, worldBlockPos: glm.vec3) -> Block:
        chunkY = int(worldBlockPos.y)
        chunkX, chunkZ = [int(i) for i in worldBlockPos.xz % _CHUNK_WIDTH]
        return self._get_block(chunkX, chunkY, chunkZ)

    def update(self) -> bool:
        updated = False
        for chunkBlockID in filter(None, self.solid_blocks):
            block: Block = self.blocks(chunkBlockID)
            if block.update_side_visibility():
                self.set_block_instance(block)
                updated = True
        return updated

    def set_block_instance(self, block: Block, clear=False):
        (
            self.block_face_ids[block.block_id_in_chunk],
            self.block_face_tex_dims[block.block_id_in_chunk],
            self.block_positions[block.block_id_in_chunk]
        ) = block.get_instance_data() if not clear else (None, None, None)

    def get_block_and_face_instance_data(self) -> list[np.array] | list[None]:
        block_face_ids = list(filter((None).__ne__, self.block_face_ids))
        block_face_tex_dims = list(filter((None).__ne__, self.block_face_tex_dims))
        block_positions = list(filter((None).__ne__, self.block_positions))
        if not block_positions:
            return [None, None, None]
        return [
            np.concatenate(block_face_ids),
            np.concatenate(block_face_tex_dims),
            np.concatenate(block_positions)
        ]

    def get_chunks_to_render(self, renderedChunks: list[list], chunksLeftFromHere: int,
                             listPos: glm.vec2 = None, visitedChunks=None):
        renderDistanceFromHere = (chunksLeftFromHere - 1) // 2
        if visitedChunks is None:
            visitedChunks = []
            listPos = glm.vec2(renderDistanceFromHere, renderDistanceFromHere)
            Chunk._num_chunks_to_visit = len(renderedChunks)
            Chunk._num_chunks_to_visit *= Chunk._num_chunks_to_visit
            Chunk._currentDepth = 0
            Chunk._chunksToRender = chunksLeftFromHere - 1
        else:
            Chunk._currentDepth += 1
        minPos: glm.vec2 = listPos - renderDistanceFromHere
        maxPos: glm.vec2 = listPos + renderDistanceFromHere
        limits = [minPos, maxPos]
        for i in range(2):
            for limit in limits:
                if i:
                    limit[0] = min(limit[0], Chunk._chunksToRender)
                    limit[1] = min(limit[1], Chunk._chunksToRender)
                else:
                    limit[0] = max(limit[0], 0)
                    limit[1] = max(limit[1], 0)

        visitedChunks.append(self.chunk_id)
        renderedChunks[int(listPos[0])][int(listPos[1])] = self
        self._num_chunks_to_visit -= 1
        if self._num_chunks_to_visit:
            if chunksLeftFromHere:
                for cardinalDir in Chunk.Cardinal:
                    neighbour: Chunk | None = self.neighbourChunk(cardinalDir)
                    if neighbour:
                        if neighbour.not_empty:
                            if neighbour.chunk_id not in visitedChunks:
                                shiftedPos = listPos + self.neighbourOffsets[cardinalDir]
                                if minPos[0] <= shiftedPos[0] <= maxPos[0] and minPos[1] <= shiftedPos[1] <= maxPos[1]:
                                    neighbour.get_chunks_to_render(renderedChunks, chunksLeftFromHere, shiftedPos, visitedChunks)

        if Chunk._currentDepth:
            Chunk._currentDepth -= 1
        else:
            del Chunk._currentDepth
            del Chunk._num_chunks_to_visit

    def serialize(self) -> bytes:
        serialized_data = bytearray()

        # Serialize size as three floats
        serialized_data.extend(struct.pack("fff", self.size.x, self.size.y, self.size.z))

        # Serialize other attributes
        for block in self._blocks:
            serialized_data.extend(block.serialize())

        return bytes(serialized_data)

    @classmethod
    def deserialize(cls, binary_data: bytes):
        import struct
        import glm

        # Offset to track position in binary_data
        offset = 0

        # Deserialize size (as three floats)
        if len(binary_data) < 12:
            raise ValueError("Insufficient data to read size")
        size_x, size_y, size_z = struct.unpack("fff", binary_data[offset:offset + 12])
        size = glm.vec3(size_x, size_y, size_z)
        offset += 12

        # Define block size (this should match the size of serialized block data)
        block_size = Block.calculate_serialization_size()  # Assuming a method to get the serialized size of a block

        # Deserialize blocks
        blocks = []
        while offset + block_size <= len(binary_data):
            block_data = binary_data[offset:offset + block_size]
            blocks.append(Block.deserialize(block_data))
            offset += block_size

        # Check if there is extra data after the last block
        if offset != len(binary_data):
            raise ValueError("Extra data found after block serialization")

        chunk = cls(size)
        chunk._blocks = blocks
        return chunk



import copy
import struct

from MineClone.Block import *
from MineClone.Block import _QUADS_IN_BLOCK

_CHUNK_WIDTH = 4
_CHUNK_HEIGHT = 4
_CHUNK_SIZE = glm.vec3(_CHUNK_WIDTH, _CHUNK_HEIGHT, _CHUNK_WIDTH)
_CHUNK_SIZE_HALF = _CHUNK_SIZE / 2

_BLOCKS_IN_CHUNK = _CHUNK_WIDTH * _CHUNK_WIDTH * _CHUNK_HEIGHT
_QUADS_IN_CHUNK = _QUADS_IN_BLOCK * _BLOCKS_IN_CHUNK


class Chunk(PhysicalBox):
    class ID(Enum):
        Null = 0
        Valid = auto()

    class Cardinal(Enum):
        West = 0
        SouthWest = auto()
        South = auto()
        SouthEast = auto()
        East = auto()
        NorthEast = auto()
        North = auto()
        NorthWest = auto()

    chunkID: ID = ID.Null

    blocks: list[Block] = [Block(glm.vec3(x, y, z)) for x in range(_CHUNK_WIDTH)
                                                       for y in range(_CHUNK_HEIGHT)
                                                       for z in range(_CHUNK_WIDTH)]

    block_instances: list[np.ndarray] = [None] * _BLOCKS_IN_CHUNK

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

    def _get_chunk_block_id(self, x: int, y: int, z: int) -> int:
        return x * _CHUNK_WIDTH * _CHUNK_HEIGHT + y * _CHUNK_WIDTH + z

    def __init__(self, worldChunkPos: glm.vec2, blocks: list[Block] = None):
        super().__init__()
        if worldChunkPos is not None:
            if blocks == None:
                self.blocks = copy.deepcopy(Chunk.blocks)
            else:
                self.blocks = blocks
            self.block_instances = copy.deepcopy(Chunk.block_instances)
            self.not_null_blocks = copy.deepcopy(Chunk.block_instances)
            self.num_not_null_blocks = 0
            self.chunkID = Chunk.ID.Valid
            self.worldChunkPos: glm.vec2 = worldChunkPos
            self._neighbourPos: dict[Chunk.Cardinal, glm.vec2] = {k: v + self.worldChunkPos for k, v in
                                                                  Chunk.neighbourOffsets.items()}
            self.worldChunkBlockPos: glm.vec3 = glm.vec3(worldChunkPos[0], 0, worldChunkPos[1]) * _CHUNK_WIDTH
            self.bounds = AABB.from_pos_size(self.worldChunkBlockPos + _CHUNK_SIZE_HALF, _CHUNK_SIZE + glm.vec3(1))

        self.is_chunk: bool = self.chunkID == Chunk.ID.Valid

    @property
    def not_empty(self) -> bool:
        return self.num_not_null_blocks != 0

    def init(self, world, worldChunkID: int):
        from MineClone.World import World
        self.world: World = world
        self.worldChunkID: int = worldChunkID
        self.is_chunk = self.chunkID == Chunk.ID.Valid
        self.neighbourChunks: dict[Chunk.Cardinal, Chunk] = {k: self.world.get_chunk_from_world_chunk_pos(v) for k, v in
                                                             self._neighbourPos.items()}
        self.octree: SpatialTree = Octree(self.bounds)
        for x in range(_CHUNK_WIDTH):
            for y in range(_CHUNK_HEIGHT):
                for z in range(_CHUNK_WIDTH):
                    blockID = Block.ID(random.randrange(0, len(Block.ID)))
                    self.set_block(self._get_chunk_block_id(x, y, z), blockID)

    def set_block(self, chunkBlockID: int, blockID: Block.ID):
        block = self.blocks[chunkBlockID]
        if not block.initialised:
            block.init(self, chunkBlockID, blockID)
            self.octree.insert(block)
        else:
            self.set_block_instance(block)
        if self.not_null_blocks[chunkBlockID] and not block.is_block:
            self.num_not_null_blocks -= 1
            self.not_null_blocks[chunkBlockID] = None
        elif not self.not_null_blocks[chunkBlockID] and block.is_block:
            self.num_not_null_blocks += 1
            self.not_null_blocks[chunkBlockID] = block

    def query_aabb_blocks(self, boxRange: AABB, hitBlocks: set[Block] = None) -> set[Block]:
        return self.octree.query_aabb(boxRange, hitBlocks)

    def query_segment_blocks(self, ray: Ray, hitBlocks: set[Block] = None) -> set[Block]:
        return self.octree.query_segment(ray, hitBlocks)

    def get_world_pos(self, blockPos: glm.vec3) -> glm.vec3:
        return self.worldChunkBlockPos + blockPos  # Chunk Pos in world plus chunk rel coordinate

    def _get_block(self, x: int, y: int, z: int) -> Block:
        return self.blocks[self._get_chunk_block_id(x, y, z)]

    def get_block(self, worldBlockPos: glm.vec3, chunkBlockPos: glm.vec3 = None) -> Block:
        if chunkBlockPos is None:
            chunkBlockPos = worldBlockPos - self.worldChunkBlockPos
        chunkBlockX, chunkBlockY, chunkBlockZ = [int(i) for i in chunkBlockPos]

        if chunkBlockY not in range(_CHUNK_HEIGHT):
            return None
        xOutOfRange, zOutOfRange = chunkBlockX not in range(_CHUNK_WIDTH), chunkBlockZ not in range(_CHUNK_WIDTH)
        if xOutOfRange or zOutOfRange:
            if xOutOfRange:
                if chunkBlockX < 0:
                    cardinalDir: Chunk.Cardinal = Chunk.Cardinal.West
                else:
                    cardinalDir: Chunk.Cardinal = Chunk.Cardinal.East
            if zOutOfRange:
                if chunkBlockZ < 0:
                    cardinalDir: Chunk.Cardinal = Chunk.Cardinal.South
                else:
                    cardinalDir: Chunk.Cardinal = Chunk.Cardinal.North
            chunk: Chunk = self.neighbourChunks[cardinalDir]
            if chunk:
                return chunk.get_block(worldBlockPos)
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
        for block in filter(None, self.not_null_blocks):
            if block.update_side_visibility():
                self.block_instances[block.chunkBlockID] = block.get_face_instance_data()
                updated = True
        return updated

    def set_block_instance(self, block: Block):
        self.block_instances[block.chunkBlockID] = block.get_face_instance_data()
        self.world.flag_chunk_update(self)

    def get_block_face_instance_data(self):
        block_instances = list(filter((None).__ne__, self.block_instances))
        if not block_instances:
            return None
        return np.concatenate(block_instances, dtype=block_instances[0].dtype)

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

        visitedChunks.append(self.worldChunkID)
        renderedChunks[int(listPos[0])][int(listPos[1])] = self
        self._num_chunks_to_visit -= 1
        if self._num_chunks_to_visit:
            if chunksLeftFromHere:
                for cardinalDir, neighbour in self.neighbourChunks.items():
                    if neighbour.is_chunk:
                        if neighbour.worldChunkID not in visitedChunks:
                            shiftedPos = listPos + self.neighbourOffsets[cardinalDir]
                            if minPos[0] <= shiftedPos[0] <= maxPos[0] and minPos[1] <= shiftedPos[1] <= maxPos[1]:
                                neighbour.get_chunks_to_render(renderedChunks, chunksLeftFromHere, shiftedPos, visitedChunks)

        if Chunk._currentDepth:
            Chunk._currentDepth -= 1
        else:
            del Chunk._currentDepth
            del Chunk._num_chunks_to_visit

    def serialize(self) -> bytes:
        # Serialize all blocks into a single byte string
        serialized_blocks = b"".join([block.serialize() for block in self.blocks])

        # Pack the number of blocks (unsigned int), worldChunkPos (two floats), and serialized blocks
        return (
                struct.pack("I", len(self.blocks)) +  # Number of blocks
                struct.pack("ff", *self.worldChunkPos) +  # World chunk position (two floats)
                serialized_blocks  # Serialized blocks data
        )

    @classmethod
    def deserialize(cls, binary_data: bytes):
        # Ensure there is enough data for the initial unpack of the number of blocks
        if len(binary_data) < 8:  # 4 bytes for the number of blocks + 4 bytes for two floats
            raise ValueError("Insufficient data to read the number of blocks and worldChunkPos")

        # Unpack the number of blocks (unsigned int)
        num_blocks = struct.unpack("I", binary_data[:4])[0]

        # Unpack the worldChunkPos (two floats)
        world_chunk_pos = glm.vec2(struct.unpack("ff", binary_data[4:12]))

        blocks = []
        offset = 12  # Initial offset after the number of blocks and worldChunkPos

        block_size = struct.calcsize("fffi")  # Assuming block size is 16 bytes (4 bytes for each float and int)

        for _ in range(num_blocks):
            # Ensure there is enough data to read the next block
            if offset + block_size > len(binary_data):
                raise ValueError(f"Insufficient data to read block {_ + 1}")

            # Extract each block's binary data and deserialize it
            block_data = binary_data[offset:offset + block_size]
            blocks.append(Block.deserialize(block_data))
            offset += block_size

        # Return an instance of the class initialized with the deserialized data
        return cls(world_chunk_pos, blocks)

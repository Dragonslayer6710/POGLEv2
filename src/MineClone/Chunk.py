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
    class Cardinal(Enum):
        West = 0
        SouthWest = auto()
        South = auto()
        SouthEast = auto()
        East = auto()
        NorthEast = auto()
        North = auto()
        NorthWest = auto()

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

        self.block_instances = copy.deepcopy(Chunk.block_instances)
        self.num_solid_blocks = 0

        self.normal_chunk_pos: glm.vec2 = chunk_array_offset
        self.chunk_pos: glm.vec2 = chunk_array_offset - _WORLD_CHUNK_AXIS_LENGTH
        self.chunk_id: int = World.chunk_id_from_chunk_array_offset(self.normal_chunk_pos)
        self.world: World | None = None

        self._neighbourChunkCache: dict[Chunk.Cardinal, Chunk | None] = {}
        self.neighbourPositions: dict[Chunk.Cardinal, glm.vec2] = {k: v + self.normal_chunk_pos for k, v in
                                                                   Chunk.neighbourOffsets.items()}
        self.worldChunkBlockPos: glm.vec3 = glm.vec3(self.chunk_pos[0], 0, self.chunk_pos[1]) * _CHUNK_WIDTH
        self.bounds = AABB.from_pos_size(self.worldChunkBlockPos + _CHUNK_SIZE_HALF, _CHUNK_SIZE + glm.vec3(1))
        self.octree: SpatialTree = Octree(self.bounds)
        if not blocks:
            for x in range(_CHUNK_WIDTH):
                for y in range(_CHUNK_HEIGHT):
                    for z in range(_CHUNK_WIDTH):
                        block: Block = Block(glm.vec3(x,y,z), Block.ID(random.randrange(1, len(Block.ID))))
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
            if None is self.solid_blocks[block.id_from_chunk]:
                self.solid_blocks[block.id_from_chunk] = block.id_from_chunk
                self.num_solid_blocks += 1
                self.octree.insert(block)
        else:
            if self.solid_blocks[block.id_from_chunk] == block.id_from_chunk:
                self.num_solid_blocks -= 1
                self.solid_blocks[block.id_from_chunk] = None
                self.octree.remove(block)
        if self.solid_blocks[block.id_from_chunk]:
            self.block_instances[block.id_from_chunk] = block.get_face_instance_data()
        else:
            self.block_instances[block.id_from_chunk] = None
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
                    self.block_instances[adjBlock.id_from_chunk] = adjBlock.get_face_instance_data()

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
                self.block_instances[chunkBlockID] = block.get_face_instance_data()
                updated = True
        return updated

    def set_block_instance(self, block: Block):
        self.block_instances[block.id_from_chunk] = block.get_face_instance_data()

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
        # Serialize all blocks into a single byte string
        serialized_blocks = b"".join([block.serialize() for block in self._blocks])

        # Pack the number of blocks (unsigned int), worldChunkPos (two floats), and serialized blocks
        return (
                struct.pack("I", len(self._blocks)) +  # Number of blocks
                struct.pack("ff", *self.normal_chunk_pos) +  # World chunk position (two floats)
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

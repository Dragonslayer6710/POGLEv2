import copy

from MineClone.Block import *
from MineClone.Block import _QUADS_IN_BLOCK

_CHUNK_WIDTH = 8
_CHUNK_WIDTH_RANGE = range(_CHUNK_WIDTH)
_CHUNK_HEIGHT = 8
_CHUNK_HEIGHT_RANGE = range(_CHUNK_HEIGHT)

_BLOCKS_IN_CHUNK = _CHUNK_WIDTH * _CHUNK_WIDTH * _CHUNK_HEIGHT
_QUADS_IN_CHUNK = _QUADS_IN_BLOCK * _BLOCKS_IN_CHUNK


class Chunk:
    class ID(Enum):
        Null = 0
        Valid = auto()

    class Side(Enum):
        East = 0
        South = auto()
        West = auto()
        North = auto()

    chunkID: ID = ID.Null

    blocks: list[list[list[Block]]] = [
        [
            [
                Block(glm.vec3(x, y, z)) for z in _CHUNK_WIDTH_RANGE
            ] for y in _CHUNK_HEIGHT_RANGE
        ] for x in _CHUNK_WIDTH_RANGE
    ]

    block_instances: list[np.ndarray] = [None] * _CHUNK_WIDTH * _CHUNK_HEIGHT * _CHUNK_WIDTH

    adjChunkOffsets: dict[Side, glm.vec2] = {
        Side.East: glm.vec2(-1, 0),
        Side.South: glm.vec2(0,-1),
        Side.West: glm.vec2(1, 0),
        Side.North: glm.vec2(0, 1),
    }

    def __init__(self, worldChunkPos: glm.vec2):
        self.blocks = copy.deepcopy(Chunk.blocks)
        if None != worldChunkPos:
            self.chunkID = Chunk.ID.Valid
            self.worldChunkPos: glm.vec2 = worldChunkPos
            self.worldBlockPos: glm.vec3 = glm.vec3(worldChunkPos[0], 0, worldChunkPos[1]) * _CHUNK_WIDTH
            self.adjChunks: dict[Chunk.Side, Chunk] = None

    def init(self):
        for blockPlane in self.blocks:
            for blockRow in blockPlane:
                for block in blockRow:
                    block.init()
                    self.block_instances[block.chunkBlockID] = block.get_instance_data()

    def set(self, world, worldChunkID: int):
        from MineClone.World import World
        self.world: World = world
        self.worldChunkID: int = worldChunkID
        self.adjChunks = {
            side: self.world.get_chunk(self.worldChunkPos + offset) for side, offset in self.adjChunkOffsets.items()
        }

        for x in _CHUNK_WIDTH_RANGE:
            for y in _CHUNK_HEIGHT_RANGE:
                for z in _CHUNK_WIDTH_RANGE:
                    block = self.blocks[x][y][z]
                    chunkBlockID = x * _CHUNK_WIDTH * _CHUNK_HEIGHT + y * _CHUNK_WIDTH + z
                    blockID = Block.ID(random.randrange(0, len(Block.ID)))
                    block.set(self, chunkBlockID, blockID)
    def get_world_pos(self, blockPos: glm.vec3) -> glm.vec3:
        return self.worldBlockPos + blockPos  # Chunk Pos in world plus chunk rel coordinate

    def _get_block(self, x: int, y: int, z: int) -> Block:
        return self.blocks[x][y][z]

    def get_block(self, worldBlockPos: glm.vec3, chunkBlockPos: glm.vec3 = None) -> Block:
        if chunkBlockPos == None:
            chunkBlockPos = worldBlockPos - self.worldBlockPos
        chunkBlockX, chunkBlockY, chunkBlockZ = [int(i) for i in chunkBlockPos]
        worldBlockX, worldBlockY, worldBlockZ = [int(i) for i in worldBlockPos]

        if chunkBlockY not in _CHUNK_HEIGHT_RANGE:
            return BLOCK_NULL
        xOutOfRange, zOutOfRange = chunkBlockX not in _CHUNK_WIDTH_RANGE, chunkBlockZ not in _CHUNK_WIDTH_RANGE
        if xOutOfRange or zOutOfRange:
            if xOutOfRange:
                if chunkBlockX < 0:
                    side: Chunk.Side = Chunk.Side.East
                else:
                    side: Chunk.Side = Chunk.Side.West
            if zOutOfRange:
                if chunkBlockZ < 0:
                    side: Chunk.Side = Chunk.Side.South
                else:
                    side: Chunk.Side = Chunk.Side.North
            chunk: Chunk = self.adjChunks[side]
            if CHUNK_NULL == chunk:
                return BLOCK_NULL
            else:
                return chunk.get_block(worldBlockPos)
        else:
            return self._get_block(chunkBlockX, chunkBlockY, chunkBlockZ)

    def update(self) -> bool:
        updated = False
        for blockPlane in self.blocks:
            for blockRow in blockPlane:
                for block in blockRow:
                    if block.update_side_visibility():
                        self.block_instances[block.chunkBlockID] = block.get_instance_data()
                        updated = True
        return updated

    def get_instance_data(self):
        block_instances = list(filter((None).__ne__, self.block_instances))
        if len(block_instances) == 0:
            return None
        return np.concatenate(block_instances, dtype=block_instances[0].dtype)


CHUNK_NULL = Chunk(None)

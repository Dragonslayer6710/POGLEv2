import copy

from MineClone.Block import *
from MineClone.Block import _QUADS_IN_BLOCK

_CHUNK_WIDTH = 4
_CHUNK_WIDTH_RANGE = range(_CHUNK_WIDTH)
_CHUNK_HEIGHT = 4
_CHUNK_HEIGHT_RANGE = range(_CHUNK_HEIGHT)

_BLOCKS_IN_CHUNK = _CHUNK_WIDTH * _CHUNK_WIDTH * _CHUNK_HEIGHT
_QUADS_IN_CHUNK = _QUADS_IN_BLOCK * _BLOCKS_IN_CHUNK


class Chunk:
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

    blocks: list[list[list[Block]]] = [
        [
            [
                Block(glm.vec3(x, y, z)) for z in _CHUNK_WIDTH_RANGE
            ] for y in _CHUNK_HEIGHT_RANGE
        ] for x in _CHUNK_WIDTH_RANGE
    ]

    block_instances: list[np.ndarray] = [None] * _CHUNK_WIDTH * _CHUNK_HEIGHT * _CHUNK_WIDTH

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

    def __init__(self, worldChunkPos: glm.vec2):
        if None != worldChunkPos:
            self.blocks = copy.deepcopy(Chunk.blocks)
            self.block_instances = copy.deepcopy(Chunk.block_instances)
            self.chunkID = Chunk.ID.Valid
            self.worldChunkPos: glm.vec2 = worldChunkPos
            self._neighbourPos: dict[Chunk.Cardinal, glm.vec2] = {k: v + self.worldChunkPos for k, v in
                                                                  Chunk.neighbourOffsets.items()}
            self.worldChunkBlockPos: glm.vec3 = glm.vec3(worldChunkPos[0], 0, worldChunkPos[1]) * _CHUNK_WIDTH
        self.is_chunk: bool = self.chunkID == Chunk.ID.Valid

    def init(self, world, worldChunkID: int):
        from MineClone.World import World
        self.world: World = world
        self.worldChunkID: int = worldChunkID
        self.is_chunk = self.chunkID == Chunk.ID.Valid
        self.neighbourChunks: dict[Chunk.Cardinal, Chunk] = {k: self.world.get_chunk_from_world_chunk_pos(v) for k, v in
                                                             self._neighbourPos.items()}
        for x in _CHUNK_WIDTH_RANGE:
            for y in _CHUNK_HEIGHT_RANGE:
                for z in _CHUNK_WIDTH_RANGE:
                    block = self.blocks[x][y][z]
                    chunkBlockID = x * _CHUNK_WIDTH * _CHUNK_HEIGHT + y * _CHUNK_WIDTH + z
                    blockID = Block.ID(random.randrange(0, len(Block.ID)))
                    block.init(self, chunkBlockID, blockID)
                    self.block_instances[block.chunkBlockID] = block.get_instance_data()

    def get_world_pos(self, blockPos: glm.vec3) -> glm.vec3:
        return self.worldChunkBlockPos + blockPos  # Chunk Pos in world plus chunk rel coordinate

    def _get_block(self, x: int, y: int, z: int) -> Block:
        return self.blocks[x][y][z]

    def get_block(self, worldBlockPos: glm.vec3, chunkBlockPos: glm.vec3 = None) -> Block:
        if chunkBlockPos == None:
            chunkBlockPos = worldBlockPos - self.worldChunkBlockPos
        chunkBlockX, chunkBlockY, chunkBlockZ = [int(i) for i in chunkBlockPos]

        if chunkBlockY not in _CHUNK_HEIGHT_RANGE:
            return BLOCK_NULL
        xOutOfRange, zOutOfRange = chunkBlockX not in _CHUNK_WIDTH_RANGE, chunkBlockZ not in _CHUNK_WIDTH_RANGE
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
            if CHUNK_NULL == chunk:
                return BLOCK_NULL
            else:
                return chunk.get_block(worldBlockPos)
        else:
            return self._get_block(chunkBlockX, chunkBlockY, chunkBlockZ)

    def is_pos_in_chunk(self, worldPos: glm.vec3) -> bool:
        chunkPosX, chunkPosZ = [int(i) for i in (worldPos - self.worldChunkBlockPos).xz]
        if chunkPosX not in _CHUNK_WIDTH_RANGE:
            return False
        if chunkPosZ not in _CHUNK_WIDTH_RANGE:
            return False
        return True

    def get_block_from_chunk_pos(self, chunkX: int, chunkY: int, chunkZ: int):
        return self._get_block(chunkX, chunkY, chunkZ)

    def get_block_from_world_block_pos(self, worldBlockPos: glm.vec3) -> Block:
        chunkY = int(worldBlockPos.y)
        chunkX, chunkZ = [int(i) for i in worldBlockPos.xz % _CHUNK_WIDTH]
        return self._get_block(chunkX, chunkY, chunkZ)

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


CHUNK_NULL = Chunk(None)

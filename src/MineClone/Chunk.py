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
        Side.South: glm.vec2(0, -1),
        Side.West: glm.vec2(1, 0),
        Side.North: glm.vec2(0, 1),
    }

    diagChunkOffsets: list[dict[Side, glm.vec2]] = [
        {Side.East: adjChunkOffsets[Side.East], Side.West: adjChunkOffsets[Side.West]},
        {Side.South: adjChunkOffsets[Side.South], Side.North: adjChunkOffsets[Side.North]}
    ]
    _diagSet = False

    def __init__(self, worldChunkPos: glm.vec2):
        if not Chunk._diagSet:
            Chunk.diagChunkOffsets: dict[Chunk.Side, dict[Chunk.Side, glm.vec2]] = {
                sideZ: {sideX: offsetX + offsetZ for sideX, offsetX in Chunk.diagChunkOffsets[0].items()} for
                sideZ, offsetZ
                in Chunk.diagChunkOffsets[1].items()}
            Chunk._diagSet = True

        if None != worldChunkPos:
            self.blocks = copy.deepcopy(Chunk.blocks)
            self.block_instances = copy.deepcopy(Chunk.block_instances)
            self.chunkID = Chunk.ID.Valid
            self.worldChunkPos: glm.vec2 = worldChunkPos
            self.worldChunkBlockPos: glm.vec3 = glm.vec3(worldChunkPos[0], 0, worldChunkPos[1]) * _CHUNK_WIDTH
        self.is_chunk: bool = self.chunkID == Chunk.ID.Valid

    def init(self, world, worldChunkID: int):
        from MineClone.World import World
        self.world: World = world
        self.adjChunks: dict[Chunk.Side, Chunk] = {
            side: self.world.get_chunk_from_world_chunk_pos(self.worldChunkPos + offset) for side, offset in
            self.adjChunkOffsets.items()
        }
        self.diagChunks: dict[Chunk.Side, dict[Chunk.Side, Chunk]] = {
            sideZ: {sideX: self.world.get_chunk_from_world_chunk_pos(self.worldChunkPos + offset) for sideX, offset in
                    sideXDict.items()} for sideZ, sideXDict in self.diagChunkOffsets.items()}
        self.worldChunkID: int = worldChunkID
        self.is_chunk = self.chunkID == Chunk.ID.Valid

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

    def get_chunks_to_render(self, renderedChunks: list[list], renderDistanceFromHere: int, listPos: glm.vec2 = None,
                             visitedChunks: list = []):
        visitedChunks.append(self.worldChunkID)
        halfPoint = renderDistanceFromHere // 2
        if not listPos:
            listPos = glm.vec2(halfPoint, halfPoint)
        minPos: glm.vec2 = listPos - halfPoint
        maxPos: glm.vec2 = listPos + halfPoint
        renderedChunks[int(listPos[0])][int(listPos[1])] = self
        if renderDistanceFromHere:
            for side, adjChunk in self.adjChunks.items():
                if adjChunk.is_chunk:
                    if adjChunk.worldChunkID not in visitedChunks:
                        shiftedPos = listPos + self.adjChunkOffsets[side]
                        if minPos[0] <= shiftedPos[0] <= maxPos[0] and minPos[1] <= shiftedPos[1] <= maxPos[1]:
                            adjChunk.get_chunks_to_render(
                                renderedChunks,
                                renderDistanceFromHere - 1,
                                shiftedPos,
                                visitedChunks
                            )
                if side in self.diagChunks.keys():
                    for sideX, diagChunk in self.diagChunks[side].items():
                        if diagChunk.is_chunk:
                            if diagChunk.worldChunkID not in visitedChunks:
                                shiftedPos = listPos + self.diagChunkOffsets[side][sideX]
                                if minPos[0] <= shiftedPos[0] <= maxPos[0] and minPos[1] <= shiftedPos[1] <= maxPos[1]:
                                    diagChunk.get_chunks_to_render(
                                        renderedChunks,
                                        renderDistanceFromHere - 1,
                                        shiftedPos,
                                        visitedChunks
                                    )
                                else:
                                    print()
        else:
            print()


CHUNK_NULL = Chunk(None)

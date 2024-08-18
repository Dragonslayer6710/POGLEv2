import copy
import os.path

from MineClone.Chunk import *
from MineClone.Chunk import _BLOCKS_IN_CHUNK, _QUADS_IN_CHUNK, _CHUNK_HEIGHT, _CHUNK_WIDTH, _CHUNK_SIZE

_WORLD_CHUNK_AXIS_LENGTH = 4
_WORLD_CHUNK_RANGE = range(-_WORLD_CHUNK_AXIS_LENGTH, _WORLD_CHUNK_AXIS_LENGTH+1)
_CHUNKS_IN_ROW = len(_WORLD_CHUNK_RANGE)

_BLOCKS_IN_ROW = _CHUNKS_IN_ROW * _CHUNK_WIDTH
_WORLD_SIZE = glm.vec3(_BLOCKS_IN_ROW, _CHUNK_HEIGHT, _BLOCKS_IN_ROW)
_WORLD_SIZE_HALF = _WORLD_SIZE / 2

_WORLD_MID_POINT = glm.vec3(0, _WORLD_SIZE_HALF.y, 0)

_CHUNKS_IN_WORLD = _CHUNKS_IN_ROW * _CHUNKS_IN_ROW

_BLOCKS_IN_WORLD = _BLOCKS_IN_CHUNK * _CHUNKS_IN_WORLD
_QUADS_IN_WORLD = _QUADS_IN_CHUNK * _CHUNKS_IN_WORLD

class World(PhysicalBox):
    chunks: list[list[Chunk]] = [[Chunk(glm.vec2(x, z)) for z in _WORLD_CHUNK_RANGE] for x in _WORLD_CHUNK_RANGE]
    chunk_instances: list[np.ndarray] = [None] * len(_WORLD_CHUNK_RANGE) * len(_WORLD_CHUNK_RANGE)
    def __init__(self):
        self.bounds = AABB.from_pos_size(_WORLD_MID_POINT, _WORLD_SIZE)
        self.chunks = copy.deepcopy(World.chunks)
        self.worldWidth = len(self.chunks)

        self.quadtree: QuadTree = QuadTree.XZ(self.bounds, _CHUNK_SIZE)
        for chunkX in range(self.worldWidth):
            for chunkZ in range(self.worldWidth):
                chunk = self.chunks[chunkX][chunkZ]
                worldChunkID = chunkX * self.worldWidth + chunkZ
                chunk.init(self, worldChunkID)
                self.quadtree.insert(chunk)
                self.chunk_instances[chunk.worldChunkID] = chunk.get_instance_data()
    def query_aabb_chunks(self, boxRange: AABB) -> set[Chunk]:
        return self.quadtree.query_aabb(boxRange)

    def query_aabb_blocks(self, boxRange: AABB)-> set[Block]:
        hitBlocks: set[Block] = set()
        for hitChunk in self.query_aabb_chunks(boxRange):
            hitChunk.query_aabb_blocks(boxRange, hitBlocks)
        return hitBlocks


    def _get_chunk(self, x: int, z: int) -> Chunk:
        return self.chunks[x][z]

    def get_chunk_from_world_chunk_pos(self, worldChunkPos: glm.vec2) -> Chunk:
        if worldChunkPos[0] not in _WORLD_CHUNK_RANGE or worldChunkPos[1] not in _WORLD_CHUNK_RANGE:
            return CHUNK_NULL
        x, z = [int(i) for i in worldChunkPos + _WORLD_CHUNK_AXIS_LENGTH]
        return self._get_chunk(x, z)

    def get_chunk_from_world_block_pos(self, worldBlockPos: glm.vec3) -> Chunk:
        return self.get_chunk_from_world_chunk_pos(glm.floor(glm.vec2(worldBlockPos.xz) / _CHUNK_WIDTH))

    def get_block_from_world_block_pos(self, worldBlockPos: glm.vec3) -> Block:
        self.get_chunk_from_world_block_pos(worldBlockPos).get_block_from_world_block_pos(worldBlockPos)

    def update(self) -> bool:
        updated = False
        for chunkRow in self.chunks:
            for chunk in chunkRow:
                if chunk.update():
                    self.chunk_instances[chunk.worldChunkID] = chunk.get_instance_data()
                    updated = True
        return updated

    def get_instance_data(self):
        chunk_instances = list(filter((None).__ne__, self.chunk_instances))
        if len(chunk_instances) == 0:
            return None
        return np.concatenate(chunk_instances, dtype=chunk_instances[0].dtype)
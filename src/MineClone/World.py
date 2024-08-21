import copy
import os.path

from MineClone.Chunk import *
from MineClone.Chunk import _BLOCKS_IN_CHUNK, _QUADS_IN_CHUNK, _CHUNK_HEIGHT, _CHUNK_WIDTH, _CHUNK_SIZE

_WORLD_CHUNK_AXIS_LENGTH = 4
_WORLD_CHUNK_RANGE = range(-_WORLD_CHUNK_AXIS_LENGTH, _WORLD_CHUNK_AXIS_LENGTH + 1)
_CHUNKS_IN_ROW = len(_WORLD_CHUNK_RANGE)

_BLOCKS_IN_ROW = _CHUNKS_IN_ROW * _CHUNK_WIDTH
_WORLD_SIZE = glm.vec3(_BLOCKS_IN_ROW, _CHUNK_HEIGHT, _BLOCKS_IN_ROW)
_WORLD_SIZE_HALF = _WORLD_SIZE / 2

_WORLD_MID_POINT = glm.vec3(0, _WORLD_SIZE_HALF.y, 0)
_WORLD_MIN_POINT = _WORLD_MID_POINT - _WORLD_SIZE_HALF
_WORLD_MAX_POINT = _WORLD_MIN_POINT + _WORLD_SIZE_HALF

_CHUNKS_IN_WORLD = _CHUNKS_IN_ROW * _CHUNKS_IN_ROW

_BLOCKS_IN_WORLD = _BLOCKS_IN_CHUNK * _CHUNKS_IN_WORLD
_QUADS_IN_WORLD = _QUADS_IN_CHUNK * _CHUNKS_IN_WORLD


class World(PhysicalBox):
    # Initialize chunks as a 1D list
    chunks: list[Chunk] = [Chunk(glm.vec2(x, z)) for x in _WORLD_CHUNK_RANGE for z in _WORLD_CHUNK_RANGE]
    chunk_instances: list[np.ndarray] = [None] * _CHUNKS_IN_WORLD

    def _get_world_chunk_id(self, x: int, z: int) -> int:
        return x * _CHUNKS_IN_ROW + z

    def __init__(self):
        self.bounds = AABB.from_pos_size(_WORLD_MID_POINT, _WORLD_SIZE + glm.vec3(1))
        self.chunks = copy.deepcopy(World.chunks)
        self.not_empty_chunks: list[Chunk] = copy.deepcopy(self.chunk_instances)
        self.worldWidth = len(_WORLD_CHUNK_RANGE)
        self.chunks_needing_update: set[Chunk] = set()

        self.quadtree: QuadTree = QuadTree.XZ(self.bounds, _CHUNK_SIZE + glm.vec3(1))
        for x in range(self.worldWidth):
            for z in range(self.worldWidth):
                chunkIndex = self._get_world_chunk_id(x, z)
                chunk = self.chunks[chunkIndex]
                worldChunkID = chunkIndex
                chunk.init(self, worldChunkID)
                self.quadtree.insert(chunk)
                self.not_empty_chunks[worldChunkID] = chunk

    def query_aabb_chunks(self, boxRange: AABB) -> set[Chunk]:
        return self.quadtree.query_aabb(boxRange)

    def query_aabb_blocks(self, boxRange: AABB) -> set[Block]:
        hitBlocks: set[Block] = set()
        for hitChunk in self.query_aabb_chunks(boxRange):
            hitChunk.query_aabb_blocks(boxRange, hitBlocks)
        return hitBlocks

    def query_segment_chunks(self, ray: Ray) -> set[Chunk]:
        return self.quadtree.query_segment(ray)

    def query_segment_blocks(self, ray: Ray) -> set[Block]:
        hitBlocks: set[Block] = set()
        for hitChunk in self.query_segment_chunks(ray):
            hitChunk.query_segment_blocks(ray, hitBlocks)
        return hitBlocks

    def _get_chunk(self, x: int, z: int) -> Chunk:
        return self.chunks[self._get_world_chunk_id(x, z)]

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
        for chunk in list(filter((None).__ne__, self.not_empty_chunks)):
            chunk.update()
            self.chunk_instances[chunk.worldChunkID] = chunk.get_block_face_instance_data()
            updated = True
        return updated

    def flag_chunk_update(self, chunk: Chunk):
        self.chunks_needing_update.add(chunk)
    def run_partial_update(self):
        updated: bool = False
        for i in range(len(self.chunks_needing_update)):
            updated = True
            chunk: Chunk = self.chunks_needing_update.pop()
            self.chunk_instances[chunk.worldChunkID] = chunk.get_block_face_instance_data()
        return updated

    def get_instance_data(self):
        chunk_instances = list(filter((None).__ne__, self.chunk_instances))
        if not chunk_instances:
            return None
        return np.concatenate(chunk_instances, dtype=chunk_instances[0].dtype)

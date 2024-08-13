import copy

from MineClone.Chunk import *
from MineClone.Chunk import _BLOCKS_IN_CHUNK, _QUADS_IN_CHUNK

_WORLD_CHUNK_AXIS_LENGTH = 4
_WORLD_CHUNK_RANGE = range(-_WORLD_CHUNK_AXIS_LENGTH, _WORLD_CHUNK_AXIS_LENGTH+1)
_CHUNKS_IN_ROW = len(_WORLD_CHUNK_RANGE)
_CHUNKS_IN_WORLD = _CHUNKS_IN_ROW * _CHUNKS_IN_ROW

_BLOCKS_IN_WORLD = _BLOCKS_IN_CHUNK * _CHUNKS_IN_WORLD
_QUADS_IN_WORLD = _QUADS_IN_CHUNK * _CHUNKS_IN_WORLD

class World:
    chunks: list[list[Chunk]] = [[Chunk(glm.vec2(x, z)) for z in _WORLD_CHUNK_RANGE] for x in _WORLD_CHUNK_RANGE]
    chunk_instances: list[np.ndarray] = [None] * len(_WORLD_CHUNK_RANGE) * len(_WORLD_CHUNK_RANGE)
    def __init__(self):
        self.chunks = copy.deepcopy(World.chunks)
        self.worldWidth = len(self.chunks)
        for chunkX in range(self.worldWidth):
            for chunkZ in range(self.worldWidth):
                chunk = self.chunks[chunkX][chunkZ]
                worldChunkID = chunkX * self.worldWidth + chunkZ
                chunk.set(self, worldChunkID)

        self.init()

    def init(self):
        for chunkRow in self.chunks:
            for chunk in chunkRow:
                chunk.init()
                self.chunk_instances[chunk.worldChunkID] = chunk.get_instance_data()
    def _get_chunk(self, x: int, z: int) -> Chunk:
        return self.chunks[x][z]

    def get_chunk(self, worldChunkPos: glm.vec2) -> Chunk:
        if worldChunkPos[0] not in _WORLD_CHUNK_RANGE or worldChunkPos[1] not in _WORLD_CHUNK_RANGE:
            return CHUNK_NULL
        x, z = [int(i) for i in worldChunkPos + _WORLD_CHUNK_AXIS_LENGTH]
        return self._get_chunk(x, z)


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
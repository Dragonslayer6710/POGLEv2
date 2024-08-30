import copy
import os.path
import struct

import numpy as np

from MineClone.Chunk import *
from MineClone.Chunk import _BLOCKS_IN_CHUNK, _QUADS_IN_CHUNK, _CHUNK_HEIGHT, _CHUNK_WIDTH, _CHUNK_SIZE

_WORLD_CHUNK_AXIS_LENGTH = 1
_WORLD_CHUNK_RANGE = range(-_WORLD_CHUNK_AXIS_LENGTH, _WORLD_CHUNK_AXIS_LENGTH + 1)
_CHUNKS_IN_ROW = len(_WORLD_CHUNK_RANGE)
_BLOCKS_IN_ROW = _CHUNKS_IN_ROW * _CHUNK_WIDTH

_WORLD_MAX = glm.vec3(
    _BLOCKS_IN_ROW/2,
    _CHUNK_HEIGHT,
    _BLOCKS_IN_ROW/2
)
_WORLD_MIN = glm.vec3(
    -_WORLD_MAX.x,
    0,
    -_WORLD_MAX.z
)

_WORLD_SIZE = _WORLD_MAX - _WORLD_MIN

_WORLD_CENTER = _WORLD_MIN + _WORLD_SIZE / 2


_CHUNKS_IN_WORLD = _CHUNKS_IN_ROW * _CHUNKS_IN_ROW

_BLOCKS_IN_WORLD = _BLOCKS_IN_CHUNK * _CHUNKS_IN_WORLD
_QUADS_IN_WORLD = _QUADS_IN_CHUNK * _CHUNKS_IN_WORLD


class World:
    pass


class WorldRenderer:
    def _build_mesh(self):
        pass


class World(PhysicalBox):
    block_face_ids: list[np.array] = [None] * _CHUNKS_IN_WORLD
    block_face_tex_dims: list[np.array] = [None] * _CHUNKS_IN_WORLD
    block_positions: list[np.array] = [None] * _CHUNKS_IN_WORLD

    not_empty_chunks: list[int | None] = [None] * _CHUNKS_IN_WORLD

    # World chunk array index from chunk array offset (chunk.x/z = 0 to _CHUNKS_IN_ROW)
    @staticmethod
    def chunk_id_from_chunk_array_offset(chunk_array_offset: glm.vec2) -> int:
        return int(chunk_array_offset[0] * _CHUNKS_IN_ROW + chunk_array_offset[1])

    # World chunk array offset vec2 from chunk position in world
    # (chunk.x/z = -WORLD_CHUNK_AXIS_LENGTH to WORLD_CHUNK_AXIS_LENGTH)
    @staticmethod
    def chunk_array_offset_from_chunk_pos(chunk_pos: glm.vec2) -> glm.vec2:
        return chunk_pos + _WORLD_CHUNK_AXIS_LENGTH

    # World chunk array index from chunk position in world
    # (chunk.x/z = -WORLD_CHUNK_AXIS_LENGTH to WORLD_CHUNK_AXIS_LENGTH)
    @staticmethod
    def chunk_id_from_chunk_pos(chunk_pos: glm.vec2) -> int:
        return World.chunk_id_from_chunk_array_offset(
            World.chunk_array_offset_from_chunk_pos(chunk_pos)
        )

    # World chunk array offset vec2 from position in world
    # (worldPos.x/z = -inf to inf)
    @staticmethod
    def chunk_array_offset_from_world_pos(worldPos: glm.vec3) -> glm.vec2:
        return World.chunk_array_offset_from_chunk_pos(
            Chunk.pos_from_world_pos(worldPos)
        )

    # World chunk array offset index from position in world
    # (worldPos.x/z = -inf to inf)
    @staticmethod
    def chunk_id_from_world_pos(worldPos: glm.vec3) -> int:
        return World.chunk_id_from_chunk_array_offset(
            World.chunk_array_offset_from_world_pos(worldPos)
        )

    def chunk_from_world_pos(self, worldPos: glm.vec3) -> Chunk:
        return self.chunks(World.chunk_id_from_world_pos(worldPos))

    def chunk_from_chunk_array_offset(self, chunk_array_offset: glm.vec2) -> Chunk:
        return self.chunks(World.chunk_id_from_chunk_array_offset(chunk_array_offset))

    def chunk_from_chunk_pos(self, chunk_pos: glm.vec2) -> Chunk:
        return self.chunks(self.chunk_id_from_chunk_pos(chunk_pos))

    def __init__(self, chunks: list[Chunk] = None):
        self.bounds = AABB.from_pos_size(_WORLD_CENTER, _WORLD_SIZE)

        self._chunks: list[Chunk] = []

        self.block_face_ids = copy.deepcopy(World.block_face_ids)
        self.block_face_tex_dims = copy.deepcopy(World.block_face_tex_dims)
        self.block_positions = copy.deepcopy(World.block_positions)

        self.not_empty_chunks = copy.deepcopy(World.not_empty_chunks)

        self.worldWidth = len(_WORLD_CHUNK_RANGE)

        self.quadtree: QuadTree = QuadTree.XZ(self.bounds, _CHUNK_SIZE + glm.vec3(1))
        self.renderer: WorldRenderer | None = None

        if chunks is None:
            self._initialize_default_chunks()
        else:
            self._initialize_with_chunks(chunks)

        for chunk in self._chunks:
            self.update_chunk_in_world(chunk)
        self.update()

    def _initialize_default_chunks(self):
        for x in range(self.worldWidth):
            for z in range(self.worldWidth):
                chunk = Chunk(glm.vec2(x, z))
                chunk.link_world(self)
                self._chunks.append(chunk)

    def _initialize_with_chunks(self, chunks):
        for chunk in chunks:
            chunk.link_world(self)
            self._chunks.append(chunk)

    def link_renderer(self, renderer: WorldRenderer):
        self.renderer = renderer

    def update_chunk_in_world(self, chunk: Chunk):
        if chunk.not_empty:
            if None is self.not_empty_chunks[chunk.chunk_id]:
                self.not_empty_chunks[chunk.chunk_id] = chunk.chunk_id
                self.quadtree.insert(chunk)
        else:
            if self.not_empty_chunks[chunk.chunk_id] == chunk.chunk_id:
                self.not_empty_chunks[chunk.chunk_id] = None
                self.quadtree.remove(chunk)
        if self.not_empty_chunks[chunk.chunk_id]:
            self.set_chunk_instance(chunk)
        else:
            self.set_chunk_instance(chunk, True)
        if self.renderer:
            self.renderer._build_mesh()


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

    def chunks(self, worldChunkID: int) -> Chunk:
        return self._chunks[worldChunkID]

    def update(self) -> bool:
        updated = False
        for worldChunkID in list(filter((None).__ne__, self.not_empty_chunks)):
            chunk: Chunk = self.chunks(worldChunkID)
            chunk.update()
            self.set_chunk_instance(chunk)
            updated = True
        return updated

    def get_instance_data(self):
        block_instances = list(filter((None).__ne__, self.block_positions))
        block_face_instances = list(filter((None).__ne__, self.block_face_tex_dims))
        if not block_instances:
            return None, None
        return (
            np.concatenate(block_instances),
            np.concatenate(block_face_instances)
        )

    def set_chunk_instance(self, chunk: Chunk, clear=False):
        (
            self.block_face_ids[chunk.chunk_id],
            self.block_face_tex_dims[chunk.chunk_id],
            self.block_positions[chunk.chunk_id]
        ) = chunk.get_block_and_face_instance_data() if not clear else (None, None, None)

    def get_block_and_face_instance_data(self, worldChunkID: int):
        return (
            self.block_face_ids[worldChunkID],
            self.block_face_tex_dims[worldChunkID],
            self.block_positions[worldChunkID]
        )

    def get_chunk(self, normal_chunk_pos: glm.vec2 = None) -> Chunk:
        worldChunkX, worldChunkZ = [int(i) for i in normal_chunk_pos]

        xOutOfRange, zOutOfRange = worldChunkX not in _WORLD_CHUNK_RANGE, worldChunkZ not in _WORLD_CHUNK_RANGE
        if xOutOfRange or zOutOfRange:
            return None
        else:
            return self.chunk_from_chunk_array_offset(normal_chunk_pos)

    def serialize(self) -> bytes:
        # Serialize the number of chunks
        num_chunks_data = struct.pack('I', len(self._chunks))

        # Serialize each chunk
        chunk_data = b""
        for chunk in self._chunks:
            chunk_serialized = chunk.serialize()
            chunk_size = struct.pack('I', len(chunk_serialized))
            chunk_data += chunk_size + chunk_serialized

        # Serialize chunk instances
        chunk_instances_data = b"".join([
            instance.tobytes() if instance is not None else b""
            for instance in self.block_face_tex_dims
        ])

        # Serialize not_empty_chunks
        not_empty_chunks_data = struct.pack(f'{len(self.not_empty_chunks)}I', *self.not_empty_chunks)

        return num_chunks_data + chunk_data + chunk_instances_data + not_empty_chunks_data

    @classmethod
    def deserialize(cls, binary_data: bytes):
        offset = 0

        # Unpack the number of chunks
        if len(binary_data) < 4:
            raise ValueError("Insufficient data to read the number of chunks")
        num_chunks = struct.unpack('I', binary_data[offset:offset + 4])[0]
        offset += 4

        chunks = []
        for _ in range(num_chunks):
            if offset + 4 > len(binary_data):
                raise ValueError("Insufficient data to read chunk size")
            chunk_size = struct.unpack('I', binary_data[offset:offset + 4])[0]
            offset += 4

            if offset + chunk_size > len(binary_data):
                raise ValueError(
                    f"Insufficient data to read the complete chunk. Expected size: {chunk_size}, available data: {len(binary_data) - offset}")

            chunk_data = binary_data[offset:offset + chunk_size]
            chunks.append(Chunk.deserialize(chunk_data))
            offset += chunk_size

        # Deserialize the chunk instances
        chunk_instances = []
        for _ in range(_CHUNKS_IN_WORLD):
            instance_size = Block.positionInstanceLayout.nbytes * _BLOCKS_IN_CHUNK
            if offset + instance_size > len(binary_data):
                chunk_instances.append(None)
            else:
                chunk_instance_data = binary_data[offset:offset + instance_size]
                chunk_instances.append(
                    np.frombuffer(chunk_instance_data, dtype=Block.positionInstanceLayout.dtype).reshape((6, 4, 4)))
                offset += instance_size

        # Deserialize the not_empty_chunks
        not_empty_chunks_format = f'{_CHUNKS_IN_WORLD}I'
        not_empty_chunks_size = struct.calcsize(not_empty_chunks_format)
        if offset + not_empty_chunks_size > len(binary_data):
            raise ValueError("Insufficient data to read not_empty_chunks")
        not_empty_chunks = list(
            struct.unpack(not_empty_chunks_format, binary_data[offset:offset + not_empty_chunks_size]))
        offset += not_empty_chunks_size

        # Create the world instance
        world = cls(chunks)
        world.block_face_tex_dims = chunk_instances
        world.not_empty_chunks = not_empty_chunks

        return world


class WorldRenderer:
    def __init__(self, world: World, renderDistance: int = 1):
        world.link_renderer(self)
        self.world: World = world

        self._chunk_ids_in_world: list[list[int]] = [
            [World.chunk_id_from_chunk_pos(glm.vec2(x,z)) for z in _WORLD_CHUNK_RANGE] for x in _WORLD_CHUNK_RANGE
        ]
        self._origin_chunk_pos: glm.vec2 | None = None
        self._origin_chunk_id: int | None = None
        self._set_origin_chunk_pos(glm.vec2() + _WORLD_CHUNK_AXIS_LENGTH)

        self._render_distance: int | None = None
        self._render_bounds: AABB | None = None
        self._rMin: range | None = None
        self._rMax: range | None = None
        self._xRenderRange: range | None = None
        self._zRenderRange: range | None = None
        self.rendered_chunk_ids: list[list[int | None]] | None = None

        self.worldBlockShader: ShaderProgram = ShaderProgram("block", "block")
        self.worldBlockShader.bind_uniform_block("Matrices")
        self.worldBlockShader.bind_uniform_block("BlockSides")
        self.worldMesh: Mesh = None

        self.set_render_distance(renderDistance)

    def _get_origin_chunk_id(self) -> int:
        return self._chunk_ids_in_world[int(self._origin_chunk_pos[0])][int(self._origin_chunk_pos[1])]

    def _set_origin_chunk_id(self):
        self._origin_chunk_id: int = self._get_origin_chunk_id()

    def _set_origin_chunk_pos(self, origin_chunk_pos: glm.vec2):
        self._origin_chunk_pos = origin_chunk_pos
        self._set_origin_chunk_id()

    def set_render_distance(self, renderDistance: int):
        self._render_distance = renderDistance
        self._set_rendered_chunk_bounds()

    def _set_rendered_chunk_bounds(self):
        size: int | glm.vec3 = 2 * self._render_distance
        size = glm.vec3(size, 0, size)
        pos: glm.vec3 = glm.vec3(self._origin_chunk_pos[0], 0, self._origin_chunk_pos[1])

        self._render_bounds = AABB.from_pos_size(pos, size)
        self._rMin = self._render_bounds.min
        self._rMax = self._render_bounds.max
        self._xRenderRange = range(int(self._rMin.x), int(self._rMax.x)+1)
        self._zRenderRange = range(int(self._rMin.z), int(self._rMax.z)+1)
        self._set_rendered_chunk_ids()

    def _set_rendered_chunk_ids(self):
        self.rendered_chunk_ids = [
            self._chunk_ids_in_world[x][z] if x > -1 and z > -1 and x < _CHUNKS_IN_ROW and z < _CHUNKS_IN_ROW else None for x in self._xRenderRange for z in self._zRenderRange
        ]
        self._build_mesh()

    def update_origin(self, originPos: glm.vec3):
        chunk_pos: glm.vec2 = glm.clamp(World.chunk_array_offset_from_world_pos(originPos), 0,
                                        len(self._chunk_ids_in_world)-1)
        if not chunk_pos == self._origin_chunk_pos:  # New position is outside of origin chunk
            self._set_origin_chunk_pos(chunk_pos)
            self._set_rendered_chunk_bounds()
    def get_instance_data(self):
        block_face_ids = []
        block_face_tex_dims = []
        block_positions = []
        for rendered_chunk_id in self.rendered_chunk_ids:
            (
                ids,
                tex_dims,
                positions
            ) = self.world.get_block_and_face_instance_data(rendered_chunk_id) if rendered_chunk_id is not None else (None, None, None)
            block_face_ids.append(ids)
            block_face_tex_dims.append(tex_dims)
            block_positions.append(positions)
        block_face_ids = list(filter((None).__ne__, block_face_ids))
        block_face_tex_dims = list(filter((None).__ne__, block_face_tex_dims))
        block_positions = list(filter((None).__ne__, block_positions))
        if len(block_positions) == 0:
            return np.array([]), np.array([]), np.array([])
        return (
            np.concatenate(block_face_ids).astype(np.int32),
            np.concatenate(block_face_tex_dims, dtype=np.float32),
            np.concatenate(block_positions, dtype=np.float32)
        )

    def _build_mesh(self):
        self.worldMesh = BlockMesh(*self.get_instance_data())

    def draw(self, projection: glm.mat4, view: glm.mat4):
        #return
        self.worldMesh.draw(self.worldBlockShader, projection, view)
        # CubeMesh(NMM(self.world.pos,s=self.world.size), alpha=0.5).draw(projection, view)
        # for chunk in self.world._chunks:
        #     CubeMesh(NMM(chunk.pos, s=chunk.size), color=Color.BLUE, alpha=0.5).draw(projection, view)
    def __str__(self):
        print_str = ""
        for z in range(self._chunks_in_rendered_row):
            for x in range(self._chunks_in_rendered_row):
                if x > 0:
                    print_str += ",\t"
                chunk = self.rendered_chunk_ids[x][z]
                if chunk.is_chunk:
                    print_str += f"({chunk.worldChunkPos[0]},\t{chunk.worldChunkPos[1]})"
                else:
                    print_str += f"(NaN,\tNaN)"
                if self._chunks_in_rendered_row - 1 == x:
                    print_str += "\n"
        return print_str

from __future__ import annotations
from POGLE.Core.Core import *

import copy

import numpy as np

from MineClone.Face import FaceTex, _face_tex_cache, test_faces

from POGLE.Physics.SpatialTree import *
from dataclasses import dataclass


class BlockID(Enum, metaclass=REnum):
    Air = 0
    Grass = auto()
    Dirt = auto()
    Stone = auto()


_block_id_cache: Dict[int, BlockID] = {
    block_id.value: block_id for block_id in BlockID
}

FACES_IN_BLOCK_RANGE = range(6)
NULL_FACES = [FaceTex.Null for _ in FACES_IN_BLOCK_RANGE]

SECTION_WIDTH: int = 16
SECTION_WIDTH_HALF = SECTION_WIDTH // 2
SECTION_SIZE: int = SECTION_WIDTH ** 2
SECTION_EXTENTS: glm.vec2 = glm.vec2(SECTION_WIDTH, SECTION_WIDTH)

CHUNK_HEIGHT: int = 256
CHUNK_HEIGHT_HALF: int = CHUNK_HEIGHT // 2
CHUNK_SIZE: int = CHUNK_HEIGHT

REGION_WIDTH: int = 32
REGION_WIDTH_HALF = REGION_WIDTH // 2
REGION_SIZE: int = REGION_WIDTH ** 2
REGION_EXTENTS: glm.vec2 = glm.vec2(REGION_WIDTH, REGION_WIDTH)

WORLD_WIDTH: int = 3
WORLD_WIDTH_HALF = WORLD_WIDTH // 2
WORLD_SIZE: int = WORLD_WIDTH ** 2
WORLD_EXTENTS: glm.vec2 = glm.vec2(WORLD_WIDTH, WORLD_WIDTH)

BLOCKS_IN_SECTION: int = SECTION_SIZE
BLOCKS_IN_SECTION_HALF: int = BLOCKS_IN_SECTION // 2
BLOCKS_IN_SECTION_WIDTH: int = SECTION_WIDTH
BLOCKS_IN_SECTION_WIDTH_HALF: int = SECTION_WIDTH_HALF

BLOCKS_IN_CHUNK: int = CHUNK_SIZE * BLOCKS_IN_SECTION
BLOCKS_IN_CHUNK_HALF: int = BLOCKS_IN_CHUNK // 2
BLOCKS_IN_CHUNK_HEIGHT: int = CHUNK_HEIGHT
BLOCKS_IN_CHUNK_HEIGHT_HALF: int = CHUNK_HEIGHT_HALF

BLOCKS_IN_REGION: int = REGION_SIZE * BLOCKS_IN_CHUNK
BLOCKS_IN_REGION_HALF: int = BLOCKS_IN_REGION // 2
BLOCKS_IN_REGION_WIDTH: int = REGION_WIDTH * BLOCKS_IN_SECTION_WIDTH
BLOCKS_IN_REGION_WIDTH_HALF: int = BLOCKS_IN_REGION_WIDTH // 2

BLOCKS_IN_WORLD: int = WORLD_SIZE * BLOCKS_IN_REGION
BLOCKS_IN_WORLD_HALF: int = BLOCKS_IN_WORLD // 2
BLOCKS_IN_WORLD_WIDTH: int = WORLD_WIDTH * BLOCKS_IN_REGION_WIDTH
BLOCKS_IN_WORLD_WIDTH_HALF: int = BLOCKS_IN_WORLD_WIDTH // 2


def _get_a_b_index(a_width: int, a_width_half: int, a_x: int, a_z: int, major_axis: int = 1) -> int:
    """
    Compute the 1D index based on a 2D coordinate.
    """
    if major_axis:
        return (a_z + a_width_half) * a_width + (a_x + a_width_half)
    else:
        return (a_x + a_width_half) * a_width + (a_z + a_width_half)


def get_a_b_index(a_width: int, a_width_half: int, a_x: Union[int, glm.vec2, glm.vec3], a_z: Optional[int] = None,
                  major_axis: int = 1) -> int:
    """
    Compute the 1D index based on a coordinate input (scalar or vector).
    """
    if isinstance(a_x, glm.vec3):
        return _get_a_b_index(a_width, a_width_half, a_x.x, a_x.z, major_axis)
    elif isinstance(a_x, glm.vec2):
        return _get_a_b_index(a_width, a_width_half, a_x.x, a_x.y, major_axis)
    else:
        return _get_a_b_index(a_width, a_width_half, a_x, a_z, major_axis)


def get_world_region_index(world_x: Union[int, glm.vec2, glm.vec3], world_z: Optional[int] = None) -> int:
    """
    Get the index of a region based on world coordinates.
    """
    return get_a_b_index(WORLD_WIDTH, WORLD_WIDTH_HALF, world_x, world_z)


def get_region_chunk_index(region_x: Union[int, glm.vec2, glm.vec3], region_z: Optional[int] = None) -> int:
    """
    Get the index of a chunk based on region coordinates.
    """
    return get_a_b_index(REGION_WIDTH, REGION_WIDTH_HALF, region_x, region_z)


def get_chunk_section_index(chunk_y: Union[int, glm.vec2, glm.vec3]) -> int:
    """
    Get the index of a section based on chunk Y coordinate.
    """
    if isinstance(chunk_y, glm.vec3):
        return int(chunk_y.y)
    return chunk_y


def get_section_block_index(section_x: Union[int, glm.vec2, glm.vec3], section_z: Optional[int] = None,
                            major_axis: int = 1) -> int:
    """
    Get the index of a section based on section coordinates.
    """
    return get_a_b_index(SECTION_WIDTH, SECTION_WIDTH_HALF, section_x, section_z, major_axis)


def get_chunk_block_index(chunk_y: int, section_block_index: int) -> int:
    """
    Get the index of a block based on chunk Y coordinate and section block index.
    """
    return chunk_y * SECTION_SIZE + section_block_index


def get_region_block_index(region_x: int, region_z: int, chunk_block_index: int) -> int:
    """
    Get the index of a block based on region coordinates and chunk block index.
    """
    return region_x * CHUNK_SIZE * SECTION_SIZE + region_z * CHUNK_SIZE + chunk_block_index


def get_world_block_index(world_x: int, world_z: int, region_block_index: int) -> int:
    """
    Get the index of a block based on world coordinates and region block index.
    """
    return world_x * REGION_SIZE * CHUNK_SIZE * SECTION_SIZE + world_z * REGION_SIZE * CHUNK_SIZE + region_block_index


def get_a_coords(a_width: int, a_width_half: int, index: int, major_axis: int = 1, a_y: Optional[int] = None) -> Union[
    glm.vec2, glm.vec3]:
    """
    Convert a 1D index back into 2D or 3D coordinates.
    """
    if major_axis:
        a_z = index // a_width - a_width_half
        a_x = index % a_width - a_width_half
    else:
        a_x = index // a_width - a_width_half
        a_z = index % a_width - a_width_half

    if a_y is not None:
        return glm.vec3(a_x, a_y, a_z)
    else:
        return glm.vec2(a_x, a_z)


def get_world_region_coords(region_in_world_index: int) -> glm.vec2:
    """
    Get world region coordinates from index.
    """
    return get_a_coords(WORLD_WIDTH, WORLD_WIDTH_HALF, region_in_world_index)


def get_region_chunk_coords(chunk_in_region_index: int) -> glm.vec2:
    """
    Get region chunk coordinates from index.
    """
    return get_a_coords(REGION_WIDTH, REGION_WIDTH_HALF, chunk_in_region_index)


def get_section_block_coords(block_in_section_index: int) -> glm.vec2:
    """
    Get section block coordinates from index.
    """
    return get_a_coords(BLOCKS_IN_SECTION_WIDTH, BLOCKS_IN_SECTION_WIDTH_HALF, block_in_section_index)


def get_chunk_block_coords(block_in_section_index: int, block_height: int) -> glm.vec3:
    """
    Get chunk block coordinates from index.
    """
    return get_a_coords(BLOCKS_IN_SECTION_WIDTH, BLOCKS_IN_SECTION_WIDTH_HALF, block_in_section_index, a_y=block_height)


def get_region_block_coords(block_in_region_index: int, block_height: int) -> glm.vec3:
    """
    Get region block coordinates from index.
    """
    return get_a_coords(BLOCKS_IN_REGION_WIDTH, BLOCKS_IN_REGION_WIDTH_HALF, block_in_region_index, a_y=block_height)


def get_world_block_coords(block_in_world_index: int, block_height: int) -> glm.vec3:
    """
    Get world block coordinates from index.
    """
    return get_a_coords(BLOCKS_IN_WORLD_WIDTH, BLOCKS_IN_WORLD_WIDTH_HALF, block_in_world_index, a_y=block_height)


_pack_fixed_length_string_cache: Dict[int, Dict[str, bytes]] = {}


def pack_fixed_length_string(s: str, length: int) -> bytes:
    """
    Packs a string into a fixed-length byte string.

    Args:
        s: The string to pack.
        length: The desired length of the packed byte string.

    Returns:
        A byte string of the specified length, padded with zeros if necessary.

    Raises:
        ValueError: If the string is longer than the specified length.
    """
    cached_length_dict = _pack_fixed_length_string_cache.get(length)
    if cached_length_dict:
        if s in cached_length_dict:
            return cached_length_dict[s]
    else:
        cached_length_dict = _pack_fixed_length_string_cache[length] = {}

    # Encode the string
    encoded_str = s.encode('utf-8')
    # Pad the encoded string if shorter than the specified length
    if len(encoded_str) < length:
        encoded_str = encoded_str.ljust(length, b'\x00')
    elif len(encoded_str) > length:
        raise ValueError(f"String length {len(encoded_str)} exceeds the specified length {length}")
    if not cached_length_dict.get(s):
        cached_length_dict[s] = encoded_str
    # Return the packed data
    return encoded_str


_SOLID_MASK = 0b00000001
_OPAQUE_MASK = 0b00000010
_FACES_IN_BLOCK_BYTES = 6 * struct.calcsize(STRUCT_FORMAT_SHORT)
_PROPERTY_NAME_BYTES = 16
_PROPERTY_BOOLS_START = _PROPERTY_NAME_BYTES
_PROPERTY_FACES_START = _PROPERTY_BOOLS_START + 1
_PROPERTY_FACES_DTYPE = int


class BlockState():
    _properties_dict: Dict[BlockID, Dict[str, Any]] = {
        BlockID.Air: {"name": "Air", "is_solid": False, "is_opaque": False, "face_textures": []},
        BlockID.Dirt: {"name": "Dirt", "is_solid": True, "is_opaque": False, "face_textures": [FaceTex.Dirt]*6},
        BlockID.Stone: {"name": "Stone", "is_solid": True, "is_opaque": True, "face_textures": [FaceTex.Stone]*6},
        BlockID.Grass: {"name": "Grass", "is_solid": True, "is_opaque": True, "face_textures": [FaceTex.GrassSide]*4 + [FaceTex.GrassTop] + [FaceTex.Dirt]},
    }

    SERIALIZED_FMT_STRS = ["H"]
    SERIALIZED_SIZE = np.sum([struct.calcsize(fmt_str) for fmt_str in SERIALIZED_FMT_STRS])

    class Side(Enum, metaclass=REnum):
        Null = -1
        West = auto()
        South = auto()
        East = auto()
        North = auto()
        Top = auto()
        Bottom = auto()

        @property
        def opposite(self) -> BlockState.Side:
            return BlockState._opposite_dict[self]

    _opposite_dict: dict[BlockState.Side, BlockState.Side] = {
        Side.West: Side.East,
        Side.East: Side.West,
        Side.South: Side.North,
        Side.North: Side.South,
        Side.Top: Side.Bottom,
        Side.Bottom: Side.Top
    }

    def __init__(self, block_id: Union[int, np.uint16, BlockID] = BlockID.Air):
        if isinstance(block_id, Union[int, np.uint16]):
            block_id = _block_id_cache[block_id]
        self.ID: BlockID = block_id

        self._visible_faces = self._visible_faces_cache = 0
        self._visible_faces_cache_valid = False

    def set_block(self, block_id: Union[int, BlockID]):
        was_opaque = self.is_opaque
        was_solid = self.is_solid
        self.ID: BlockID = block_id

        self.update_visible_faces()

    def update_visible_faces(self):
        if self.visible_faces:
            if BlockID.Air == self.ID:
                self._visible_faces = 0
                return
            # TODO: adjacency checks

    @property
    def properties(self) -> Dict[str, Any]:
        return BlockState._properties_dict[self.ID]

    @property
    def visible_faces(self):
        if self._visible_faces_cache_valid:
            return self._visible_faces_cache
        self._visible_faces_cache = bin(self._visible_faces).count("1")
        self._visible_faces_cache_valid = True
        return self._visible_faces_cache

    @property
    def faces(self):
        if not self.properties:
            raise RuntimeError("Cannot access faces until block properties are set")
        return self.properties.face_textures

    def hide_face(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = BlockState.Side(side)
        if self.face_visible(side):
            self._visible_faces &= ~(1 << side.value)
            self._visible_faces -= 1

    def reveal_face(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = BlockState.Side(side)
        if not self.face_visible(side):
            self._visible_faces |= 1 << side.value
            self._visible_faces += 1

    def face_visible(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = BlockState.Side(side)
        return self._visible_faces & (1 << side.value)

    def get_face_instances(self):
        return b"".join([face.get_instance() for face in self.faces])

    def get_instance(self):
        if self.section_in_chunk_index is None:
            raise RuntimeError("Cannot Get Block Instance If No Section index is Set")
        return

    @property
    def name(self):
        return self.properties.name

    @property
    def is_solid(self):
        return self.properties["is_solid"]

    @property
    def is_opaque(self):
        return self.properties["is_opaque"]

    @property
    def is_renderable(self):
        return self._visible_faces and self.ID != BlockID.Air  # If not air and have faces that are visible, it is renderable

    def serialize(self) -> bytes:
        return struct.pack("H", self.ID.value)

    @classmethod
    def deserialize(cls, packed_data: bytes) -> BlockState:
        # Unpack the serialized data
        return cls(block_id=np.frombuffer(packed_data, dtype=np.uint16)[0])

    @staticmethod
    def serialize_array(block_states: list[BlockState]) -> bytes:
        return struct.pack(f'<{len(block_states)}H', *[bs.ID.value for bs in block_states])

    @staticmethod
    def deserialize_array(packed_data: bytes) -> list[BlockState]:
        unpacked_data = struct.unpack(f'<{len(packed_data) // 2}H', packed_data)
        return list(map(BlockState, unpacked_data))


    def __repr__(self):
        return f"Block(id: {self.ID}, name: {self.name}, solid: {self.is_solid}, opaque: {self.is_opaque}, visible_faces:{self._visible_faces})"

class Block(PhysicalBox):
    def __init__(self, pos: glm.vec3, block_state: BlockState):
        super().__init__(AABB.from_pos_size(pos))
        self.block_state: BlockState = block_state

_block_state_dict: Dict[BlockID, bytes] = {
    BlockID.Air: BlockState().serialize(),
    BlockID.Dirt: BlockState(BlockID.Dirt).serialize(),
    BlockID.Grass: BlockState(BlockID.Grass).serialize(),
    BlockID.Stone: BlockState(BlockID.Stone).serialize()
}


def get_block(block_id: BlockID = BlockID.Air) -> BlockState:
    return BlockState.deserialize(_block_state_dict[block_id])


def NULL_BLOCK_STATE() -> BlockState:
    return get_block()


import time

import struct

from timeit import timeit


# def deserialize_block_states(packed_data: bytes, size: int) -> list[BlockState]:
#     block_id_values = np.frombuffer(packed_data, dtype=np.uint16)
#     return [BlockState(block_id=block_id_value) for block_id_value in block_id_values]

TEST_MULTIPLIER = 1

def test_blocks(wider_timer: bool = False):
    one_at_a_time_tests = BLOCKS_IN_CHUNK
    null_block_state = NULL_BLOCK_STATE()
    null_block_state_bytes = null_block_state.serialize()

    all_at_once_tests = TEST_MULTIPLIER * one_at_a_time_tests
    if not os.path.exists("blocks.bin"):
        null_block_state_array = [NULL_BLOCK_STATE() for _ in range(all_at_once_tests)]
        null_block_state_array_bytes = BlockState.serialize_array(null_block_state_array)
        with open("blocks.bin", "wb") as f:
            f.write(null_block_state_array_bytes)
    else:
        with open("blocks.bin", "rb") as f:
            null_block_state_array_bytes = f.read()

    if not wider_timer:
        print(f"Testing Block generation with a Chunk's worth of blocks one by one: {one_at_a_time_tests}")
        start_time = time.perf_counter()
    for _ in range(one_at_a_time_tests):
        block = BlockState.deserialize(null_block_state_bytes)
    if not wider_timer:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        one_by_one_average_time = one_at_a_time_tests / execution_time
        print(f"Deserialize time: {execution_time:.4f} seconds\nBlocks deserialized per second: {one_by_one_average_time:.4f}")

        print(f"Testing Block generation with {TEST_MULTIPLIER}xChunk's worth of blocks at once: {all_at_once_tests}")
        start_time = time.perf_counter()
    blocks = BlockState.deserialize_array(null_block_state_array_bytes)
    if not wider_timer:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        all_at_once_average_time = all_at_once_tests / execution_time
        print(f"Deserialize time: {execution_time:.4f} seconds\nBlocks deserialized per second: {all_at_once_average_time:.4f}")

        print(f"All At Once took {all_at_once_average_time / one_by_one_average_time:.4f}x one at a time's time")

    start_time = time.perf_counter()
    BlockState.serialize_array(blocks)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Serialize time: {execution_time:.4f} seconds")
import cProfile
import pstats

if __name__ == "__main__":
    test_blocks()
    #execution_time = timeit(test_blocks, number=10)
    #print(f"Execution time: {execution_time:.4f} seconds")
    #cProfile.run("test_blocks()", "blocks.prof")

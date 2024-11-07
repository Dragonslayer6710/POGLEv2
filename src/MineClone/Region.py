from __future__ import annotations

import os
import pickle
import struct
from dataclasses import dataclass, field
from typing import Callable, BinaryIO

import nbtlib

from Chunk import *

if TYPE_CHECKING:
    from World import World

REGION_CHUNK_WIDTH: int = 32
_REGION_CHUNK_WIDTH_ID_RANGE: range = range(REGION_CHUNK_WIDTH)

REGION_CHUNK_EXTENTS: glm.vec3 = glm.vec3(REGION_CHUNK_WIDTH, 1, REGION_CHUNK_WIDTH)
REGION_NUM_CHUNKS: int = int(np.prod(REGION_CHUNK_EXTENTS))
_REGION_CHUNK_ID_RANGE: range = range(REGION_NUM_CHUNKS)

REGION_CHUNK_OFFSETS: List[glm.vec3] = [glm.vec3(x, 0, z) for x in _REGION_CHUNK_ID_RANGE for z in
                                        _REGION_CHUNK_ID_RANGE]

REGION_BLOCK_WIDTH: int = REGION_CHUNK_WIDTH * CHUNK_BLOCK_WIDTH
REGION_BLOCK_EXTENTS: glm.vec3 = REGION_CHUNK_EXTENTS * CHUNK_BLOCK_EXTENTS
REGION_NUM_BLOCKS: int = REGION_NUM_CHUNKS * CHUNK_NUM_BLOCKS

REGION_BASE_AABB = AABB.from_pos_size(size=REGION_BLOCK_EXTENTS)

REGION_CHUNK_NONE_LIST = [None] * REGION_NUM_CHUNKS

PROCESS_POOL_SIZE = max(1, os.cpu_count() - 1)

WORLD_REGION_WIDTH: int = 3

if WORLD_REGION_WIDTH % 3:
    raise RuntimeError(f"World Region Width: {WORLD_REGION_WIDTH} is not acceptable."
                       f" It must be a multiple of 3")
WORLD_REGION_WIDTH_HALF: int = WORLD_REGION_WIDTH // 2
_WORLD_REGION_WIDTH_ID_RANGE: range = range(-WORLD_REGION_WIDTH_HALF, WORLD_REGION_WIDTH_HALF + 1)


######### Coordinate - Coordinate Translation #########
# World Region Coords to World Coords
def wr_coords_to_w_coords(wr_coords: glm.vec2) -> glm.vec3:
    return glm.vec3(wr_coords[0], 1, wr_coords[1]) * REGION_BLOCK_EXTENTS // 2


# World Coords to Region Chunk Coords
def w_coords_to_rc_coords(w_coords: glm.vec3) -> glm.vec2:
    return glm.vec2(w_coords.x // CHUNK_BLOCK_WIDTH % REGION_CHUNK_WIDTH,
                    w_coords.z // CHUNK_BLOCK_WIDTH % REGION_CHUNK_WIDTH)


def w_coords_to_wr_coords(w_coords: glm.vec3) -> glm.vec2:
    return glm.vec2(
        int(w_coords.x / (REGION_BLOCK_EXTENTS.x / 2)),
        int(w_coords.z / (REGION_BLOCK_EXTENTS.z / 2)),
    )  # Drops the decimal part


######################################################

######### Coordinate - ID Translation #########

def wr_coords_to_wrid(wr_coords: glm.vec2) -> int:
    # Assuming WORLD_REGION_WIDTH_HALF is defined as half the width of your region grid
    return int((wr_coords[0] + WORLD_REGION_WIDTH_HALF) +
               (wr_coords[1] + WORLD_REGION_WIDTH_HALF) * WORLD_REGION_WIDTH)


# World Coords to World Region ID
def w_coords_to_wrid(w_coords: glm.vec3) -> int:
    return wr_coords_to_wrid(
        w_coords_to_wr_coords(w_coords)
    )


# Region Chunk Coords to Region Chunk ID
def rc_coords_to_rcid(rc_coords: Union[int, glm.vec2], z: Optional[int] = None) -> int:
    if isinstance(rc_coords, int):
        if z:
            return rc_coords + z * REGION_CHUNK_WIDTH
        else:
            raise TypeError("If rc_coords is int, z must be provided")
    return int(rc_coords[0] + rc_coords[1] * REGION_CHUNK_WIDTH)


# World Coords to Region Chunk ID
def w_coords_to_rcid(w_coords: glm.vec3) -> int:
    return rc_coords_to_rcid(
        w_coords_to_rc_coords(w_coords)
    )


###############################################

######### ID - Coordinate Translation #########
# World Region ID to World Region Coords
def wrid_to_wr_coords(wrid: int) -> glm.vec2:
    x = wrid % WORLD_REGION_WIDTH  # Get x coordinate
    z = wrid // WORLD_REGION_WIDTH  # Get z coordinate
    return glm.vec2(x, z)  # Return as glm.vec2 object


# World Region ID to World Coords
def wrid_to_w_coords(wrid: int) -> glm.vec3:
    return wr_coords_to_w_coords(wrid_to_wr_coords(wrid))


###############################################


_CHUNK_MASKS = tuple(1 << i for i in _REGION_CHUNK_ID_RANGE)


@dataclass
class Region(MCPhys, aabb=REGION_BASE_AABB):
    chunks: List[Optional[Chunk]] = field(default_factory=lambda: copy(REGION_CHUNK_NONE_LIST))

    _from_bytes: bool = False
    world: Optional[World] = None

    def __post_init__(self):
        super().__post_init__()
        if self.index is None:
            return

        self._renderable_dict: Dict[int, Chunk] = {}
        self._renderable_bitmask: int = 0
        self._renderable_list_cache: List[Chunk] = []
        self._num_renderable_cache: int = 0
        self._renderable_cache_valid: bool = False

        self._update_deque: deque[Chunk] = deque()
        self._awaiting_update: bool = False

        self.initialized: bool = False

    def _get_chunk_id(self, x: int, z: int) -> int:
        return (z * REGION_CHUNK_WIDTH) + x

    def initialize(self, region_pos: glm.vec3, world: Optional[World] = None):
        if world:
            if self.world:
                raise RuntimeError("Attempted to set world of a region already set in a world")
            self.world = world

        self.pos += region_pos

        if self._from_bytes:
            for chunk in self.chunks:
                if chunk is not None:
                    chunk.initialize(self)

        self.enqueue_update()
        self.initialized = True

    def enqueue_update(self):
        if self._awaiting_update:
            raise RuntimeError("Attempted to put Region into update queue whilst already awaiting an update")
        self._awaiting_update = True
        self.world.enqueue_region(self)

    def stack_update(self):
        if self._awaiting_update:
            raise RuntimeError("Attempted to put Region into update queue whilst already awaiting an update")
        self._awaiting_update = True
        self.world.stack_region(self)

    def enqueue_chunk(self, chunk: Chunk):
        self._update_deque.append(chunk)

    def stack_chunk(self, chunk: Chunk):
        self._update_deque.appendleft(chunk)

    def update(self):
        if not self._awaiting_update:
            return  # Shouldn't be possible
        while self._update_deque:
            chunk = self._update_deque.popleft()
            chunk.update()
            self.update_chunk_in_lists(chunk)
        self._awaiting_update = False

    def _is_index_out_of_bounds(self, chunk_index):
        if not -1 < chunk_index < REGION_NUM_CHUNKS:
            raise ValueError(f"Chunk Index: {chunk_index} is out of region bounds")

    def init_chunk(self, chunk_index: int):
        self._is_index_out_of_bounds(chunk_index)
        self.chunks[chunk_index] = Chunk(chunk_index, region=self)
        if not self._awaiting_update:
            self.enqueue_update()

    def get_chunk(self, chunk_index: int) -> Optional[Chunk]:
        # self._is_index_out_of_bounds(chunk_index)
        return self.chunks[chunk_index]

    def init_chunk_from_rc_coords(self, rc_coords: glm.vec2):
        try:
            self.init_chunk(rc_coords_to_rcid(rc_coords))
        except ValueError:
            raise ValueError(f"Region Chunk Coords: {rc_coords} are out of bounds")

    def get_chunk_from_rc_coords(self, rc_coords: Union[int, glm.vec2], z: Optional[int] = None) -> int:
        try:
            return self.get_chunk(rc_coords_to_rcid(rc_coords, z))
        except ValueError:
            if isinstance(rc_coords, int):
                raise ValueError(f"Region Chunk Coords: ({rc_coords}, {z}) are out of bounds")
            else:
                raise ValueError(f"Region Chunk Coords: {rc_coords} are out of bounds")

    def _is_chunk_solid(self, chunk: Chunk):
        return self._solid_bitmask & _CHUNK_MASKS[chunk.index]

    def _chunk_is_solid(self, chunk: Chunk):
        if not self._is_chunk_solid(chunk):  # Check if Chunk was not solid
            self._solid_dict[chunk.index] = chunk
            self._solid_bitmask |= _CHUNK_MASKS[chunk.index]
            self._solid_cache_valid = False

    def _chunk_is_not_solid(self, chunk: Chunk):
        if self._is_chunk_solid(chunk):  # Check if Chunk was solid
            self._solid_dict.pop(chunk.index)
            self._solid_bitmask &= ~_CHUNK_MASKS[chunk.index]
            self._solid_cache_valid = False

    def _is_chunk_renderable(self, chunk: Chunk):
        return self._renderable_bitmask & _CHUNK_MASKS[chunk.index]

    def _chunk_is_renderable(self, chunk: Chunk):
        if not self._is_chunk_renderable(chunk):  # Check if Chunk was not renderable
            self._renderable_dict[chunk.index] = chunk
            self._renderable_bitmask |= _CHUNK_MASKS[chunk.index]
            self._renderable_cache_valid = False

    def _chunk_is_not_renderable(self, chunk: Chunk):
        if self._is_chunk_renderable(chunk):  # Check if Chunk was renderable
            self._renderable_dict.pop(chunk.index)
            self._renderable_bitmask &= ~_CHUNK_MASKS[chunk.index]
            self._renderable_cache_valid = False

    def update_chunk_in_lists(self, chunk: Chunk):
        if chunk.is_renderable:
            self._chunk_is_renderable(chunk)
        else:
            self._chunk_is_not_renderable(chunk)

    def _validate_renderable_cache(self):
        self._renderable_list_cache = self._renderable_dict.values()
        self._num_renderable_cache = len(self._renderable_list_cache)
        self._renderable_cache_valid = True

    @property
    def renderable_chunks(self):
        if not self._renderable_cache_valid:
            self._validate_renderable_cache()
        return self._renderable_list_cache

    @property
    def num_renderable_chunks(self):
        if not self._renderable_cache_valid:
            self._validate_renderable_cache()
        return self._num_renderable_cache

    @property
    def non_none_chunks(self) -> Tuple[Chunk]:
        return tuple(filter(lambda x: x is not None, self.chunks))

    @property
    def is_renderable(self) -> bool:
        return bool(self._num_renderable_cache)

    def get_block(self, block_pos: glm.vec3) -> Optional[Block]:
        if self.bounds.intersectPoint(block_pos):
            local_pos = [int(i) for i in block_pos.xz - self.pos]
            chunk = self.get_chunk(self._get_chunk_id(*local_pos))
            if chunk is None:
                return None
            return chunk.get_block(block_pos)
        else:
            return self.world.get_block(block_pos)

    def serialize(self) -> bytes:
        # create byte arrays to store chunk locations, timestamps and payloads
        chunk_locations = bytearray(4096)  # 1024 entries * 4 bytes each
        chunk_timestamps = bytearray(4096)  # 1024 entries * 4 bytes each
        chunk_data = bytearray()  # store actual chunk data here

        # Keep track of current sector offset (starting after the header)
        current_sector_offset = 2  # Locations and timestamps use up 2 sectors (4096 bytes each)

        for chunk_index, chunk in enumerate(self.chunks):
            if isinstance(chunk, Chunk):
                compressed_data = chunk.serialize(compress=True)

                # Length of remaining chunk data bytes = compression type byte + compressed chunk
                comp_data_length = 1 + len(compressed_data)

                # Calculate how many 4096-byte sectors this compressed data will take
                sector_count = (comp_data_length + 4096) // 4096

                # Write location data (3-byte sector offset + 1-byte sector count)
                chunk_locations[chunk_index * 4:chunk_index * 4 + 3] = current_sector_offset.to_bytes(3,
                                                                                                      byteorder="big")
                chunk_locations[chunk_index * 4 + 3] = sector_count

                # Write timestamp (4 bytes Unix timestamp)
                chunk_timestamps[chunk_index * 4:chunk_index * 4 + 4] = chunk.timestamp.to_bytes(4, byteorder="big")

                # Add the chunk data to the payload
                chunk_data.extend(comp_data_length.to_bytes(4, byteorder="big"))  # Length of the compressed data
                chunk_data.append(compression_type)  # 1 byte for the compression type
                chunk_data.extend(compressed_data)  # Actual compressed data

                # Pad this chunk's data to the nearest 4096 bytes
                chunk_padding_length = (4096 - (len(chunk_data) % 4096)) % 4096
                chunk_data.extend(b'\x00' * chunk_padding_length)

                # Update the current sector offset by the number of sectors used
                current_sector_offset += sector_count

            else:
                # Mark missing chunks with zero in the locations and timestamps
                chunk_locations[chunk_index * 4:chunk_index * 4 + 4] = b"\x00\x00\x00\x00"
                chunk_timestamps[chunk_index * 4:chunk_index * 4 + 4] = b"\x00\x00\x00\x00"

        # Combine everything into the final region file bytes
        region_bytes = chunk_locations + chunk_timestamps + chunk_data
        return region_bytes

    @classmethod
    def deserialize(self, region_data: bytes, wrid: int) -> Region:
        # Parse the 4096-byte chunk locations table and 4096-byte timestamps table
        chunk_locations = region_data[:4096]
        chunk_timestamps = region_data[4096:8192]
        chunk_payloads = region_data[8192:]

        chunks = []

        for chunk_index in range(1024):  # There are up to 1024 chunks in a region (32x32 grid)
            # Extract the chunk location info (3 bytes offset, 1 byte sector count)
            location_entry = chunk_locations[chunk_index * 4: chunk_index * 4 + 4]
            sector_offset = int.from_bytes(location_entry[:3], byteorder="big")
            sector_count = location_entry[3]

            if sector_offset == 0 and sector_count == 0:
                # No data for this chunk (it's empty)
                chunks.append(None)
                continue

            # Extract the timestamp for the chunk (4 bytes)
            timestamp_entry = chunk_timestamps[chunk_index * 4: chunk_index * 4 + 4]
            timestamp = int.from_bytes(timestamp_entry, byteorder="big")

            # Calculate the starting position of the chunk data in the payload (sector_offset is in sectors of 4096 bytes)
            start = (sector_offset * 4096) - 8192  # Subtract 8192 bytes for location and timestamp tables
            end = start + (sector_count * 4096)

            # Extract the chunk data from the payload
            chunk_data = chunk_payloads[start:end]

            # First 4 bytes represent the length of the compressed data
            data_length = int.from_bytes(chunk_data[:4], byteorder="big")

            # Next 1 byte represents the compression type (usually 2 for GZip or Zlib)
            compression_type = chunk_data[4]

            # The remaining bytes are the compressed chunk data = start at index 5 and go to data_length - 1
            compressed_chunk_data = chunk_data[5:4 + data_length]

            # Create the Chunk object from the NBT data (assuming you have a method for this)
            chunk = Chunk.deserialize(compressed_chunk_data, compression_type)
            chunk.timestamp = timestamp  # Reassign the timestamp to the chunk

            # Append the chunk to the list
            chunks.append(chunk)
        return Region(wrid, chunks)


if __name__ == "__main__":
    def test():
        from World import World
        world_path = os.getcwd() + "\\region_test_world"

        region_before_save = Region(4)
        World.save_region_to_file(region_before_save, world_path)
        region_after_save = World.load_region_from_file(4, world_path)
        quit()


    num_tests = 1
    print(timeit(test, number=num_tests))

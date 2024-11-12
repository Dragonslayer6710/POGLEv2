from __future__ import annotations

import os
import pickle
import struct
from dataclasses import dataclass, field
from typing import Callable, BinaryIO

import nbtlib
import numpy as np

from Chunk import *

if TYPE_CHECKING:
    from World import World

PROCESS_POOL_SIZE = max(1, os.cpu_count() - 1)

@dataclass
class Region(MCPhys, aabb=REGION.AABB):
    chunks: List[List[Optional[Chunk]]] = field(default_factory=lambda: copy(REGION.NONE_LIST))

    _from_bytes: bool = False
    world: Optional[World] = None

    def __post_init__(self):
        if self.index == 0:
            self.index = glm.ivec2((WORLD.WIDTH + 1) // 2)
        super().__post_init__()
        if self.index is None:
            return

        self._update_deque: deque[Chunk] = deque()
        self._awaiting_update: bool = False

        self.initialized: bool = False

    def initialize(self, region_pos: glm.vec3, world: Optional[World] = None):
        if world:
            if self.world:
                raise RuntimeError("Attempted to set world of a region already set in a world")
            self.world = world

        self.pos += region_pos
        self.region_noise_map: Optional[np.ndarray] = None

        if self._from_bytes:
            for axis in self.chunks:
                for chunk in axis:
                    if chunk is not None:
                        chunk.initialize(self)
        else:
            self.region_noise_map = generate_region_noise(
                self.world.world_noise_map,
                self.index,
                self.pos
            )

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
            print(f"Chunk ({chunk.index.x}, {chunk.index.y}) Updated!")
        self._awaiting_update = False

    def _is_index_out_of_bounds(self, chunk_index: glm.ivec2):
        if not np.prod(glm.ivec2(-1) < chunk_index < glm.ivec2(REGION.NUM_CHUNKS)):
            raise ValueError(f"Chunk Index: {chunk_index} is out of region bounds")

    def init_chunk(self, rcid: glm.ivec2):
        self._is_index_out_of_bounds(rcid)
        self.chunks[rcid[1]][rcid[0]] = Chunk(rcid, region=self)
        if not self._awaiting_update:
            self.enqueue_update()

    def get_chunk(self, rcid: glm.ivec2) -> Optional[Chunk]:
        return self.chunks[rcid[1]][rcid[0]]

    @property
    def non_none_chunks(self) -> Tuple[Chunk]:
        return tuple(filter(lambda x: x is not None, self.chunks))

    def get_block(self, block_pos: glm.vec3) -> Optional[Block]:
        #if block_pos == glm.vec3(0.5, 0.5, -15.5):
        #print(block_pos)
        if self.bounds.intersect_point(block_pos):
            local_pos = w_to_rc(block_pos)
            chunk = self.get_chunk(local_pos)
            if chunk is None:
                return None
            block = chunk.get_block(block_pos)
        else:
            block = self.world.get_block(block_pos)
        if block is not None:
            if block_pos != block.pos:
                raise Exception("Wrong Block!")
        return block

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

    def get_shape(self):
        block_instances = []
        block_face_ids = []
        block_face_tex_ids = []
        block_face_tex_size_ids = []
        for z in REGION.WIDTH_RANGE:
            for x in REGION.WIDTH_RANGE:
                chunk = self.chunks[z][x]
                if chunk is None:
                    continue
                else:
                    block_instances.extend(chunk.block_instances)
                    block_face_ids.extend(chunk.block_face_ids)
                    block_face_tex_ids.extend(chunk.block_face_tex_ids)
                    block_face_tex_size_ids.extend(chunk.block_face_tex_size_ids)
        return DataLayout(
            [
                VertexAttribute("a_Position", TexQuad._positions),
                VertexAttribute("a_Alpha", [1.0, 1.0, 1.0, 1.0]),
                VertexAttribute("a_TexUV", TexQuad._tex_uvs),
                VertexAttribute("a_Model", block_instances, divisor=6),
                VertexAttribute("a_FaceID", np.concatenate(block_face_ids), divisor=1),
                VertexAttribute("a_FaceTexID", np.concatenate(block_face_tex_ids), divisor=1),
                VertexAttribute("a_FaceTexSizeID", np.concatenate(block_face_tex_size_ids), divisor=1),
            ]
        )
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

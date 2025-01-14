from __future__ import annotations

from MineClone.Chunk import *

if TYPE_CHECKING:
    from MineClone.World import World

PROCESS_POOL_SIZE = max(1, os.cpu_count() - 1)


@dataclass
class Region(MCPhys, aabb=REGION.AABB):
    chunks: List[List[Optional[Chunk]]] = field(default_factory=lambda: REGION.NONE_LIST.copy())

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

        self.neighbours: Dict[glm.vec2, Region] = {}
        self.chunk_query_cache: Dict[glm.vec2, Chunk] = {}

    def initialize(self, region_pos: glm.vec3, world: Optional[World] = None):
        if world:
            if self.world:
                raise RuntimeError("Attempted to set world of a region already set in a world")
            self.world = world

        self.pos += region_pos

        if self._from_bytes:
            for axis in self.chunks:
                for chunk in axis:
                    if chunk is not None:
                        chunk.initialize(self)
                        self.chunk_query_cache[chunk.pos.xz] = chunk

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
        chunk = self.chunks[rcid[1]][rcid[0]] = Chunk(rcid, region=self)
        if not self._awaiting_update:
            self.enqueue_update()

    def get_chunk(self, chunk_pos: Union[glm.ivec2, glm.vec2]) -> Optional[Chunk]:
        # If index vec2
        if isinstance(chunk_pos, glm.ivec2):
            return self.chunks[chunk_pos[1]][chunk_pos[0]]

        # Otherwise check cache
        cached_chunk = self.chunk_query_cache.get(chunk_pos)
        if cached_chunk is not None:  # get from cache
            return cached_chunk

        # Convert vec2 to index vec2
        local_pos = w_to_rc(chunk_pos)

        # If in bounds
        if self.bounds.intersect_point(chunk_pos, "y"):
            return self.get_chunk(local_pos)

        # Out of bounds so get from neighbour
        # Get Region index of neighbour
        neighbour_region_pos = w_to_wr(chunk_pos)

        # Get offset in terms of regions
        region_offset = neighbour_region_pos - self.index

        neighbour = self.neighbours.get(region_offset)
        if neighbour is None:
            neighbour = self.world.get_region(neighbour_region_pos)
            if neighbour is None:
                return None
            if neighbour.index == self.index:
                raise Exception("Self cannot be neighbour")
            self.neighbours[region_offset] = neighbour
        chunk = neighbour.get_chunk(chunk_pos)
        if chunk is not None:
            if chunk_pos != chunk.pos:
                raise Exception("Wrong Chunk!")
            self.chunk_query_cache[chunk_pos] = chunk
        return chunk

    def get_block(self, block_pos: glm.vec3) -> Optional[Block]:
        chunk = self.get_chunk(block_pos.xz)
        if chunk is not None:
            return chunk.get_block(block_pos)

    def query(self, bounds: AABB) -> List[Chunk]:
        chunks = []
        min_chunk = self.get_chunk(bounds.min.xz)
        if min_chunk is not None:
            chunks.append(min_chunk)
        else:
            min_chunk = self.get_chunk(bounds.pos.xz)
            if min_chunk is not None:
                chunks.append(min_chunk)
            else:
                max_chunk = self.get_chunk(bounds.max.xz)
                if max_chunk is not None:
                    chunks.append(max_chunk)
                else:
                    raise RuntimeError("No chunks found for bounds: " + str(bounds))
            return chunks

        max_chunk = self.get_chunk(bounds.max.xz)
        if max_chunk is not min_chunk:
            for x in range(min_chunk.index[0] + 1, max_chunk.index[0] + 1):
                for z in range(min_chunk.index[1] + 1, max_chunk.index[1]+1):
                    chunks.append(self.chunks[x][z])
        return chunks

    def query(self, bounds: AABB) -> List[Chunk]:
        min_point = glm.floor(bounds.min)
        max_point = glm.floor(bounds.max)
        min_is_max_point = min_point == max_point

        if self.bounds.contains(bounds):
            min_chunk = self.get_chunk(min_point.xz)
            if min_is_max_point:
                return [min_chunk]

            max_chunk = self.get_chunk(max_point.xz)
        elif self.bounds.intersect_point(bounds.min, "y"):
            min_chunk = self.get_chunk(min_point.xz)
            max_chunk = self.chunks[-1][-1]
        elif self.bounds.intersect_point(bounds.max, "y"):
            min_chunk = self.chunks[0][0]
            max_chunk = self.get_chunk(max_point.xz)
        else:
            raise RuntimeError(f"""
Bounds does not intersect Region:
    - Bounds:
        - Min: {bounds.min}
        - Pos: {bounds.pos}
        - Max: {bounds.max}

    - Chunk:
        - Min: {self.min}
        - Pos: {self.pos}
        - Max: {self.max}
""")

        if max_chunk is not min_chunk:
            chunks = []
            for z in range(min_chunk.index[1], max_chunk.index[1] + 1):
                for x in range(min_chunk.index[0], max_chunk.index[0] + 1):
                    chunks.append(self.chunks[z][x])
            return chunks
        else:
            return [min_chunk]

    @property
    def non_none_chunks(self) -> Tuple[Chunk]:
        return tuple(filter(lambda x: x is not None, self.chunks))

    def serialize(self) -> bytes:
        # create byte arrays to store chunk locations, timestamps and payloads
        chunk_locations = bytearray(4096)  # 1024 entries * 4 bytes each
        chunk_timestamps = bytearray(4096)  # 1024 entries * 4 bytes each
        chunk_data = bytearray()  # store actual chunk data here

        # Keep track of current sector offset (starting after the header)
        current_sector_offset = 2  # Locations and timestamps use up 2 sectors (4096 bytes each)

        chunk_index = 0
        for chunks_on_z in self.chunks:
            for chunk in chunks_on_z:
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

                chunk_index += 1
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
        z = -1
        for chunk_index in range(1024):  # There are up to 1024 chunks in a region (32x32 grid)
            # Extract the chunk location info (3 bytes offset, 1 byte sector count)
            location_entry = chunk_locations[chunk_index * 4: chunk_index * 4 + 4]
            sector_offset = int.from_bytes(location_entry[:3], byteorder="big")
            sector_count = location_entry[3]

            x = chunk_index % REGION.WIDTH
            if x == 0:
                z += 1
                chunks.append([])

            if sector_offset == 0 and sector_count == 0:
                # No data for this chunk (it's empty)
                chunks[z].append(None)
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
            chunks[z].append(chunk)
        return Region(wrid, chunks, _from_bytes=True)


    def get_shape(self) -> BlockShape:
        block_instances = []
        block_face_instances = []
        # block_face_ids = []
        # block_face_tex_ids = []
        # block_face_tex_size_ids = []
        for z in REGION.WIDTH_RANGE:
            for x in REGION.WIDTH_RANGE:
                chunk = self.chunks[z][x]
                if chunk is None:
                    continue
                else:
                    block_instances.extend(chunk.block_instance_data)
                    block_face_instances.extend(chunk.block_face_instance_data)
                    # block_face_ids.extend(chunk.block_face_ids)
                    # block_face_tex_ids.extend(chunk.block_face_tex_ids)
                    # block_face_tex_size_ids.extend(chunk.block_face_tex_size_ids)
        return BlockShape(
            block_instances,
            block_face_instances
        )

    def get_mesh(self) -> BlockShapeMesh:
        return BlockShapeMesh(self.get_shape())


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

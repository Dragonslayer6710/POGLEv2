from MineClone.Section import *


class Chunk(PhysicalBox):
    HEIGHT: int = 256
    HEIGHT_RANGE = range(HEIGHT)
    SIZE: glm.vec3 = glm.vec3(SECTION_WIDTH, HEIGHT, SECTION_WIDTH)
    SERIALIZED_FMT_STRS = ["ff"]
    SERIALIZED_HEADER_SIZE = np.sum([struct.calcsize(fmt_str) for fmt_str in SERIALIZED_FMT_STRS])
    SERIALIZED_SIZE = SERIALIZED_HEADER_SIZE + HEIGHT * SECTION_SERIALIZED_SIZE

    _NULL_SECTIONS: List[Section] = [NULL_SECTION() for _ in HEIGHT_RANGE]

    def __init__(self, pos: glm.vec2, sections: Optional[List[Section]] = None, multi_threaded: bool = False):
        super().__init__(AABB.from_pos_size(glm.vec3(pos[0], Chunk.HEIGHT / 2, pos[1]), Chunk.SIZE))
        if Chunk._NULL_SECTIONS is None:
            Chunk._NULL_SECTIONS = [None for _ in Chunk.HEIGHT_RANGE]

        self._solid_sections_cache = 0
        self._renderable_sections_cache = 0
        self._solid_cache_valid = False
        self._renderable_cache_valid = False

        self.sections: List[Section] = None

        self.octree: Octree = Octree(AABB.from_pos_size(self.pos, Chunk.SIZE), SECTION_SIZE)
        self._solid_sections: List[Section] = copy.deepcopy(Chunk._NULL_SECTIONS)
        self._renderable_sections: List[Section] = copy.deepcopy(Chunk._NULL_SECTIONS)
        if sections:
            self.sections: List[Section] = sections
            [self.update_section_in_lists(section) for section in self.sections]
        else:
            self.sections: List[Section] = [NULL_SECTION() for _ in Chunk.HEIGHT_RANGE]
            if multi_threaded:
                pass
            else:
                for section_in_chunk_index in Chunk.HEIGHT_RANGE:
                    section: Section = self.sections[section_in_chunk_index]
                    self.set_section(section)
                    self.update_section_in_lists(section)

    def _create_section(self, y: int, pos: glm.vec2) -> Section:
        return Section(y, pos)

    def update_section_in_lists(self, section: Section):
        index = section.section_in_chunk_index
        is_solid = bool(section.solid_blocks)
        is_renderable = bool(section.renderable_blocks)

        if not self._solid_sections[index]:
            if is_solid:
                self._solid_sections[index] = section
                self.octree.insert(section)
                self._solid_cache_valid = False  # Invalidate the cache
        else:
            if not is_solid:
                self._solid_sections[index] = None
                self._solid_cache_valid = False  # Invalidate the cache

        if not self._renderable_sections[index]:
            if is_renderable:
                self._renderable_sections[index] = section
                self._renderable_sections_cache = False  # Invalidate the cache
        else:
            if not is_renderable:
                self._renderable_sections[index] = None
                self._renderable_sections_cache = False  # Invalidate the cache

    def get_face_instances(self):
        return b"".join([section.get_face_instances() for section in self._renderable_sections])

    def get_section_instances(self):
        return b"".join(section.get_block_instances() for section in self._renderable_sections)

    @property
    def solid_sections(self) -> int:
        # Check if the cache is valid; if not, recalculate
        if not self._solid_cache_valid:
            self._solid_sections_cache = sum(1 for section in self._solid_sections if section)
            self._solid_cache_valid = True  # Mark the cache as valid
        return self._solid_sections_cache

    @property
    def renderable_sections(self) -> int:
        # Check if the cache is valid; if not, recalculate
        if not self._renderable_cache_valid:
            self._renderable_sections_cache = sum(1 for section in self._renderable_sections if section)
            self._renderable_cache_valid = True  # Mark the cache as valid
        return self._renderable_sections_cache

    def set_section(self, section: Section):
        """An example of how a method that sets sections should invalidate caches as necessary."""
        self.sections[section.section_in_chunk_index] = section
        self.update_section_in_lists(section)

    def clear_section(self, section_index: int):
        """An example of clearing a section and invalidating caches."""
        self.sections[section_index] = None
        self._solid_sections[section_index] = None
        self._renderable_sections[section_index] = None
        self._solid_cache_valid = False  # Invalidate solid sections cache
        self._renderable_cache_valid = False  # Invalidate renderable sections cache

    def serialize(self) -> bytes:
        pos_bytes = struct.pack("ff", *self.pos.xz)
        chunk_header_bytes = bytearray(pos_bytes)

        for section in self.sections:
            chunk_header_bytes.extend(section.serialize())

        return bytes(chunk_header_bytes)

    @classmethod
    def deserialize(cls, packed_data: bytes) -> Section:
        # Unpack the section header
        pos = glm.vec2(struct.unpack("ff", packed_data[:8]))

        sections = copy.deepcopy(Chunk._NULL_SECTIONS)
        # The remaining bytes are section data so iterate
        for i in Chunk.HEIGHT_RANGE:
            # Extract the section bytes
            start = Chunk.SERIALIZED_HEADER_SIZE + i * SECTION_SERIALIZED_SIZE
            end = start + SECTION_SERIALIZED_SIZE
            section_bytes = packed_data[start:end]

            # Deserialize the section
            sections[i] = Section.deserialize(section_bytes)

        # Create a new instance of Chunk using the unpacked values and deserialized sections
        return cls(
            pos=pos,
            sections=sections
        )

    def __repr__(self):
        return f"Chunk(solid_sections: {self.solid_sections}, renderable_sections:{self.renderable_sections})"


class ChunkRange(PhysicalBox):
    def __init__(self, pos: glm.vec2, size: glm.vec2):
        super().__init__(AABB.from_pos_size(pos, size))


if __name__ == "__main__":
    import time

    # Test synchronous section generation
    print("Testing Chunk generation...")
    start_time = time.time()
    syncro_chunk = Chunk(pos=glm.vec2(0, 0), multi_threaded=False)
    end_time = time.time()
    # serialized_syncro_chunk = syncro_chunk.serialize()
    # deserialized_syncro_chunk = Chunk.deserialize(serialized_syncro_chunk)
    print("Pre-Serialized Chunk:", syncro_chunk)
    # print("Deserialized Synchronous Chunk:", deserialized_syncro_chunk)
    print(f"Standard generation took {end_time - start_time:.4f} seconds")

    # Test multi_threaded section generation
    print("Testing Multi_Threaded Chunk generation...")
    start_time = time.time()
    multi_threaded_chunk = Chunk(pos=glm.vec2(0, 0), multi_threaded=True)
    end_time = time.time()
    # serialized_multi_threaded_chunk = multi_threaded_chunk.serialize()
    # deserialized_multi_threaded_chunk = Chunk.deserialize(serialized_multi_threaded_chunk)
    print("Pre-Serialized Multi_Threaded Chunk:", multi_threaded_chunk)
    # print("Deserialized Multi_Threaded Chunk:", deserialized_multi_threaded_chunk)
    print(f"Multi-Threaded generation took {end_time - start_time:.4f} seconds")

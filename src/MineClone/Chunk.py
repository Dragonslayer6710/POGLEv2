from __future__ import annotations

from Section import *

if TYPE_CHECKING:
    from Region import Region

CHUNK_BLOCK_HEIGHT: int = 256
CHUNK_BLOCK_HEIGHT_HALF: int = CHUNK_BLOCK_HEIGHT // 2
CHUNK_NUM_SECTIONS: int = CHUNK_BLOCK_HEIGHT
CHUNK_HEIGHT_RANGE: range = range(CHUNK_NUM_SECTIONS)
CHUNK_SECTION_RANGE: range = CHUNK_HEIGHT_RANGE
CHUNK_SECTION_RANGE_LIST: List[int] = list(CHUNK_SECTION_RANGE)
CHUNK_NUM_BLOCKS: int = CHUNK_NUM_SECTIONS * SECTION_NUM_BLOCKS
CHUNK_BLOCK_EXTENTS: glm.vec3 = glm.vec3(SECTION_BLOCK_WIDTH, CHUNK_BLOCK_HEIGHT, SECTION_BLOCK_WIDTH)
CHUNK_BLOCK_EXTENTS_HALF: glm.vec3 = CHUNK_BLOCK_EXTENTS / 2

# Precompute offsets for each chunk
BLOCK_OFFSETS_PER_SECTION: List[int] = [
    SECTION_NUM_BLOCKS * chunk_id for chunk_id in CHUNK_SECTION_RANGE
]

# Use itertools.product to generate Cartesian product of x and z values
SECTION_POSITIONS_IN_CHUNK = [
    glm.vec3(SECTION_BLOCK_WIDTH * x, 0,
             SECTION_BLOCK_WIDTH * z) - CHUNK_BLOCK_EXTENTS_HALF + SECTION_BLOCK_EXTENTS_HALF
    for x, z in itertools.product(SECTION_BLOCK_WIDTH_RANGE, repeat=2)
]

CHUNK_BASE_CENTRE: glm.vec3 = glm.vec3(0, CHUNK_BLOCK_HEIGHT_HALF, 0)

CHUNK_BASE_AABB: AABB = AABB.from_pos_size(
    CHUNK_BASE_CENTRE,
    CHUNK_BLOCK_EXTENTS
)

NULL_SECTIONS = [None] * CHUNK_NUM_SECTIONS

SECTION_AABBS_IN_CHUNK: List[AABB] = [
    copy(SECTION_BASE_AABB) for pos in SECTION_POSITIONS_IN_CHUNK
]

class Chunk(PhysicalBox):
    def __init__(self, id: int, aabb: AABB = CHUNK_BASE_AABB, section_offset: int = 0, region: Region = None,
                 chunk_data: Optional[Tuple[int, int]] = None):
        super().__init__(aabb)
        section_aabbs = [copy(aabb) for aabb in SECTION_AABBS_IN_CHUNK]

        self.id: int = id

        self.region: Region = region

        self.octree: Octree = Octree(self.bounds)
        self.sections: List[Optional[Section]] = []
        if chunk_data is None:
            self._solid_sections: int = 0
            self._solid_cache: List[Optional[int]] = NULL_SECTIONS.copy()
            self._solid_cache_valid: bool = False

            self._renderable_sections: int = 0
            self._renderable_cache: List[Optional[int]] = NULL_SECTIONS.copy()
            self._renderable_cache_valid: bool = False

            # self.section_range = list(range(section_offset, section_offset + CHUNK_NUM_SECTIONS))

            self.section_states = NULL_SECTIONS.copy()
            self.sections: List[Optional[Section]] = NULL_SECTIONS.copy()
            for section_id in CHUNK_SECTION_RANGE:
                section = self.sections[section_id] = Section(
                    section_id,
                    section_aabbs[section_id],
                    section_id * SECTION_NUM_BLOCKS,
                    self
                )
                self.update_section_in_lists(section)
        else:
            self.section_range, self._solid_sections, self._renderable_sections = chunk_data

    def is_section_solid(self, section_id: int):
        return (self._solid_sections & (1 << section_id)) != 0

    def section_is_solid(self, section: Section):
        if not self.is_section_solid(section.id):  # if wasn't solid
            self._solid_sections |= (1 << section.id)
            self._solid_cache[section.id] = section
            self.octree.insert(section)

    def section_is_not_solid(self, section: Section):
        if self.is_section_solid(section.id):  # if was solid
            self._solid_sections &= ~(1 << section.id)
            self._solid_cache[section.id] = None
            self.octree.remove(section)

    def is_section_renderable(self, section_id: int):
        return (self._renderable_sections & (1 << section_id)) != 0

    def section_is_renderable(self, section: Section):
        if not self.is_section_renderable(section.id):  # if wasn't renderable
            self._renderable_sections |= (1 << section.id)
            self._renderable_cache[section.id] = section
            self.octree.insert(section)

    def section_is_not_renderable(self, section: Section):
        if self.is_section_renderable(section.id):  # if was renderable
            self._renderable_sections &= ~(1 << section.id)
            self._renderable_cache[section.id] = None
            self.octree.remove(section)

    def update_section_in_lists(self, section: Section):
        if section.is_solid:
            self.section_is_solid(section)
        else:
            self.section_is_not_solid(section)

        if section.is_renderable:
            self.section_is_renderable(section)
        else:
            self.section_is_not_renderable(section)

    def get_face_instances(self):
        return b"".join([section.get_face_instances() for section in self.renderable_sections if section])

    def get_block_instances(self):
        return b"".join([section.get_block_instances() for section in self.renderable_sections if section])

    @property
    def solid_sections(self):
        if not self._solid_cache_valid:
            self._solid_cache = [section if self.is_section_solid(section) else None for section in self.sections]
            self._solid_cache_valid = True
        return self._solid_cache

    @property
    def is_solid(self):
        return bool(self._solid_sections)

    @property
    def renderable_sections(self):
        if not self._renderable_cache_valid:
            self._renderable_cache = [section if self.is_section_renderable(section) else None for section in
                                      self.sections]
            self._renderable_cache_valid = True
        return self._renderable_cache

    @property
    def is_renderable(self):
        return bool(self._renderable_sections)

    def serialize(self) -> Tuple[
        bytes, bytes, bytes, bytes]:  # chunk_bytes, section_bytes, block_position_bytes, block_state_bytes
        chunk_bytes = (
                struct.pack("fff", self.pos.x, self.pos.y, self.pos.z) +
                struct.pack("<H", self._solid_sections) +
                struct.pack("<H", self._renderable_sections)  # +
            # struct.pack("<I", self.section_range[0]) +
            # struct.pack("<I", self.section_range[1])
        )
        section_bytes, block_position_bytes, block_state_bytes = Section.serialize_array(self.sections)

        return chunk_bytes, section_bytes, block_position_bytes, block_state_bytes

    @classmethod
    def deserialize(cls, id: int, chunk_bytes: bytes, section_bytes: bytes, block_position_bytes: bytes,
                    block_state_bytes: bytes, region: Optional['Region'] = None) -> Chunk:
        chunk_pos = glm.vec3(*struct.unpack("fff", chunk_bytes[:12]))
        chunk_data = (
            struct.unpack("<H", chunk_bytes[12:14])[0],
            struct.unpack("<H", chunk_bytes[14:])[0],
            # struct.unpack("<I", chunk_bytes[16:20]),
            # struct.unpack("<I", chunk_bytes[20:])
        )
        chunk = cls(
            id=id,
            aabb=AABB.from_pos_size(chunk_pos, CHUNK_BLOCK_EXTENTS),
            region=region,
            chunk_data=chunk_data,
        )

        sections, blocks = Section.deserialize_array(
            CHUNK_SECTION_RANGE_LIST,
            (section_bytes, block_position_bytes, block_state_bytes),
            chunk
        )
        chunk.sections = sections

        return chunk

    @staticmethod
    def serialize_array(chunks: List[Chunk]) -> Tuple[bytes, bytes, bytes, bytes]:
        chunk_bytes = b""
        section_bytes = b""
        block_position_bytes = b""
        block_state_bytes = b""
        for chunk in chunks:
            chunk_bytes += chunk.serialize()[0]
            section_bytes += chunk.serialize()[1]
            block_position_bytes += chunk.serialize()[2]
            block_state_bytes += chunk.serialize()[3]
        return chunk_bytes, section_bytes, block_position_bytes, block_state_bytes

    @staticmethod
    def deserialize_array(ids: List[int], packed_data: Tuple[bytes, bytes, bytes, bytes], region: Optional['Region'] = None) -> Tuple[List[Chunk], List[Section], List[Block]]:
        chunks = []
        sections = []
        blocks = []
        for id, chunk_bytes, section_bytes, block_position_bytes, block_state_bytes in zip(ids, packed_data[0], packed_data[1], packed_data[2], packed_data[3]):
            chunk, sections, blocks = Chunk.deserialize(id, chunk_bytes, section_bytes, block_position_bytes, block_state_bytes, region)
            chunks.append(chunk)
            sections.extend(sections)
            blocks.extend(blocks)

        return chunks, sections, blocks

    def __repr__(self):
        return f"Chunk(id={self.id}, pos={self.pos}, sections={self.sections})"

    def __str__(self):
        return self.__repr__()



if __name__ == "__main__":
    def test():
        Chunk(0)
    num_tests = 1
    filename = "chunk"
    cProfile.run(f"[test() for _ in range({num_tests})]", f"{filename}.prof")

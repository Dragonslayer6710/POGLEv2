from Section import *

CHUNK_BLOCK_HEIGHT: int = 256
CHUNK_BLOCK_HEIGHT_HALF: int = CHUNK_BLOCK_HEIGHT / 2
CHUNK_NUM_SECTIONS: int = CHUNK_BLOCK_HEIGHT
CHUNK_HEIGHT_RANGE: range = range(CHUNK_NUM_SECTIONS)
CHUNK_SECTION_RANGE: range = CHUNK_HEIGHT_RANGE
CHUNK_NUM_BLOCKS: int = CHUNK_NUM_SECTIONS * SECTION_NUM_BLOCKS
CHUNK_BLOCK_EXTENTS: glm.vec3 = glm.vec3(SECTION_BLOCK_WIDTH, CHUNK_BLOCK_HEIGHT, SECTION_BLOCK_WIDTH)
CHUNK_BLOCK_EXTENTS_HALF: glm.vec3 = CHUNK_BLOCK_EXTENTS / 2

# Precompute offsets for each chunk
BLOCK_OFFSETS_PER_SECTION: List[int] = [
    SECTION_NUM_BLOCKS * chunk_id for chunk_id in CHUNK_SECTION_RANGE
]

import itertools
# Use itertools.product to generate Cartesian product of x and z values
SECTION_POSITIONS_IN_CHUNK = [
    glm.vec3(SECTION_BLOCK_WIDTH * x, 0, SECTION_BLOCK_WIDTH * z) - CHUNK_BLOCK_EXTENTS_HALF + SECTION_BLOCK_EXTENTS_HALF
    for x, z in itertools.product(CHUNK_SECTION_RANGE, repeat=2)
]

CHUNK_BASE_CENTRE: glm.vec3 = glm.vec3(0, CHUNK_BLOCK_HEIGHT_HALF, 0)

CHUNK_BASE_AABB: AABB = AABB.from_pos_size(
    CHUNK_BASE_CENTRE,
    CHUNK_BLOCK_EXTENTS
)


class Chunk(PhysicalBox):
    def __init__(self, aabb: AABB = CHUNK_BASE_AABB, section_offset: int = 0, region: 'Region' = None):
        super().__init__(aabb)
        from Region import Region
        self.region: Region = region

        block_base_offset = section_offset * SECTION_NUM_BLOCKS
        self.sections: List[Optional[Section]] = [None] * CHUNK_NUM_SECTIONS
        for section_id, block_offset in enumerate(BLOCK_OFFSETS_PER_SECTION):
            section_pos = SECTION_POSITIONS_IN_CHUNK[section_id] + self.pos
            self.sections[section_id] = Section(
                AABB.from_pos_size(section_pos, SECTION_BLOCK_EXTENTS),
                block_base_offset + block_offset,
                self
            )


if __name__ == "__main__":
    # Example usage
    chunk = Chunk()
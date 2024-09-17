from Block import *
from Block import _block_state_cache

SECTION_BLOCK_WIDTH: int = 16
SECTION_BLOCK_WIDTH_RANGE: range = range(SECTION_BLOCK_WIDTH)
SECTION_BLOCK_EXTENTS: glm.vec3 = glm.vec3(SECTION_BLOCK_WIDTH, 1, SECTION_BLOCK_WIDTH)
SECTION_BLOCK_EXTENTS_HALF: glm.vec3 = SECTION_BLOCK_EXTENTS / 2
SECTION_NUM_BLOCKS: int = SECTION_BLOCK_WIDTH ** 2
SECTION_BLOCK_RANGE: range = range(SECTION_NUM_BLOCKS)

import itertools
BLOCK_POSITIONS_IN_SECTION = [
    glm.vec3(x, 0, z) - SECTION_BLOCK_EXTENTS_HALF + BLOCK_EXTENTS_HALF
    for x, z in itertools.product(SECTION_BLOCK_WIDTH_RANGE, repeat=2)
]
NULL_BLOCK_STATES = [NULL_BLOCK_STATE] * SECTION_NUM_BLOCKS
NULL_BLOCKS = [None] * SECTION_NUM_BLOCKS

SECTION_BASE_CENTRE: glm.vec3 = glm.vec3(0, 0.5, 0)

SECTION_BASE_AABB: AABB = AABB.from_pos_size(
    SECTION_BASE_CENTRE,
    SECTION_BLOCK_EXTENTS
)

class Section(PhysicalBox):
    def __init__(self, aabb: AABB = SECTION_BASE_AABB, block_start: int = 0, chunk: 'Chunk' = None):
        super().__init__(aabb)
        from Chunk import Chunk
        self.chunk: Chunk = chunk

        self.block_range = range(block_start, block_start + SECTION_NUM_BLOCKS)
        self.block_states = NULL_BLOCK_STATES.copy()
        self.blocks: List[Optional[Block]] = NULL_BLOCKS.copy()
        for block_id in SECTION_BLOCK_RANGE:
            self.blocks[block_id] = Block(
                BLOCK_POSITIONS_IN_SECTION[block_id] + self.pos,
                self.block_states[block_id],
                self
            )


if __name__ == "__main__":
    # Example usage
    section_a = Section()
    section_b = Section(256)
    section_b.block_states[0] = _block_state_cache[BlockID.Stone]
    print(id(section_b.block_states[0]))
    print(id(section_a.block_states[0]))
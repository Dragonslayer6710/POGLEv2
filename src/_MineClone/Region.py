from Chunk import *

REGION_CHUNK_WIDTH: int = 32
REGION_CHUNK_WIDTH_RANGE: range = range(REGION_CHUNK_WIDTH)
REGION_CHUNK_EXTENTS: glm.vec2(REGION_CHUNK_WIDTH)
REGION_NUM_CHUNKS: int = REGION_CHUNK_WIDTH ** 2
REGION_CHUNK_RANGE: range = range(REGION_NUM_CHUNKS)
REGION_NUM_BLOCKS: int = REGION_NUM_CHUNKS * CHUNK_NUM_BLOCKS
REGION_BLOCK_WIDTH: int = REGION_CHUNK_WIDTH * SECTION_BLOCK_WIDTH
REGION_BLOCK_WIDTH_HALF: int = REGION_BLOCK_WIDTH / 2
REGION_BLOCK_EXTENTS: glm.vec3 = glm.vec3(REGION_BLOCK_WIDTH, CHUNK_BLOCK_HEIGHT, REGION_BLOCK_WIDTH)
REGION_BLOCK_EXTENTS_HALF: glm.vec3 = REGION_BLOCK_EXTENTS / 2

# Precompute offsets for each chunk
SECTION_OFFSETS_PER_CHUNK: List[int] = [
    CHUNK_NUM_SECTIONS * chunk_id for chunk_id in REGION_CHUNK_RANGE
]

# Use itertools.product to generate Cartesian product of x and z values
#precomputed_positions = list(itertools.product(REGION_CHUNK_RANGE, repeat=2))
CHUNK_POSITIONS_IN_REGION = [
    glm.vec3(
        SECTION_BLOCK_WIDTH * x,
        0,
        SECTION_BLOCK_WIDTH * z
    ) - REGION_BLOCK_EXTENTS_HALF + CHUNK_BLOCK_EXTENTS_HALF
    for x, z in itertools.product(REGION_CHUNK_WIDTH_RANGE, repeat=2)
]

REGION_BASE_CENTRE: glm.vec3 = glm.vec3(0, CHUNK_BLOCK_HEIGHT_HALF, 0)

REGION_BASE_AABB: AABB = AABB.from_pos_size(
    REGION_BASE_CENTRE,
    REGION_BLOCK_EXTENTS
)

NULL_CHUNKS = [None] * REGION_NUM_CHUNKS

class Region(PhysicalBox):
    def __init__(self, aabb: AABB = REGION_BASE_AABB, chunk_offset: int = 0, world: 'World' = None):
        super().__init__(aabb)
        from World import World
        self.world: World = world
        self.chunk_positions = CHUNK_POSITIONS_IN_REGION.copy()

        section_base_offset = chunk_offset * CHUNK_NUM_SECTIONS
        self.section_offsets = [
            section_base_offset + section_offset for section_offset in SECTION_OFFSETS_PER_CHUNK
        ]
        self.chunks: List[Optional[Chunk]] = NULL_CHUNKS.copy()

    def create_chunk(self, chunk_id):
        self.chunks[chunk_id] = Chunk(
            AABB.from_pos_size(
                self.chunk_positions[chunk_id] + self.pos,
                CHUNK_BLOCK_EXTENTS
            ),
            self.section_offsets[chunk_id],
            self
        )

    def get_chunk(self, chunk_id):
        return self.chunks[chunk_id]



if __name__ == "__main__":
    # Example usage
    region = Region()
    region.create_chunk(0)
    region.create_chunk(1023)
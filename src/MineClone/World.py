from __future__ import annotations
from Region import *

WORLD_REGION_WIDTH_FACTOR: int = 3
WORLD_REGION_WIDTH_BASE: int = 1
WORLD_REGION_WIDTH: int = WORLD_REGION_WIDTH_FACTOR * WORLD_REGION_WIDTH_BASE
WORLD_REGION_WIDTH_HALF: int = WORLD_REGION_WIDTH // 2
WORLD_REGION_WIDTH_RANGE: range = range(-WORLD_REGION_WIDTH_HALF, WORLD_REGION_WIDTH_HALF + 1)
WORLD_REGION_EXTENTS: glm.vec2 = glm.vec2(WORLD_REGION_WIDTH)
WORLD_NUM_REGIONS: int = WORLD_REGION_WIDTH ** 2
WORLD_REGION_RANGE: range = range(WORLD_NUM_REGIONS)
WORLD_NUM_CHUNKS: int = WORLD_NUM_REGIONS * REGION_NUM_CHUNKS
WORLD_BLOCK_WIDTH: int = WORLD_REGION_WIDTH * REGION_BLOCK_WIDTH
WORLD_BLOCK_WIDTH_HALF: int = WORLD_BLOCK_WIDTH // 2
WORLD_BLOCK_WIDTH_RANGE: range = range(-WORLD_BLOCK_WIDTH_HALF, WORLD_BLOCK_WIDTH_HALF)
WORLD_NUM_BLOCKS: int = WORLD_NUM_REGIONS * REGION_NUM_BLOCKS
WORLD_BLOCK_RANGE: range = range(WORLD_NUM_BLOCKS)
WORLD_BLOCK_EXTENTS: glm.vec3 = glm.vec3(WORLD_BLOCK_WIDTH, CHUNK_BLOCK_HEIGHT, WORLD_BLOCK_WIDTH)
WORLD_BLOCK_EXTENTS_HALF: glm.vec3 = WORLD_BLOCK_EXTENTS / 2

# Precompute offsets for each region
CHUNK_OFFSETS_PER_REGION: List[int] = [
    REGION_NUM_CHUNKS * region_id for region_id in WORLD_REGION_RANGE
]

# Precompute AABBs for each region
REGION_AABBS: List[AABB] = [
    AABB.from_pos_size(
        glm.vec3(
            REGION_BLOCK_WIDTH * x,
            CHUNK_BLOCK_HEIGHT_HALF,
            REGION_BLOCK_WIDTH * z
        ),
        REGION_BLOCK_EXTENTS
    ) for x, z in itertools.product(WORLD_REGION_WIDTH_RANGE, repeat=2)
]

WORLD_BLOCK_CENTRE: glm.vec3 = glm.vec3(0, CHUNK_BLOCK_HEIGHT_HALF, 0)
class World(PhysicalBox):
    def __init__(self) -> World:
        super().__init__(AABB.from_pos_size(WORLD_BLOCK_CENTRE, WORLD_BLOCK_EXTENTS))
        self.chunk_offsets: List[int] = [
            chunk_start for chunk_start in CHUNK_OFFSETS_PER_REGION
        ]
        self.regions: List[Optional[Region]] = [None] * WORLD_NUM_REGIONS
        self.blocks: Dict[int: Block] = {}

    def create_region(self, region_id: int):
        self.regions[region_id] = Region(
            region_id,
            REGION_AABBS[region_id],
            self.chunk_offsets[region_id],
            self
        )

    def _validate_chunk_region_ids(self,
                                   region_id: Optional[int] = None,
                                   chunk_id: Optional[int] = None) -> Tuple[int, int]:
        # If both are None, raise an error
        if region_id is None and chunk_id is None:
            raise ValueError("One of Region ID or Chunk ID must be provided")

        # Handle the case where only chunk_id is provided
        if chunk_id is not None:
            # Normalize chunk_id to be within the correct range, accounting for negative values
            chunk_id = chunk_id % WORLD_NUM_CHUNKS
            region_id = chunk_id // REGION_NUM_CHUNKS
            chunk_id = chunk_id % REGION_NUM_CHUNKS

            # Check if region_id is out of bounds
            if region_id >= WORLD_NUM_REGIONS:
                raise ValueError(f"Calculated region_id {region_id} is out of bounds")
        elif region_id is not None:
            # Normalize region_id and chunk_id to be within bounds, accounting for negative values
            region_id = region_id % WORLD_NUM_REGIONS
            chunk_id = chunk_id % REGION_NUM_CHUNKS
        else:
            raise ValueError("Invalid combination of region_id and chunk_id provided")
        return region_id, chunk_id

    def create_chunk(self, region_id: Optional[int] = None, chunk_id: Optional[int] = None):
        region_id, chunk_id = self._validate_chunk_region_ids(region_id, chunk_id)
        if self.regions[region_id] is None:
            self.create_region(region_id)
        self.regions[region_id].create_chunk(chunk_id)

    def get_chunk(self, region_id: Optional[int] = None, chunk_id: Optional[int] = None) -> Chunk:
        region_id, chunk_id = self._validate_chunk_region_ids(region_id, chunk_id)
        region = self.regions[region_id]
        if region:
            chunk = region.get_chunk(chunk_id)
            if chunk:
                return chunk
            else:
                raise RuntimeError("Region Exists but chunk doesn't")
        raise RuntimeError("Region Does not exist")

if __name__ == "__main__":
    world = World()
    world.create_chunk(chunk_id=0)
    world.create_chunk(chunk_id=1023)
    world.create_chunk(chunk_id=1024)

    world.create_chunk(chunk_id=-1024)
    world.create_chunk(chunk_id=-1)

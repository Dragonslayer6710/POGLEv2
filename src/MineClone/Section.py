from __future__ import annotations
from MineClone.Block import *

SECTION_WIDTH: int = 16
SECTION_WIDTH_RANGE = range(SECTION_WIDTH)
SECTION_NUM_BLOCKS = SECTION_WIDTH ** 2
BLOCK_IN_SECTION_INDEX_RANGE = list(range(SECTION_NUM_BLOCKS))
BLOCK_IN_SECTION_OFFSETS = [glm.vec3(x, 0, z) for x in SECTION_WIDTH_RANGE for z in SECTION_WIDTH_RANGE]
SECTION_SIZE = glm.vec3(SECTION_WIDTH, 1, SECTION_WIDTH)
SECTION_NULL_BLOCKS: List[Optional[Block]] = [NULL_BLOCK() for _ in BLOCK_IN_SECTION_INDEX_RANGE]
SECTION_SERIALIZED_FMT_STRS = ["H", "ff"]
SECTION_SERIALIZED_HEADER_SIZE = np.sum([struct.calcsize(fmt_str) for fmt_str in SECTION_SERIALIZED_FMT_STRS])
SECTION_SERIALIZED_SIZE = SECTION_SERIALIZED_HEADER_SIZE + SECTION_NUM_BLOCKS * Block.SERIALIZED_SIZE
SECTION_BLOCK_BYTES_START_LIST: List[List[int]] = [SECTION_SERIALIZED_HEADER_SIZE + i * Block.SERIALIZED_SIZE for i in
                                                   range(SECTION_NUM_BLOCKS)]
SECTION_BLOCK_BYTES_END_LIST: List[List[int]] = [SECTION_BLOCK_BYTES_START_LIST[i] + Block.SERIALIZED_SIZE for i in range(SECTION_NUM_BLOCKS)]

class Section(PhysicalBox):
    def __init__(self, sectionPos: glm.vec2 = glm.vec2(), section_in_chunk_index: int = 0, blocks: Optional[List[BlockID]] = None):
        super().__init__(AABB.from_pos_size(glm.vec3(sectionPos[0], 0, sectionPos[1])))
        self.section_in_chunk_index: int = 0

        self._solid_blocks_cache = 0
        self._renderable_blocks_cache = 0
        self._solid_cache_valid = False
        self._renderable_cache_valid = False

        self.quadtree: QuadTree = None

        self._solid_blocks: List[Optional[Block]] = [None] * SECTION_NUM_BLOCKS
        self._renderable_blocks: List[Optional[Block]] = [None] * SECTION_NUM_BLOCKS

        self.blocks: List[Optional[Block]] = []

        if blocks:
            self.blocks = blocks
        else:
            self.blocks = [NULL_BLOCK() for _ in BLOCK_IN_SECTION_INDEX_RANGE]
        self.set_section_in_chunk(section_in_chunk_index)

    def _init_block(self, block_in_section_index: int) -> Block:
        block = self.blocks[block_in_section_index]
        offset = BLOCK_IN_SECTION_OFFSETS[block_in_section_index]
        block.set_block_in_chunk(self.section_in_chunk_index + block_in_section_index, self.pos + offset)
        return block

    def update_block_in_lists(self, block: Block):
        index = block.block_in_section_index
        is_solid = block.is_solid
        is_renderable = block.is_renderable

        if not self._solid_blocks[index]:
            if is_solid:
                self._solid_blocks[index] = block
                self.quadtree.insert(block)
                self._solid_cache_valid = False
        else:
            if not is_solid:
                self._solid_blocks[index] = None
                self._solid_cache_valid = False

        if not self._renderable_blocks[index]:
            if is_renderable:
                self._renderable_blocks[index] = block
                self._renderable_blocks_cache = False
        else:
            if not is_renderable:
                self._renderable_blocks[index] = None
                self._renderable_blocks_cache = False

    def set_section_in_chunk(self, sectionY: int):
        self.section_in_chunk_index = sectionY
        self.bounds = AABB.from_pos_size(glm.vec3(self.pos.x, sectionY, self.pos.z), SECTION_SIZE)
        self.quadtree = QuadTree.XZ(self.bounds)
        for block in self.blocks:
            self._init_block(block.block_in_section_index)
            self.update_block_in_lists(block)

    def get_face_instances(self):
        return b"".join([block.get_face_instances() for block in self._renderable_blocks if block])

    def get_block_instances(self):
        return b"".join(block.get_instance() for block in self._renderable_blocks if block)

    @property
    def solid_blocks(self) -> int:
        if not self._solid_cache_valid:
            self._solid_blocks_cache = sum(1 for block in self._solid_blocks if block)
            self._solid_cache_valid = True
        return self._solid_blocks_cache

    @property
    def renderable_blocks(self) -> int:
        if not self._renderable_cache_valid:
            self._renderable_blocks_cache = sum(1 for block in self._renderable_blocks if block)
            self._renderable_cache_valid = True
        return self._renderable_blocks_cache

    def set_block(self, block_in_section_index: int, block_id: BlockID):
        if BlockID.Air == block_id:
            self.clear_block(block_in_section_index)
            return
        self.blocks[block_in_section_index].set_block(block_id)
        self.update_block_in_lists(self.blocks[block_in_section_index])

    def clear_block(self, block_in_section_index: int):
        self.blocks[block_in_section_index].set_block(BlockID.Air)
        self._solid_blocks[block_in_section_index] = None
        self._renderable_blocks[block_in_section_index] = None
        self._solid_cache_valid = False
        self._renderable_cache_valid = False

    def serialize(self) -> bytes:
        index_bytes = struct.pack("H", self.section_in_chunk_index)
        pos_bytes = struct.pack("ff", *self.pos.xz)
        block_header_bytes = index_bytes + pos_bytes

        block_bytes = b"".join(block.serialize() for block in self.blocks if block)
        return block_header_bytes + block_bytes

    @classmethod
    def deserialize(cls, packed_data: bytes) -> Section:
        section_in_chunk_index = struct.unpack("H", packed_data[:2])[0]
        section_pos = glm.vec2(struct.unpack("ff", packed_data[2:10]))

        blocks = [None for _ in BLOCK_IN_SECTION_INDEX_RANGE]
        for i in range(SECTION_NUM_BLOCKS):
            start = SECTION_BLOCK_BYTES_START_LIST[i]
            end = SECTION_BLOCK_BYTES_END_LIST[i]
            block_bytes = packed_data[start:end]
            blocks[i] = Block.deserialize(block_bytes)

        return cls(
            sectionPos=section_pos,
            section_in_chunk_index=section_in_chunk_index,
            blocks=blocks
        )

    def __repr__(self):
        return f"Section(block_index: {self.section_in_chunk_index}, solid_blocks: {self.solid_blocks}, renderable_blocks:{self.renderable_blocks})"


_NULL_SECTION_BYTES = Section().serialize()


def NULL_SECTION():
    return Section.deserialize(_NULL_SECTION_BYTES)


if __name__ == "__main__":
    import time

    start_time = time.time()
    [NULL_SECTION() for _ in range(256)]
    end_time = time.time()
    print(f"A: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    [NULL_SECTION() for _ in range(256)]
    end_time = time.time()
    print(f"B: {end_time - start_time:.4f} seconds")

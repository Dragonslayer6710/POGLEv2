from __future__ import annotations

import pickle

from Block import *
from Block import _block_state_cache

from POGLE.Physics.SpatialTree import Octree

import itertools
from timeit import timeit
import cProfile
from copy import copy

SECTION_BLOCK_WIDTH: int = 16
SECTION_BLOCK_WIDTH_RANGE: range = range(SECTION_BLOCK_WIDTH)
SECTION_BLOCK_EXTENTS: glm.vec3 = glm.vec3(SECTION_BLOCK_WIDTH, 1, SECTION_BLOCK_WIDTH)
SECTION_BLOCK_EXTENTS_HALF: glm.vec3 = SECTION_BLOCK_EXTENTS / 2
SECTION_NUM_BLOCKS: int = SECTION_BLOCK_WIDTH ** 2
SECTION_BLOCK_RANGE: range = range(SECTION_NUM_BLOCKS)
SECTION_BLOCK_RANGE_LIST: List[int] = list(SECTION_BLOCK_RANGE)

BLOCK_POSITIONS_IN_SECTION = [
    glm.vec3(x, 0, z) - SECTION_BLOCK_EXTENTS_HALF + BLOCK_EXTENTS_HALF
    for x, z in itertools.product(SECTION_BLOCK_WIDTH_RANGE, repeat=2)
]
NULL_BLOCK_STATES = [NULL_BLOCK_STATE] * SECTION_NUM_BLOCKS
NULL_BLOCKS = [None] * SECTION_NUM_BLOCKS
NULL_BLOCKS_BYTES = pickle.dumps(NULL_BLOCKS)

SECTION_BASE_CENTRE: glm.vec3 = glm.vec3(0, 0.5, 0)

SECTION_BASE_AABB: AABB = AABB.from_pos_size(
    SECTION_BASE_CENTRE,
    SECTION_BLOCK_EXTENTS
)

BLOCK_AABBS_IN_SECTION: List[AABB] = [
    copy(BLOCK_BASE_AABB) for pos in BLOCK_POSITIONS_IN_SECTION
]
BLOCK_AABBS_IN_SECTION_BYTES = pickle.dumps(BLOCK_AABBS_IN_SECTION)

class Section(PhysicalBox):
    def __init__(self, id: int, aabb: AABB = SECTION_BASE_AABB, block_start: int = 0, chunk: Chunk = None,
                 section_data: Optional[Tuple[Any, Any]] = None):
        super().__init__(aabb)
        # Cache self.pos in a local variable
        block_aabbs = pickle.loads(BLOCK_AABBS_IN_SECTION_BYTES)
        position = self.pos

        # Use in-place addition if glm supports it
        for block_aabb in block_aabbs:
            block_aabb.pos += position

        self.id: int = id

        from Chunk import Chunk
        self.chunk: Chunk = chunk
        if self.chunk:
            self.pos += self.chunk.pos

        self.octree: Octree = self.chunk.octree if chunk else Octree(self.bounds)

        self.blocks: List[Optional[Block]] = []
        if section_data is None:
            self._solid_blocks: int = 0
            self._solid_cache: List[Optional[int]] = NULL_BLOCKS.copy()
            self._solid_cache_valid: bool = False

            self._renderable_blocks: int = 0
            self._renderable_cache: List[Optional[int]] = NULL_BLOCKS.copy()
            self._renderable_cache_valid: bool = False

            self.block_states = NULL_BLOCK_STATES.copy()
            self.blocks: List[Optional[Block]] = NULL_BLOCKS.copy()
            for block_id in SECTION_BLOCK_RANGE:
                block = self.blocks[block_id] = Block(
                    block_id,
                    block_aabbs[block_id],
                    self.block_states[block_id],
                    self
                )
                self.update_block_in_lists(block)
        else:
            self.block_range, self._solid_blocks, self._renderable_blocks = section_data

    def is_block_solid(self, block_id: int):
        return (self._solid_blocks & (1 << block_id)) != 0

    def block_is_solid(self, block: Block):
        if not self.is_block_solid(block.id):  # if wasn't solid
            self._solid_blocks |= (1 << block.id)
            self._solid_cache[block.id] = block
            self.octree.insert(block)

    def block_is_not_solid(self, block: Block):
        if self.is_block_solid(block.id):  # if was solid
            self._solid_blocks &= ~(1 << block.id)
            self._solid_cache[block.id] = None
            self.octree.remove(block)

    def is_block_renderable(self, block_id: int):
        return (self._renderable_blocks & (1 << block_id)) != 0

    def block_is_renderable(self, block: Block):
        if not self.is_block_renderable(block.id):
            self._renderable_blocks |= (1 << block.id)
            self._renderable_cache[block.id] = block

    def block_is_not_renderable(self, block: Block):
        if self.is_block_renderable(block.id):
            self._renderable_blocks &= ~(1 << block.id)
            self._renderable_cache[block.id] = None

    def update_block_in_lists(self, block: Block):
        if block.is_solid:
            self.block_is_solid(block)
        else:
            self.block_is_not_solid(block)

        if block.is_renderable:
            self.block_is_renderable(block)
        else:
            self.block_is_not_renderable(block)

    def get_face_instances(self):
        return b"".join([block.get_face_instances() for block in self.renderable_blocks if block])

    def get_block_instances(self):
        return b"".join(block.get_instance() for block in self.renderable_blocks if block)

    @property
    def solid_blocks(self) -> List[Block]:
        if not self._solid_cache_valid:
            self._solid_cache = [block if self.block_is_solid(block) else None for block in self.blocks]
            self._solid_cache_valid = True
        return self._solid_cache

    @property
    def is_solid(self):
        return bool(self._solid_blocks)

    @property
    def renderable_blocks(self) -> List[Block]:
        if not self._renderable_cache_valid:
            self._renderable_cache = [block if self.block_is_renderable(block) else None for block in self.blocks]
            self._renderable_cache_valid = True
        return self._renderable_cache

    @property
    def is_renderable(self):
        return bool(self._renderable_blocks)

    def set_block_state(self, block_id: int, block_state: BlockState):
        block = self.blocks[block_id]
        if block_state.block_id is BlockID.Air:
            self.clear_block(block)
            return
        block.set_state(block_state)
        self.update_block_in_lists(block)

    def clear_block(self, block: Block):
        block.set_state(BlockID.Air)
        self.block_is_not_solid(block)
        self.block_is_not_renderable(block)

    def serialize(self) -> Tuple[bytes, bytes, bytes]:
        section_bytes = (
                struct.pack("fff", self.pos.x, self.pos.y, self.pos.z) +
                struct.pack("H", self._solid_blocks) +
                struct.pack("H", self._renderable_blocks)
        )
        block_position_bytes, block_state_bytes = Block.serialize_array(self.blocks)
        return section_bytes, block_position_bytes, block_state_bytes

    @classmethod
    def deserialize(cls, id: int, packed_data: Tuple[bytes, bytes, bytes], chunk: Chunk) -> Section:
        section_pos = glm.vec3(*struct.unpack("fff", packed_data[0][2:14]))
        section_data = (
            struct.unpack("H", packed_data[0][14:16])[0],
            struct.unpack("H", packed_data[0][16:])[0]
        )
        section = cls(
            id=id,
            aabb=AABB.from_pos_size(section_pos, SECTION_BLOCK_EXTENTS),
            chunk=chunk,
            section_data=section_data
        )
        blocks, block_states = Block.deserialize_array(SECTION_BLOCK_RANGE_LIST,(packed_data[1], packed_data[2]), section)
        section.block_states = block_states

        return section

    @staticmethod
    def serialize_array(sections: List[Section]) -> Tuple[bytes, bytes, bytes]: # section_bytes, block_position_bytes, block_state_bytes
        section_bytes = b""
        block_position_bytes = b""
        block_state_bytes = b""
        for section in sections:
            section_bytes += section.serialize()[0]
            block_position_bytes += section.serialize()[1]
            block_state_bytes += section.serialize()[2]
        return section_bytes, block_position_bytes, block_state_bytes

    @staticmethod
    def deserialize_array(ids: List[int], packed_data: Tuple[bytes, bytes, bytes], chunk: Chunk) -> Tuple[List[Section], List[Block]]:
        sections = []
        blocks = []
        for id, section_bytes, block_position_bytes, block_state_bytes in zip(ids, packed_data[0], packed_data[1], packed_data[2]):
            section, blocks = Section.deserialize(id, (section_bytes, block_position_bytes, block_state_bytes), chunk)
            sections.append(section)
            blocks.extend(blocks)
        return sections, blocks


    def __repr__(self):
        return f"Section(id={self.id}, pos={self.pos}, blocks={self.blocks})"

    def __str__(self):
        return self.__repr__()



if __name__ == "__main__":
    def test():
        Section(0)
    num_tests = 256
    filename = "section"
    cProfile.run(f"[test() for _ in range({num_tests})]", f"{filename}.prof")

from Chunk import Chunk

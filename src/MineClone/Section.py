from __future__ import annotations
from MineClone.Block import *


class Section(PhysicalBox):
    WIDTH: int = 16
    NUM_BLOCKS = WIDTH * WIDTH
    WIDTH_RANGE = range(WIDTH)
    SIZE = glm.vec3(WIDTH, 1, WIDTH)
    SERIALIZED_FMT_STRS = ["H", "ff"]
    SERIALIZED_HEADER_SIZE = np.sum([struct.calcsize(fmt_str) for fmt_str in SERIALIZED_FMT_STRS])
    SERIALIZED_SIZE = SERIALIZED_HEADER_SIZE + NUM_BLOCKS * Block.SERIALIZED_SIZE

    _EMPTY_BLOCKS: List[Block] = None

    def __init__(self, chunkY: int, chunkPos: glm.vec2, blocks: Optional[List[Block]] = None):
        super().__init__(AABB.from_pos_size(glm.vec3(chunkPos[0], chunkY, chunkPos[1]), Section.SIZE))
        if Section._EMPTY_BLOCKS is None:
            Section._EMPTY_BLOCKS = [None for _ in Section.WIDTH_RANGE for _ in
                                     Section.WIDTH_RANGE]  # Pre allocate blocks
        self.chunk_index = chunkY
        self.octree: Octree = Octree(self.bounds)

        self._solid_blocks: List[Block] = copy.deepcopy(Section._EMPTY_BLOCKS)
        self._renderable_blocks: List[Block] = copy.deepcopy(Section._EMPTY_BLOCKS)
        if not blocks:
            self.blocks = copy.deepcopy(Section._EMPTY_BLOCKS)
            for z in Section.WIDTH_RANGE:
                for x in Section.WIDTH_RANGE:
                    block = Block(self.chunk_index + (z * Section.WIDTH) + x, self.pos + glm.vec3(x, chunkY, z))
                    self.blocks[block.block_in_section_index] = block
                    self.update_block_in_lists(block)
        else:
            self.blocks = blocks
            for block in self.blocks:
                self.update_block_in_lists(block)

    def update_block_in_lists(self, block: Block):
        index = block.block_in_section_index
        is_solid = block.is_solid
        is_renderable = block.is_renderable

        if not self._solid_blocks[index]:
            if is_solid:
                self._solid_blocks[index] = block
                self.octree.insert(block)
        else:
            if not is_solid:
                self._solid_blocks[index] = None
        if not self._renderable_blocks[index]:
            if is_renderable:
                self._renderable_blocks[index] = block
        else:
            if not is_renderable:
                self._renderable_blocks[index] = None

    def get_face_instances(self):
        face_instances = b"".join([block.get_face_instances() for block in self._renderable_blocks])

    def get_block_instances(self):
        block_instances = b"".join(block.get_instance() for block in self._renderable_blocks)

    @property
    def solid_blocks(self) -> int:
        return len(self._solid_blocks)

    @property
    def renderable_blocks(self) -> int:
        return len(self._renderable_blocks)

    def serialize(self) -> bytes:
        index_bytes = struct.pack("H", self.chunk_index)
        pos_bytes = struct.pack("ff", *self.pos.xz)
        section_header_bytes = index_bytes + pos_bytes

        block_bytes = b"".join(block.serialize() for block in self.blocks)
        return section_header_bytes + block_bytes

    @classmethod
    def deserialize(cls, packed_data: bytes) -> Section:
        # Unpack the section header
        section_index = struct.unpack("H", packed_data[:2])[0]
        chunkPos = glm.vec2(struct.unpack("ff", packed_data[2:10]))

        blocks = copy.deepcopy(Section._EMPTY_BLOCKS)
        # The remaining bytes are block data so iterate
        for i in range(Section.NUM_BLOCKS):
            # Extract the block bytes
            start = Section.SERIALIZED_HEADER_SIZE + i * Block.SERIALIZED_SIZE
            end = start + Block.SERIALIZED_SIZE
            block_bytes = packed_data[start:end]

            # Deserialize the block
            blocks[i] = Block.deserialize(block_bytes)

        # Create a new instance of Section using the unpacked values and deserialized blocks
        return cls(
            chunkY=section_index,
            chunkPos=chunkPos,
            blocks=blocks
        )

    def __repr__(self):
        return f"Section(chunk_index: {self.chunk_index}, solid_blocks: {self.solid_blocks}, renderable_blocks:{self.renderable_blocks})"


# Example section serialization
section = Section(chunkY=0, chunkPos=glm.vec2(0, 0))
serialized_section = section.serialize()

# Deserialization example
deserialized_section = Section.deserialize(serialized_section)
print("Pre-Serialized Section:", section)
print("Serialized Section:", serialized_section)
print("Deserialized Section:", deserialized_section)

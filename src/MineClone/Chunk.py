from MineClone.Section import *


class Chunk(PhysicalBox):
    HEIGHT: int = 256
    HEIGHT_RANGE = range(HEIGHT)
    SIZE: glm.vec3 = glm.vec3(Section.WIDTH, HEIGHT, Section.WIDTH)
    SERIALIZED_FMT_STRS = ["ff"]
    SERIALIZED_HEADER_SIZE = np.sum([struct.calcsize(fmt_str) for fmt_str in SERIALIZED_FMT_STRS])
    SERIALIZED_SIZE = SERIALIZED_HEADER_SIZE + HEIGHT * Section.SERIALIZED_SIZE

    _EMPTY_SECTIONS: List[Section] = None

    def __init__(self, pos: glm.vec2, sections: Optional[List[Section]] = None):
        super().__init__(AABB.from_pos_size(glm.vec3(pos[0], Chunk.HEIGHT / 2, pos[1]), Chunk.SIZE))
        if Chunk._EMPTY_SECTIONS is None:
            Chunk._EMPTY_SECTIONS = [None for _ in Chunk.HEIGHT_RANGE]
        self.octree: Octree = Octree(AABB.from_pos_size(self.pos, Chunk.SIZE), Section.SIZE)
        self._solid_sections: List[Section] = copy.deepcopy(Chunk._EMPTY_SECTIONS)
        self._renderable_sections: List[Section] = copy.deepcopy(Chunk._EMPTY_SECTIONS)
        if not sections:
            self.sections = copy.deepcopy(Chunk._EMPTY_SECTIONS)
            for y in Chunk.HEIGHT_RANGE:
                section = Section(y, pos)
                self.sections[section.chunk_index] = section
                self.update_section_in_lists(section)
        else:
            self.sections = sections
            for section in self.sections:
                self.update_section_in_lists(section)

    def update_section_in_lists(self, section: Section):
        index = section.chunk_index
        is_solid = bool(section.solid_blocks)
        is_renderable = bool(section.renderable_blocks)

        if not self._solid_sections[index]:
            if is_solid:
                self._solid_sections[index] = section
                self.octree.insert(section)
        else:
            if not is_solid:
                self._solid_sections[index] = None
        if not self._renderable_sections[index]:
            if is_renderable:
                self._renderable_sections[index] = section
        else:
            if not is_renderable:
                self._renderable_sections[index] = None

    def get_face_instances(self):
        face_instances = b"".join([section.get_face_instances() for section in self._renderable_sections])

    def get_section_instances(self):
        section_instances = b"".join(section.get_block_instances() for section in self._renderable_sections)

    @property
    def solid_sections(self) -> int:
        return len(self._solid_sections)

    @property
    def renderable_sections(self) -> int:
        return len(self._renderable_sections)

    def serialize(self) -> bytes:
        pos_bytes = struct.pack("ff", *self.pos.xz)
        section_header_bytes = pos_bytes

        section_bytes = b"".join(section.serialize() for section in self.sections)
        return section_header_bytes + section_bytes

    @classmethod
    def deserialize(cls, packed_data: bytes) -> Section:
        # Unpack the section header
        pos = glm.vec2(struct.unpack("ff", packed_data[:8]))

        sections = copy.deepcopy(Chunk._EMPTY_SECTIONS)
        # The remaining bytes are section data so iterate
        for i in Chunk.HEIGHT_RANGE:
            # Extract the section bytes
            start = Chunk.SERIALIZED_HEADER_SIZE + i * Section.SERIALIZED_SIZE
            end = start + Section.SERIALIZED_SIZE
            section_bytes = packed_data[start:end]

            # Deserialize the section
            sections[i] = Block.deserialize(section_bytes)

        # Create a new instance of Section using the unpacked values and deserialized sections
        return cls(
            pos=pos,
            sections=sections
        )

    def __repr__(self):
        return f"Chunk(solid_sections: {self.solid_sections}, renderable_sections:{self.renderable_sections})"


# Example section serialization
chunk = Chunk(pos=glm.vec2(0, 0))
serialized_chunk = chunk.serialize()

# Deserialization example
deserialized_chunk = Chunk.deserialize(serialized_chunk)
print("Pre-Serialized Chunk:", chunk)
print("Serialized Chunk:", serialized_chunk)
print("Deserialized Chunk:", deserialized_chunk)


class ChunkRange(PhysicalBox):
    def __init__(self, pos: glm.vec2, size: glm.vec2):
        super().__init__(AABB.from_pos_size(pos, size))

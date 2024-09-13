from MineClone.Chunk import *


class Region(ChunkRange):
    WIDTH = 32
    SIZE: glm.vec2 = glm.vec2(WIDTH)
    SERIALIZED_FMT_STRS = ["ff"]
    SERIALIZED_HEADER_SIZE = np.sum([struct.calcsize(fmt_str) for fmt_str in SERIALIZED_FMT_STRS])
    SERIALIZED_SIZE = SERIALIZED_HEADER_SIZE + (WIDTH ** 2) * Section.SECTION_SERIALIZED_SIZE

    _EMPTY_SECTIONS: List[Section] = None
    def __init__(self, pos: glm.vec2):
        super().__init__(pos, Region.SIZE)

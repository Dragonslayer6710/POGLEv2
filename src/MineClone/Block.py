from __future__ import annotations

import copy
import struct

from MineClone.Face import *
from POGLE.Physics.SpatialTree import *
from dataclasses import dataclass


class BlockID(Enum):
    Air = 0
    Grass = auto()
    Dirt = auto()
    Stone = auto()


FACES_IN_BLOCK_RANGE = range(6)
NULL_FACES = [NULL_FACE() for _ in FACES_IN_BLOCK_RANGE]


@dataclass
class BlockProperties:
    name: str
    is_solid: bool = True
    is_opaque: bool = True
    faces: Optional[Union[List[Union[int, FaceID]], List[Face]]] = None
    deserialized: bool = False

    def __post_init__(self):
        if not self.deserialized:
            if self.faces is None:
                self.faces = copy.deepcopy(NULL_FACES)
            elif isinstance(self.faces, list) and len(self.faces) != 6:
                raise ValueError("Block Properties initialized with less than or more than 6 faces")
            for side, face in enumerate(self.faces):
                if isinstance(face, Face):
                    final_face = face
                elif isinstance(face, FaceID):
                    final_face = copy.deepcopy(get_face(face))
                elif isinstance(face, int) and FaceID._value2member_map_.get(face):
                    final_face = copy.deepcopy(get_face(FaceID(face)))
                else:
                    raise ValueError("Face provided to faces is not valid")
                final_face.set_face_in_block(side)
                self.faces[side] = final_face
        del self.deserialized

    def serialize(self):
        encoded_str = self.name.encode('utf-8')
        name_bytes = struct.pack('I', len(encoded_str)) + encoded_str
        byte = 0
        if self.is_solid:
            byte |= 0b00000001  # Set bit 0
        if self.is_opaque:
            byte |= 0b00000010  # Set bit 1
        bools_byte = struct.pack("B", byte)
        face_bytes = b"".join([face.serialize() for face in self.faces])
        return name_bytes + bools_byte + face_bytes

    @classmethod
    def deserialize(cls, packed_data) -> BlockProperties:
        name_length = struct.unpack('I', packed_data[:4])[0]
        byte_start = 4 + name_length
        faces_start = byte_start + 1

        bools_byte = struct.unpack("B", packed_data[byte_start:faces_start])[0]

        faces = [None] * 6
        for i in FACES_IN_BLOCK_RANGE:
            start = faces_start + i * Face.SERIALIZED_SIZE
            end = start + Face.SERIALIZED_SIZE
            faces[i] = Face.deserialize(packed_data[start:end])

        return cls(
            name=packed_data[4:byte_start].decode("utf-8"),
            is_solid=bool(bools_byte & 0b00000001),  # Check bit 0
            is_opaque=bool(bools_byte & 0b00000010),  # Check bit 1
            faces=faces,
            deserialized=True
        )


_block_properties_dict: Dict[BlockID, bytes] = {
    BlockID.Air: BlockProperties("Air", False, False).serialize(),
    BlockID.Dirt: BlockProperties(name="Dirt", faces=[FaceID.Dirt] * 6).serialize(),
    BlockID.Grass: BlockProperties(name="Grass",
                                   faces=[FaceID.GrassSide] * 4 + [FaceID.GrassTop, FaceID.Dirt]).serialize(),
    BlockID.Stone: BlockProperties(name="Stone", faces=[FaceID.Stone] * 6).serialize()
}


def get_block_props(blockID: BlockID = BlockID.Air) -> BlockProperties:
    return BlockProperties.deserialize(_block_properties_dict[blockID])


def NULL_BLOCK_PROPS() -> BlockProperties:
    return get_block_props()


class Block(PhysicalBox):
    SERIALIZED_FMT_STRS = ["H"]
    SERIALIZED_SIZE = np.sum([struct.calcsize(fmt_str) for fmt_str in SERIALIZED_FMT_STRS])

    def __init__(self, block_id: Union[int, BlockID] = BlockID.Air):
        super().__init__(AABB.from_pos_size(glm.vec3()))

        self.ID: BlockID = None
        self.properties: BlockProperties = None
        self._visible_faces: int = 0
        self.set_block(block_id)

        self.block_in_chunk_index: int = 0
        self.block_in_section_index: int = 0

    def set_block(self, block_id: Union[int, BlockID]):
        if isinstance(block_id, int):
            block_id = BlockID(block_id)
        self.ID: BlockID = block_id
        prev_properties = self.properties
        self.properties = get_block_props(self.ID)
        if not block_id.value:
            self._visible_faces = 0

    @property
    def faces(self):
        if not self.properties:
            raise RuntimeError("Cannot access faces until block properties are set")
        return self.properties.faces

    def set_block_in_chunk(self, block_in_chunk_index: Optional[int] = None,
                           block_in_chunk_pos: Optional[glm.vec3] = None):
        if block_in_chunk_index is not None:
            self.block_in_chunk_index: int = block_in_chunk_index
            self.block_in_section_index: int = block_in_chunk_index % 256  # Changed bitmask to modulo operation for clarity
            if block_in_chunk_pos is None:
                raise ValueError("Block cannot be set in chunk without the chunk position")
            self.bounds = AABB.from_pos_size(block_in_chunk_pos)

    def hide_face(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = Side(side)
        if self.faces[side.value]:
            self._visible_faces -= 1

    def reveal_face(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = Side(side)
        if self.faces[side.value]:
            self._visible_faces += 1

    def get_face_instances(self):
        return b"".join([face.get_instance() for face in self.faces])

    def get_instance(self):
        if self.block_in_section_index is None:
            raise RuntimeError("Cannot Get Block Instance If No Section index is Set")
        return

    @property
    def name(self):
        return self.properties.name

    @property
    def is_solid(self):
        return self.properties.is_solid

    @property
    def is_opaque(self):
        return self.properties.is_opaque

    @property
    def is_renderable(self):
        return self._visible_faces and self.ID != BlockID.Air  # If not air and have faces that are visible, it is renderable

    def serialize(self) -> bytes:
        return struct.pack("H", self.ID.value)

    @classmethod
    def deserialize(cls, packed_data: bytes) -> Block:
        # Unpack the serialized data
        return cls(block_id=struct.unpack("H", packed_data[:2])[0])

    def __repr__(self):
        return f"Block(id: {self.ID}, name: {self.name}, solid: {self.is_solid}, opaque: {self.is_opaque}, visible_faces:{self._visible_faces})"


_blocks_dict: Dict[BlockID, bytes] = {
    BlockID.Air: Block().serialize(),
    BlockID.Dirt: Block(BlockID.Dirt).serialize(),
    BlockID.Grass: Block(BlockID.Grass).serialize(),
    BlockID.Stone: Block(BlockID.Stone).serialize()
}


def get_block(block_id: BlockID = BlockID.Air) -> Block:
    return Block.deserialize(_blocks_dict[block_id])


def NULL_BLOCK() -> Block:
    return get_block()

if __name__ == "__main__":
    import time

    start_time = time.time()
    for i in range(10000):
        block = copy.deepcopy(NULL_BLOCK())
    end_time = time.time()
    print(f"Deep copy: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    for i in range(10000):
        block = Block.deserialize(nb_bytes)
    end_time = time.time()
    print(f"Deserialize: {end_time - start_time:.4f} seconds")

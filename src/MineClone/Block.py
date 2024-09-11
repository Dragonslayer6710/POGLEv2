from __future__ import annotations
from MineClone.Face import *
from POGLE.Physics.SpatialTree import *
from dataclasses import dataclass


class BlockID(Enum):
    Air = 0
    Grass = auto()
    Dirt = auto()
    Stone = auto()


@dataclass
class BlockProperties:
    name: str
    is_solid: bool = True
    is_opaque: bool = True
    tex_net: Optional[List[TexRef]] = None


_BlocksProperties: Dict[BlockID, BlockProperties] = {
    BlockID.Air: BlockProperties("Air", False, False),
    BlockID.Dirt: BlockProperties(name="Dirt", tex_net=[TexRef.Dirt] * 6),
    BlockID.Grass: BlockProperties(name="Grass", tex_net=[TexRef.GrassSide] * 4 + [TexRef.GrassTop, TexRef.Dirt]),
    BlockID.Stone: BlockProperties(name="Stone", tex_net=[TexRef.Stone] * 6)
}


class Block(PhysicalBox):
    FACE_RANGE = range(6)
    _AIR_FACES = [Face(i) for i in FACE_RANGE]
    SERIALIZED_FMT_STRS = ["H", "fff", "H"]
    SERIALIZED_HEADER_SIZE = np.sum([struct.calcsize(fmt_str) for fmt_str in SERIALIZED_FMT_STRS])
    SERIALIZED_SIZE = SERIALIZED_HEADER_SIZE + 6 * Face.SERIALIZED_SIZE
    def __init__(self, block_in_chunk_index: int, pos: glm.vec3, blockID: Union[int, BlockID] = BlockID.Air, faces: Optional[List[Face]] = None):
        super().__init__(AABB.from_pos_size(pos))
        self.block_in_chunk_index: int = block_in_chunk_index
        self.block_in_section_index: int = block_in_chunk_index % 256  # Changed bitmask to modulo operation for clarity
        self.ID: BlockID = None
        self.faces: List[Face] = None
        self.properties: BlockProperties = None
        self._visible_faces: int = 0
        if not faces:
            self._visible_faces = 6
            self.set_block(blockID)
        else:
            self.ID = blockID
            self.faces = faces
            self.properties = _BlocksProperties[self.ID]
            for face in faces:
                if face.visible:
                    self._visible_faces += 1


    def set_block(self, blockID: Union[int, BlockID]):
        if isinstance(blockID, int):
            blockID = BlockID(blockID)
        if self.ID is None:
            self.faces = copy.deepcopy(Block._AIR_FACES)
        self.ID: BlockID = blockID
        prev_properties = self.properties
        self.properties = _BlocksProperties[self.ID]
        if blockID.value:
            [face.setTexID(self.properties.tex_net[i]) for i, face in enumerate(self.faces)]
        else:
            self.faces = copy.deepcopy(Block._AIR_FACES)
            self._visible_faces = 0


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
        b"".join([face.get_instance() for face in self.faces])

    def get_instance(self):
        pass

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
        return self._visible_faces and self.ID.value # If not air and have faces that are visible, it is renderable

    def serialize(self) -> bytes:
        block_header_bytes = (
            struct.pack("H", self.block_in_section_index) +
            struct.pack("fff", *self.pos) +
            struct.pack("H", self.ID.value)
        )

        face_bytes = b"".join(face.serialize() for face in self.faces)
        return block_header_bytes + face_bytes

    @classmethod
    def deserialize(cls, packed_data: bytes) -> Block:
        # Unpack the serialized data
        block_in_chunk_index = struct.unpack("H", packed_data[:2])[0]
        pos = glm.vec3(struct.unpack("fff", packed_data[2:14]))
        block_id = BlockID(struct.unpack("H", packed_data[14:16])[0])

        faces = copy.deepcopy(Block._AIR_FACES)
        # The remaining bytes are block data so iterate
        for i in range(6):
            # Extract the block bytes
            start = Block.SERIALIZED_HEADER_SIZE + i * Face.SERIALIZED_SIZE
            end = start + Face.SERIALIZED_SIZE
            face_bytes = packed_data[start:end] # 16 byte offset for header

            # Deserialize the block
            faces[i] = Face.deserialize(face_bytes)

        # Create a new instance of Block using the unpacked values
        return cls(
            block_in_chunk_index=block_in_chunk_index,
            pos=pos,
            blockID=block_id
        )

    def __repr__(self):
        return f"Block(id: {self.ID}, name: {self.name}, solid: {self.is_solid}, opaque: {self.is_opaque}, visible_faces:{self._visible_faces})"

# Example block serialization
block = Block(0, glm.vec3(1.0, 2.0, 3.0), BlockID.Grass)
serialized_block = block.serialize()

# Deserialization example
deserialized_block = Block.deserialize(serialized_block)
print("Pre-Serialized Block:", block)
print("Serialized Block:", serialized_block)
print("Deserialized Block:", deserialized_block)
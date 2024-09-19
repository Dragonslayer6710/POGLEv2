from __future__ import annotations

from Face import *
from Face import _face_tex_cache

from POGLE.Physics.Collisions import PhysicalBox, AABB

from POGLE.Core.Core import Union, Any, Dict, List, Tuple
from POGLE.Core.Core import np

from POGLE.Core.Core import struct

from dataclasses import dataclass, field


class BlockID(Renum):
    Air = 0
    Grass = auto()
    Dirt = auto()
    Stone = auto()


_block_id_cache: ImDict[int, BlockID] = ImDict(
    {member.value: member for member in BlockID}
)

_all_block_states_set: bool = False


@dataclass
class BlockState:
    block_id: Union[np.uint16, BlockID]
    face_textures: List[FaceTex]
    is_solid: bool = True
    is_opaque: bool = True

    face_textures_as_shorts: List[np.short] = field(default_factory=list)

    def __post_init__(self):
        if _all_block_states_set:
            raise RuntimeError("Cannot call __init__ once all block states have been created")
        if isinstance(self.block_id, np.uint16):
            self.block_id: BlockID = _block_id_cache[self.block_id]
        self.face_textures_as_shorts = [face_tex.value for face_tex in self.face_textures]

    @property
    def name(self) -> str:
        return self.block_id.name

    @property
    def value(self) -> int:
        return self.block_id.value

    def serialize(self) -> bytes:
        return struct.pack("H", self.value)

    @classmethod
    def deserialize(cls, packed_data: bytes) -> BlockState:
        # Unpack the serialized data
        return cls(block_id=np.frombuffer(packed_data, dtype=np.uint16)[0])

    @staticmethod
    def serialize_array(block_states: list[BlockState]) -> bytes:
        return struct.pack(f'<{len(block_states)}H', *[bs.value for bs in block_states])

    @staticmethod
    def deserialize_array(packed_data: bytes) -> list[BlockState]:
        unpacked_data = struct.unpack(f'<{len(packed_data) // 2}H', packed_data)
        return list(map(BlockState, unpacked_data))

    def __repr__(self):
        return f"Block(id: {self.value}, name: {self.name}, solid: {self.is_solid}, opaque: {self.is_opaque}, visible_faces:{self._visible_faces})"

    def __str__(self):
        return self.__repr__()


_block_state_cache: ImDict[BlockID, BlockState] = ImDict({
    BlockID.Air: BlockState(BlockID.Air, [FaceTex.Null] * 6, False, False),
    BlockID.Dirt: BlockState(BlockID.Dirt, [FaceTex.Dirt] * 6),
    BlockID.Grass: BlockState(BlockID.Grass, [FaceTex.GrassSide] * 4 + [FaceTex.GrassTop, FaceTex.Dirt]),
    BlockID.Stone: BlockState(BlockID.Stone, [FaceTex.Stone] * 6)
})
_all_block_states_set = True
NULL_BLOCK_STATE = _block_state_cache[BlockID.Air]


class Side(Renum):
    Null = -1,
    West = auto()
    South = auto()
    East = auto()
    North = auto()
    Top = auto()
    Bottom = auto()


_side_cache: ImDict[int, Side] = ImDict(
    {member.value: member for member in Side}
)

_opposite_side: ImDict[Side, Side] = ImDict({
    Side.West: Side.East,
    Side.South: Side.North,
    Side.East: Side.West,
    Side.North: Side.South,
    Side.Top: Side.Bottom,
    Side.Bottom: Side.Top
})

BLOCK_EXTENTS: glm.vec3 = glm.vec3(1)
BLOCK_EXTENTS_HALF: glm.vec3 = BLOCK_EXTENTS / 2

BLOCK_STATE_SERIALIZED_SIZE = struct.calcsize("H")  # For BlockID
BLOCK_SERIALIZED_SIZE: int = BLOCK_STATE_SERIALIZED_SIZE + struct.calcsize("fff")  # For position

BLOCK_BASE_AABB: AABB = AABB.from_pos_size(glm.vec3())

class Block(PhysicalBox):
    def __init__(self, id: int, aabb: AABB, block_state: BlockState, section: Optional[Section] = None) -> Block:
        super().__init__(aabb)
        self.id: int = id

        self.section: Section = section

        self._neighbour_refs: ImDict[Side, int] = {}

        self.state: BlockState = block_state

        self._visible_faces: int = 0
        self._visible_cache: int = 0
        self._visible_cache_valid: bool = False

    def neighbour(self, side: Union[int, Side], blocks: List[Block]):
        return

    def set_state(self, block_state: Union[BlockID, BlockState]):
        if isinstance(block_state, BlockID):
            block_state = _block_state_cache[block_state]
        was_solid = self.is_solid
        was_opaque = self.is_opaque
        self.state = block_state
        if was_solid and not self.is_solid:
            pass

    def update_face_visibility(self):
        pass

    @property
    def block_id(self) -> BlockID:
        return self.state.block_id

    @property
    def name(self) -> str:
        return self.state.name

    @property
    def value(self) -> int:
        return self.state.value

    @property
    def is_air(self) -> bool:
        return self.block_id == BlockID.Air

    @property
    def is_solid(self) -> bool:
        return self.state.is_solid

    @property
    def is_opaque(self) -> bool:
        return self.state.is_opaque

    @property
    def is_renderable(self) -> bool:
        return self._visible_faces != 0

    @property
    def visible_faces(self) -> int:
        if self.is_air:
            return 0
        if not self._visible_cache_valid:
            self._visible_cache = bin(self._visible_faces).count("1")
        return self._visible_cache

    def is_side_visible(self, side: Union[int, Side]):
        if isinstance(side, int):
            side: Side = _side_cache[side]
        return (self._visible_faces & (1 << side.value)) != 0

    def hide_face(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = _side_cache[side]
        if self.is_side_visible(side):
            self._visible_faces &= ~(1 << side.value)
            self._visible_cache -= 1

    def reveal_face(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = _side_cache[side]
        if not self.is_side_visible(side):
            self._visible_faces |= (1 << side.value)
            self._visible_cache += 1

    def serialize(self) -> Tuple[bytes, bytes]: # (position, block_state)
        return struct.pack("fff", self.pos.x, self.pos.y, self.pos.z), self.state.serialize()

    @classmethod
    def deserialize(cls, id: int, packed_data: Tuple[bytes, bytes], section: Section) -> Block:
        # Unpack the serialized data
        return cls(
            id=id,
            pos=glm.vec3(struct.unpack("fff", packed_data[0])),
            block_state=BlockState.deserialize(packed_data[1]),
            section=section
        )

    @staticmethod
    def serialize_array(blocks: List[Block]) -> Tuple[bytes, bytes]: # (block_positions, block_states)
        block_position_bytes = b""
        block_state_bytes = b""
        for block in blocks:
            block_position_bytes += struct.pack("fff", block.pos.x, block.pos.y, block.pos.z)
            block_state_bytes += block.state.serialize()
        return block_position_bytes, block_state_bytes

    @staticmethod
    def deserialize_array(ids: List[int],packed_data: Tuple[bytes, bytes], section: Section) -> Tuple[List[Block], List[BlockState]]:
        blocks = []
        block_states = []
        for i in range(len(ids)):
            block = Block.deserialize(ids[i], packed_data[i], section)
            blocks.append(block)
            block_states.append(block.state)
        return blocks, block_states

    def __repr__(self):
        return f"Block(id={self.id}, pos={self.pos}, state={self.state})"

    def __str__(self):
        return self.__repr__()


from Section import Section

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from Chunk import Chunk

from Face import *
from Face import _face_tex_cache

from POGLE.Core.Core import ImDict

from POGLE.Core.Core import Union, Any, Dict, List, Tuple
from POGLE.Core.Core import np

from POGLE.Core.Core import struct
from POGLE.Physics.Collisions import PhysicalBox, AABB

from timeit import timeit
import cProfile
from copy import copy, deepcopy
import pickle

import struct
import nbtlib

_biType = np.ushort


class BlockID(Renum):
    Air = _biType(0)
    Grass = _biType(1)
    Dirt = _biType(2)
    Stone = _biType(3)


_block_id_cache: np.ndarray = np.array([member for member in BlockID], dtype=object)


@dataclass(frozen=True)
class BlockData:
    name: str
    id: BlockID
    face_textures: Optional[List[FaceTex]]
    is_solid: bool = True
    is_opaque: bool = True
    states: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Populate the states dictionary with child class attributes
        # Use the attribute names and their values to populate the states
        for key, value in self.__dict__.items():
            if key not in ['name', 'face_textures', 'states']:
                # Assigning to the states dict
                self.states[key] = value if not isinstance(value, BlockID) else value.value

    def to_nbt(self) -> nbtlib.Compound:
        return nbtlib.Compound({
          "BlockID": nbtlib.Int(self.id.value)
        })

@dataclass(frozen=True)
class Grass(BlockData):
    grass_type: Literal["grass", "mycelium", "podzol"] = "grass"
    snowy: bool = False


@dataclass(frozen=True)
class Dirt(BlockData):
    dirt_type: Literal["normal", "coarse"] = "normal"


@dataclass(frozen=True)
class Stone(BlockData):
    stone_type: Literal[
        "stone", "granite", "granite_smooth", "diorite", "diorite_smooth", "andesite", "andesite_smooth"
    ] = "stone"


_block_data_cache: Dict[BlockData] = ImDict({
    BlockID.Air: BlockData("Air", BlockID.Air, None, False, False),
    BlockID.Grass: Grass("Grass", BlockID.Grass, (FaceTex.GrassSide,) * 4 + (FaceTex.GrassTop, FaceTex.Dirt)),
    BlockID.Dirt: Dirt("Dirt", BlockID.Dirt, (FaceTex.Dirt,) * 6),
    BlockID.Stone: Stone("Stone", BlockID.Stone, (FaceTex.Stone,) * 6)
})

AIR_BLOCK_DATA = _block_data_cache[BlockID.Air]

_sType = np.byte


class Side(Renum):
    Null = _sType(-1),
    West = _sType(0)
    South = _sType(1)
    East = _sType(2)
    North = _sType(3)
    Top = _sType(4)
    Bottom = _sType(5)


_side_cache: np.ndarray = np.array([member for member in Side], dtype=object)  # type: ignore

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

_neighbour_offset: Dict[Side, glm.vec3] = {
    Side.West: glm.vec3(-1, 0, 0),
    Side.South: glm.vec3(0, 0, 1),
    Side.East: glm.vec3(1, 0, 0),
    Side.North: glm.vec3(0, 0, -1),
    Side.Top: glm.vec3(0, 1, 0),
    Side.Bottom: glm.vec3(0, -1, 0)
}


@dataclass
class MCPhys(PhysicalBox):
    index: int = 0

    def __post_init__(self):
        super().__init__(self.__aabb)

    def __init_subclass__(cls, aabb: Union[glm.vec3, AABB], size: Optional[glm.vec3] = None):
        if isinstance(aabb, glm.vec3):
            if size is None:
                size = aabb
            aabb = AABB.from_pos_size(aabb, size)
        elif not isinstance(aabb, AABB):
            raise TypeError("MCPhys subclasses must receive AABB data in their definitions by giving an AABB or"
                            " position and size")
        cls.__aabb = aabb


@dataclass
class Block(MCPhys, aabb=BLOCK_BASE_AABB):
    data: BlockData = AIR_BLOCK_DATA
    _visible_faces: int = 0
    _visible_cache: int = 0
    _visible_cache_valid: bool = False

    chunk: Optional[Chunk] = None

    def __post_init__(self):
        super().__post_init__()

    def initialize(self, chunk: Optional[Chunk] = None):
        if chunk:
            if self.chunk:
                raise RuntimeError("Attempted to set chunk of a block already set in a chunk")
            self.chunk = chunk
        self.pos += self.chunk.pos

    def neighbour(self, side: Union[int, Side], blocks: List[Block]):
        return

    def set(self, block_data: Union[int, BlockID, BlockData]):
        if isinstance(block_data, int):
            block_data = _block_data_cache[
                _block_id_cache[block_data]
            ]
        elif isinstance(block_data, BlockID):
            block_data = _block_data_cache[block_data]
        was_solid = self.is_solid
        was_opaque = self.is_opaque
        self.data = block_data
        if was_solid and not self.is_solid:
            pass

    def update_face_visibility(self):
        pass

    @property
    def block_id(self) -> BlockID:
        return self.data.id

    @property
    def name(self) -> str:
        return self.data.name

    @property
    def value(self) -> int:
        return self.data.value

    @property
    def is_air(self) -> bool:
        return self.block_id == BlockID.Air

    @property
    def is_solid(self) -> bool:
        return self.data.is_solid

    @property
    def is_opaque(self) -> bool:
        return self.data.is_opaque

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

    def get_face_instances(self) -> bytes:
        face_bytes: bytes = b""
        for side in range(6):
            if self.is_side_visible(side):
                face_bytes += (
                        struct.pack("B", side) +
                        struct.pack("H", self.data.face_textures_as_shorts[side])
                )
            else:
                face_bytes += (
                        struct.pack("B", -1) +
                        struct.pack("H", 0)
                )

    def serialize(self) -> Tuple[bytes, bytes]:  # (position, block_state)
        return struct.pack("fff", self.pos.x, self.pos.y, self.pos.z), self.data.serialize()

    @classmethod
    def deserialize(cls, block_in_section_id: int, packed_data: Tuple[bytes, bytes], section: Section) -> Block:
        # Unpack the serialized data
        return cls(
            id=block_in_section_id,
            _aabb=AABB.from_pos_size(glm.vec3(struct.unpack("fff", packed_data[0]))),
            data=BlockState.deserialize(packed_data[1]),
            section=section
        )

    @staticmethod
    def serialize_array(blocks: List[Block]) -> Tuple[bytes, bytes]:  # (block_positions, block_states)
        block_position_bytes = b""
        block_state_bytes = b""
        for block in blocks:
            block_position_bytes += struct.pack("fff", block.pos.x, block.pos.y, block.pos.z)
            block_state_bytes += block.data.serialize()
        return block_position_bytes, block_state_bytes

    @staticmethod
    def deserialize_array(ids: List[int], packed_data: Tuple[bytes, bytes], section: Section) -> Tuple[
        List[Block], List[BlockData]]:
        blocks = []
        block_states = []
        for i in range(len(ids)):
            block = Block.deserialize(ids[i], packed_data[i], section)
            blocks.append(block)
            block_states.append(block.data)
        return blocks, block_states

    def to_nbt(self) -> nbtlib.Compound:
        return self.data.to_nbt()


if __name__ == "__main__":
    def test():
        Block(0)


    num_tests = 1_000_000
    filename = "block"
    cProfile.run(f"[test() for _ in range({num_tests})]", f"{filename}.prof")

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass, field

import glm

from POGLE.Geometry.Data import NMM
from POGLE.Renderer.Mesh import *
from POGLE.Shader import ShaderProgram, UniformBuffer, UniformBlock

if TYPE_CHECKING:
    from Chunk import Chunk

from Face import *
from Face import _face_tex_id_cache, _faceTextureAtlas

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
    face_textures: Optional[List[FaceTexID]] = field(default_factory=lambda: [FaceTexID.Null for _ in range(6)])
    face_sizes: Optional[List[FaceSizeID]] = field(default_factory=lambda: [FaceSizeID.Full for _ in range(6)])
    is_solid: bool = True
    is_opaque: bool = True
    states: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Populate the states dictionary with child class attributes
        # Use the attribute names and their values to populate the states
        for key, value in self.__dict__.items():
            if key not in ['name', 'face_textures', 'face_sizes', 'states']:
                # Assigning to the states dict
                self.states[key] = value

    def to_nbt(self) -> nbtlib.Compound:
        return nbtlib.Compound({
            "BlockID": nbtlib.Int(self.id)
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
    BlockID.Grass: Grass(
        "Grass",
        BlockID.Grass,
        (FaceTexID.GrassSide,) * 4 + (FaceTexID.GrassTop, FaceTexID.Dirt),
    ),
    BlockID.Dirt: Dirt("Dirt", BlockID.Dirt, (FaceTexID.Dirt,) * 6),
    BlockID.Stone: Stone("Stone", BlockID.Stone, (FaceTexID.Stone,) * 6)
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


_side_cache: np.ndarray[Side] = np.array([member for member in Side], dtype=object)  # type: ignore

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

_FACE_MASKS = tuple(1 << i for i in range(6))

import uuid


@dataclass
class MCPhys(PhysicalBox):
    index: int = 0

    def __post_init__(self):
        self._id = MCPhys.index  # uuid.uuid4()  # Generate a unique UUID for this instance
        MCPhys.index += 1
        super().__init__(deepcopy(self.__aabb))  # Initialize the parent class

    def __init_subclass__(cls, aabb: Union[glm.vec3, AABB], size: Optional[glm.vec3] = None):
        if isinstance(aabb, glm.vec3):
            if size is None:
                size = aabb
            aabb = AABB.from_pos_size(aabb, size)
        elif not isinstance(aabb, AABB):
            raise TypeError("MCPhys subclasses must receive AABB data in their definitions by giving an AABB or"
                            " position and size")
        cls.__aabb = aabb


_face_model_mats: Dict[Side, glm.mat4] = {
    Side.East: NMM(glm.vec3(-1.0, 0, 0), glm.vec3(0, 90, 0)),
    Side.South: NMM(glm.vec3(0, 0, 1.0)),
    Side.West: NMM(glm.vec3(1.0, 0, 0), glm.vec3(0, -90, 0)),
    Side.North: NMM(glm.vec3(0, 0, -1.0), glm.vec3(0, 180, 0)),
    Side.Top: NMM(glm.vec3(0, 1.0, 0), glm.vec3(90, 0, 0)),
    Side.Bottom: NMM(glm.vec3(0, -1.0, 0), glm.vec3(-90, 0, 0)),
}

BLOCK_NULL_INSTANCE: np.ndarray[int] = np.array([
    [i, FaceTexID.Null, FaceSizeID.Full] for i in range(6)
]).flatten()


@dataclass
class Block(MCPhys, aabb=BLOCK_BASE_AABB):
    data: Union[int, BlockData] = AIR_BLOCK_DATA

    chunk: Optional[Chunk] = None

    _id: uuid.UUID = field(init=False)  # Set to init=False to not include it in the generated __init__

    # Define equality based on the tuple
    def __eq__(self, other):
        if isinstance(other, MCPhys):
            return self._id == other._id
        return False

    # Define hash based on the tuple
    def __hash__(self):
        return hash(self._id)

    def __post_init__(self):
        super().__post_init__()
        self._visible_bitmask: int = 63
        self._visible_list: List[bool] = [False] * 6
        self._num_visible_cache: int = 0
        self._visible_cache_valid: bool = False

        self._face_model_mats: Dict[Side, glm.mat4] = {}
        self.initialized: bool = False
        self._awaiting_update: bool = False

        self.face_instances: List[int] = copy(BLOCK_NULL_INSTANCE)

    def initialize(self, chunk: Optional[Chunk] = None):
        if chunk:
            if self.chunk:
                raise RuntimeError("Attempted to set chunk of a block already set in a chunk")
            self.chunk = chunk
        self.pos += self.chunk.pos

        self.enqueue_update()
        self.initialized = True

    def enqueue_update(self):
        if self._awaiting_update:
            raise RuntimeError("Attempted to put block into update queue whilst already awaiting an update")
        self._awaiting_update = True
        self.chunk.enqueue_block(self)

    def stack_update(self):
        if self._awaiting_update:
            raise RuntimeError("Attempted to put block into update queue whilst already awaiting an update")
        self._awaiting_update = True
        self.chunk.stack_block(self)

    def neighbour(self, side: Union[int, Side]) -> Optional[Block]:
        return self.chunk.get_block(self.pos + _neighbour_offset[side])

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
        for i in range(6):
            self.face_instances[3 * i + 1] = self.data.face_textures[i]

        if was_solid and not self.is_solid:
            pass

        if self.initialized:
            self.enqueue_update()

    def update(self):
        if not self._awaiting_update:
            return  # Shouldn't be possible
        for side in range(6):
            neighbour = self.neighbour(side)
            if neighbour is None:
                self._face_is_visible(side)
            else:
                if not neighbour.is_opaque:
                    self._face_is_visible(side)
                else:
                    self._face_is_not_visible(side)
                if not neighbour._awaiting_update:
                    neighbour.stack_update()
        self._awaiting_update = False

    def _is_face_visible(self, face: int):
        return self._visible_bitmask & _FACE_MASKS[face]

    def _face_is_visible(self, face: int):
        if not self._is_face_visible(face):  # Check if Face was not visible
            self._visible_bitmask |= _FACE_MASKS[face]
            self._visible_cache_valid = False

    def _face_is_not_visible(self, face: int):
        if self._is_face_visible(face):  # Check if Face was visible
            self._visible_bitmask &= ~_FACE_MASKS[face]
            self._visible_cache_valid = False

    def _validate_visible_cache(self):
        if self._visible_cache_valid:
            return
        self._num_visible_cache = len(self._visible_list)
        self._visible_cache_valid = True

    @property
    def visible_faces(self):
        self._validate_visible_cache()
        return self._visible_list

    @property
    def num_visible_faces(self):
        self._validate_visible_cache()
        return self._num_visible_cache

    def is_face_visible(self, face: int):
        self._validate_visible_cache()
        return self._is_face_visible(face)

    @property
    def block_id(self) -> BlockID:
        return self.data.id

    @property
    def name(self) -> str:
        return self.data.name

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
        return self.num_visible_faces != 0

    def hide_face(self, side: Union[int, Side]):
        if isinstance(side, Side):
            side = _side_cache[side]
        self._face_is_not_visible(side)

    def reveal_face(self, side: Union[int, Side]):
        if isinstance(side, Side):
            side = _side_cache[side]
        self._face_is_visible(side)

    def to_nbt(self) -> nbtlib.Compound:
        return self.data.to_nbt()

    @classmethod
    def from_nbt(cls, block_in_chunk_index: int, nbt_data: nbtlib.Compound):
        block_data = _block_data_cache[
            _block_id_cache[int(nbt_data["BlockID"])]]  # Get Base BlockData from BlockID Enum
        # TODO: obtain saved block states from nbt compound tag
        return Block(block_in_chunk_index, block_data)  # Create Block Object


class BlockShape(TexQuad):
    def __init__(self, block_data: np.ndarray, face_data: np.ndarray):
        self._inst_face_data: np.ndarray = face_data
        super().__init__(model_mats=block_data)

    def list_inst_attrs(self) -> List[VA._Base]:
        return [
            VA.Float.Mat4("a_Model", divisor=6),
            VA.Int.Scalar("a_FaceID", divisor=1),
            VA.Int.Scalar("a_TexPosID", divisor=1),
            VA.Int.Scalar("a_TexSizeID", divisor=1),
        ]

    def get_instance_data_lists(self) -> List[np.ndarray]:
        return super().get_instance_data_lists() + [
            self._inst_face_data
        ]


if __name__ == "__main__":
    def test():
        Block(0)


    num_tests = 1_000_000
    filename = "block"
    cProfile.run(f"[test() for _ in range({num_tests})]", f"{filename}.prof")

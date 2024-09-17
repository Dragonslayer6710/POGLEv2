from __future__ import annotations

from Face import *
from Face import _face_tex_cache

from POGLE.Physics.Collisions import PhysicalBox, AABB

import glm
import numpy as np
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


_block_state_cache: ImDict[BlockID, BlockState] = ImDict({
    BlockID.Air: BlockState(BlockID.Air, [FaceTex.Null] * 6, False, False),
    BlockID.Dirt: BlockState(BlockID.Dirt, [FaceTex.Dirt] * 6),
    BlockID.Grass: BlockState(BlockID.Grass, [FaceTex.GrassSide] * 4 + [FaceTex.GrassTop, FaceTex.Dirt]),
    BlockID.Stone: BlockState(BlockID.Stone, [FaceTex.Stone] * 6)
})

NULL_BLOCK_STATE = _block_state_cache[BlockID.Air]

_all_block_states_set = True


class Side(Renum):
    Null = -1,
    West = auto()
    South = auto()
    East = auto()
    North = auto()
    Top = auto()
    Bottom = auto()


_side_cache: ImDict[BlockID, BlockState] = ImDict(
    {member.value: member for member in Side}
)

_opposite_side: ImDict[Side, Side] = {
    Side.West: Side.East,
    Side.South: Side.North,
    Side.East: Side.West,
    Side.North: Side.South,
    Side.Top: Side.Bottom,
    Side.Bottom: Side.Top
}

BLOCK_EXTENTS: glm.vec3 = glm.vec3(1)
BLOCK_EXTENTS_HALF: glm.vec3 = BLOCK_EXTENTS / 2
class Block(PhysicalBox):
    def __init__(self, pos: glm.vec3, block_state: BlockState, section: 'Section' = None):
        super().__init__(AABB.from_pos_size(pos))
        self._neighbour_refs: ImDict[Side, int] = {}

        self.state: BlockState = block_state

        self._visible_faces_bitmask: int = 0
        self._visible_faces_cache: int = 0
        self._visible_faces_cache_valid: bool = False

    def neighbour(self, side: Union[int, Side], blocks: List[Block]):
        return

    def set_state(self, id: BlockID):
        was_solid = self.is_solid
        was_opaque = self.is_opaque
        self.state = _block_state_cache[id]
        if was_solid and not self.is_solid:
            pass


    def update_face_visibility(self):
        pass


    @property
    def id(self) -> BlockID:
        return self.state.block_id

    @property
    def name(self) -> str:
        return self.state.name

    @property
    def value(self) -> int:
        return self.state.value

    @property
    def is_air(self) -> bool:
        return self.id == BlockID.Air

    @property
    def is_solid(self) -> bool:
        return self.state.is_solid

    @property
    def is_opaque(self) -> bool:
        return self.state.is_opaque

    @property
    def is_visible(self) -> bool:
        return self._visible_faces_bitmask

    @property
    def visible_faces(self) -> int:
        if self.is_air:
            return 0
        if not self._visible_faces_cache_valid:
            self._visible_faces_cache = bin(self._visible_faces_bitmask).count("1")
        return self._visible_faces_cache

    def is_side_visible(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = _side_cache[side]
        return (self._visible_faces_bitmask & (1 << side.value)) != 0

    def hide_face(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = _side_cache[side]
        if self.is_side_visible(side):
            self._visible_faces_bitmask &= ~(1 << side.value)
            self._visible_faces_cache -= 1

    def reveal_face(self, side: Union[int, Side]):
        if isinstance(side, int):
            side = _side_cache[side]
        if not self.is_side_visible(side):
            self._visible_faces_bitmask |= (1 << side.value)
            self._visible_faces_cache += 1

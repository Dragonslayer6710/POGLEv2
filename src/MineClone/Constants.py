from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Union, Optional

import numpy as np
import glm

from POGLE.Physics.Collisions import AABB


@dataclass
class _Constants:
    WIDTH: int
    HEIGHT: Optional[int] = None

    def __post_init__(self):
        self.WIDTH_HALF: float = self.WIDTH / 2
        self.WIDTH_HALF_INT: int = int(self.WIDTH_HALF)
        if self.HEIGHT is not None:
            self.HEIGHT_HALF: float = self.HEIGHT / 2
            self.HEIGHT_HALF_INT: int = int(self.HEIGHT_HALF)
        self.WIDTH_RANGE: range = range(self.WIDTH)
        if self.HEIGHT is not None:
            self.HEIGHT_RANGE: range = range(self.HEIGHT)
            self.HEIGHT_RANGE_REVERSE: range = range(self.HEIGHT - 1, -1, -1)
        self.EXTENTS: Union[glm.vec2, glm.vec3] = glm.vec2(self.WIDTH) if self.HEIGHT is None \
            else glm.vec3(self.WIDTH, self.HEIGHT, self.WIDTH)
        self.EXTENTS_INT: Union[glm.ivec2, glm.ivec3] = glm.ivec2(self.EXTENTS) if self.HEIGHT is None \
            else glm.ivec3(self.EXTENTS)
        self.EXTENTS_HALF: Union[glm.vec2, glm.vec3] = self.EXTENTS / 2
        self.EXTENTS_HALF_INT: Union[glm.vec2, glm.vec3] = self.EXTENTS_INT // 2
        self.NONE_LIST: Union[
            List[List[Optional[object]]],
            List[List[List[Optional[object]]]]
        ] = [  # Z Axis is outer
            [  # X Axis is inner
                None for _ in self.WIDTH_RANGE
            ] for _ in self.WIDTH_RANGE
        ] if self.HEIGHT is None else [  # Y Axis is Outer
            [  # Z Axis is Depth 1
                [  # X Axis is Depth 2
                    None for _ in self.WIDTH_RANGE
                ] for _ in self.WIDTH_RANGE
            ] for _ in self.HEIGHT_RANGE
        ]
        self._SIZE: int = self.WIDTH ** 2
        if self.HEIGHT is not None:
            self._SIZE *= self.HEIGHT


@dataclass
class _CHUNK(_Constants):
    def __init__(self, WIDTH: int, HEIGHT: int, LOWEST_Y: int):
        super().__init__(WIDTH, HEIGHT)
        self.NUM_BLOCKS = self._SIZE
        self.AABB = AABB.from_pos_size(size=self.EXTENTS)

        self.SECTION_NUM_BLOCKS: int = self._SIZE // self.HEIGHT
        self.SECTION_RANGE: range = range(self.SECTION_NUM_BLOCKS)
        self.NULL_HEIGHT_MAP: List[List[int]] = [  # Z Axis is the outer
            [  # X Axis is the inner
                None for _ in self.WIDTH_RANGE
            ] for _ in self.WIDTH_RANGE
        ]
        self.LOWEST_Y = LOWEST_Y




@dataclass
class _REGION(_Constants):
    def __init__(self, WIDTH: int):
        super().__init__(WIDTH)
        self.NUM_CHUNKS = self._SIZE
        self.NUM_BLOCKS: int = self.NUM_CHUNKS * CHUNK.NUM_BLOCKS

        self.BLOCK_EXTENTS: glm.ivec3 = glm.ivec3(self.WIDTH, 1, self.WIDTH) * glm.ivec3(CHUNK.EXTENTS)
        self.BLOCK_WIDTH: int = self.BLOCK_EXTENTS[0]

        self.AABB = AABB.from_pos_size(size=self.BLOCK_EXTENTS)

        self.CHUNK_OFFSETS: List[List[glm.vec2]] = [  # Z Axis is the outer
            [  # X Axis is the Inner
                (glm.vec2(x, z) - (self.WIDTH + 1) // 2)
                * CHUNK.WIDTH + CHUNK.WIDTH // 2
                for x in self.WIDTH_RANGE
            ] for z in self.WIDTH_RANGE
        ]

        # pos = glm.vec2(0) + self.CHUNK_OFFSETS[15][15]
        # print(f"Min: {pos - CHUNK.EXTENTS_HALF.xz}"
        #       f"\nPos: {pos}"
        #       f"\nMax: {pos + CHUNK.EXTENTS_HALF.xz}")
        # print()
        # pos = glm.vec2(0) + self.CHUNK_OFFSETS[16][16]
        # print(f"Min: {pos - CHUNK.EXTENTS_HALF.xz}"
        #       f"\nPos: {pos}"
        #       f"\nMax: {pos + CHUNK.EXTENTS_HALF.xz}")
        # quit()




@dataclass
class _WORLD(_Constants):
    def __init__(self, WIDTH: int, SPAWN_CHUNK_WIDTH: int, INITIAL_SPAWN_CHUNK_WIDTH: int):
        if WIDTH % 3:
            raise RuntimeError(
                f"World Region Width: {WIDTH} is not acceptable."
                f" It must be a multiple of 3"
            )
        elif not (SPAWN_CHUNK_WIDTH % 2):
            raise RuntimeError(f"World Spawn Chunk Width: {SPAWN_CHUNK_WIDTH} is not acceptable."
                               f" It must be an odd number!")
        elif not (INITIAL_SPAWN_CHUNK_WIDTH % 2):
            raise RuntimeError(f"Initial World Spawn Chunk Width: {INITIAL_SPAWN_CHUNK_WIDTH} is not acceptable."
                               f" It must be an odd number!")
        super().__init__(WIDTH)

        self.MID_POINT: glm.ivec2 = self.EXTENTS_HALF_INT + 1 - - self.EXTENTS_HALF_INT

        self.ORIG_WIDTH_RANGE = range(-self.EXTENTS_HALF_INT[0], self.EXTENTS_HALF_INT[0] + 1)

        self.NUM_REGIONS = self._SIZE
        self.NUM_CHUNKS: int = self.NUM_REGIONS * REGION.NUM_CHUNKS
        self.NUM_BLOCKS: int = self.NUM_REGIONS * REGION.NUM_BLOCKS

        self.BLOCK_EXTENTS: glm.ivec3 = glm.ivec3(self.WIDTH, 1, self.WIDTH) * REGION.BLOCK_EXTENTS
        self.BLOCK_WIDTH: int = self.BLOCK_EXTENTS[0]

        self.AABB = AABB.from_pos_size(glm.vec3(0, CHUNK.HEIGHT // 2, 0), size=self.BLOCK_EXTENTS)

        self.SPAWN_CHUNK_WIDTH: int = SPAWN_CHUNK_WIDTH
        self.NUM_SPAWN_CHUNKS: int = self.SPAWN_CHUNK_WIDTH ** 2

        self.SPAWN_CHUNK_WIDTH_HALF = self.SPAWN_CHUNK_WIDTH // 2
        self.SPAWN_CHUNK_RANGE: range = range(-self.SPAWN_CHUNK_WIDTH_HALF, self.SPAWN_CHUNK_WIDTH_HALF + 1)
        self.SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS: Tuple[int] = tuple(
            a * CHUNK.WIDTH for a in self.SPAWN_CHUNK_RANGE)

        self.INITIAL_SPAWN_CHUNK_WIDTH = INITIAL_SPAWN_CHUNK_WIDTH
        self.INITIAL_NUM_SPAWN_CHUNKS = self.INITIAL_SPAWN_CHUNK_WIDTH ** 2

        self.INITIAL_SPAWN_CHUNK_WIDTH_HALF = self.INITIAL_SPAWN_CHUNK_WIDTH // 2
        self.INITIAL_SPAWN_CHUNK_RANGE: Tuple[glm.vec3] = range(-self.INITIAL_SPAWN_CHUNK_WIDTH_HALF,
                                                                self.INITIAL_SPAWN_CHUNK_WIDTH_HALF + 1)
        self.INITIAL_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS: Tuple[int] = tuple(
            a * CHUNK.WIDTH for a in self.INITIAL_SPAWN_CHUNK_RANGE)

        self.INITIAL_SPAWN_CHUNK_POSITIONS: Tuple[glm.ivec2] = tuple(
            glm.ivec2(x, z)
            for x in self.INITIAL_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS
            for z in self.INITIAL_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS
        )

        self.SPAWN_CHUNK_POSITIONS: Tuple[glm.ivec2] = tuple(
            sorted(
                tuple(
                    filter(
                        lambda x: x not in self.INITIAL_SPAWN_CHUNK_POSITIONS,
                        [
                            glm.ivec2(x, z)
                            for x in self.SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS
                            for z in self.SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS
                        ]
                    )
                ),
                key=lambda pos: glm.length(glm.vec2(pos))  # Sort by distance from origin
            )
        )

        self.SURFACE_HEIGHT = 64
        self.BIOME_VARIATION = 10

def w_to_wr(w: Union[glm.vec2, glm.ivec2, glm.ivec3, glm.vec3]) -> glm.ivec2:
    if len(w) == 3:
        return w_to_wr(glm.vec2(w.x, w.z) if isinstance(w, glm.vec3) else glm.ivec2(w.x, w.z))
    else:
        return glm.ivec2((w + WORLD.BLOCK_WIDTH // 2) // REGION.BLOCK_WIDTH)


def w_to_rc(w: Union[glm.vec2, glm.ivec2, glm.ivec3, glm.vec3]) -> glm.ivec2:
    if len(w) == 3:
        return w_to_rc(glm.vec2(w.x, w.z) if isinstance(w, glm.vec3) else glm.ivec2(w.x, w.z))
    else:
        return glm.ivec2(((w + WORLD.BLOCK_WIDTH // 2) % REGION.BLOCK_WIDTH) // CHUNK.WIDTH)


def w_to_cb(w: Union[glm.ivec3, glm.vec3]) -> glm.ivec3:
    if isinstance(w, glm.ivec3):
        w = glm.vec3(w)
    w_xz: glm.vec2 = (w.xz + WORLD.BLOCK_WIDTH) % CHUNK.WIDTH
    return glm.ivec3(w_xz[0], w.y, w_xz[1])

CHUNK = _CHUNK(16, 128, 0)
REGION = _REGION(32)
WORLD = _WORLD(15, 19, 3)
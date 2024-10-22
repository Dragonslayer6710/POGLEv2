from __future__ import annotations

from itertools import repeat

from io import BytesIO
import zlib

import os.path
import pickle

import nbtlib
import numpy as np

from Block import *
from Entity import *
from Biome import *
from dataclasses import dataclass, field
from typing import Set

if TYPE_CHECKING:
    from Region import Region

from POGLE.Physics.SpatialTree import Octree

CHUNK_BLOCK_WIDTH = 16
_CHUNK_BLOCK_HORIZONTAL_ID_RANGE = range(CHUNK_BLOCK_WIDTH)

CHUNK_BLOCK_HEIGHT = 256
_CHUNK_BLOCK_VERTICAL_ID_RANGE = range(CHUNK_BLOCK_HEIGHT)

CHUNK_BLOCK_EXTENTS = glm.vec3(CHUNK_BLOCK_WIDTH, CHUNK_BLOCK_HEIGHT, CHUNK_BLOCK_WIDTH)

CHUNK_NUM_BLOCKS = int(np.prod(CHUNK_BLOCK_EXTENTS))
CHUNK_BLOCK_NONE_LIST = [None] * CHUNK_NUM_BLOCKS
_CHUNK_BLOCK_ID_RANGE = range(CHUNK_NUM_BLOCKS)

SECTION_NUM_BLOCKS: int = CHUNK_BLOCK_WIDTH ** 2
SECTION_BLOCK_RANGE: range = range(SECTION_NUM_BLOCKS)
SECTION_STARTS: List[int] = [y * SECTION_NUM_BLOCKS for y in _CHUNK_BLOCK_VERTICAL_ID_RANGE]
SECTION_ENDS: List[int] = [SECTION_STARTS[y] + SECTION_NUM_BLOCKS for y in _CHUNK_BLOCK_VERTICAL_ID_RANGE]

SECTION_NUM_BIOMES = 1
CHUNK_NUM_BIOMES = SECTION_NUM_BIOMES * CHUNK_BLOCK_HEIGHT
CHUNK_BIOME_NONE_LIST = [None] * CHUNK_NUM_BIOMES

_CHUNK_NULL_HEIGHT_MAP: Tuple[int] = tuple([0] * int(CHUNK_NUM_BLOCKS / CHUNK_BLOCK_HEIGHT))

CHUNK_BASE_AABB = AABB.from_pos_size(size=CHUNK_BLOCK_EXTENTS)

_BLOCK_MASKS = tuple(1 << i for i in _CHUNK_BLOCK_ID_RANGE)

CHUNK_LOWEST_Y = 0


def block_state_compound(blocks: List[Block]) -> nbtlib.Compound:
    palette: List[BlockID] = []
    nbt_palette: List[nbtlib.Compound] = []
    data: List[int] = []

    for block in blocks:
        index = None
        try:
            index = palette.index(block.block_id)
        except ValueError:
            index = palette.__len__()
            palette.append(block.block_id)
            nbt_palette.append(block.to_nbt())
        finally:
            data.append(index)
    compound = nbtlib.Compound({
        "palette": nbtlib.List(nbt_palette)
    })
    if len(palette) > 1:
        compound["data"] = nbtlib.LongArray(data, length=4096)
    return compound


def biome_compound(biomes: List[Biome]) -> nbtlib.Compound:
    if BIOME_NOT_IMPLEMENTED:
        return None
    palette: List[BiomeID] = []
    nbt_palette: List[nbtlib.Compound] = []
    data: List[int] = []
    for biome in biomes:
        try:
            index = palette.index(biome.biome_id)
        except ValueError:
            index = palette.__len__()
            palette.append(biome.biome_id)
            nbt_palette.append(biome.to_nbt())
        finally:
            data.append(index)
    compound = nbtlib.Compound({
        "palette": nbtlib.List(nbt_palette)
    })
    if len(palette) > 1:
        compound["data"] = nbtlib.LongArray(data, length=64)
    return compound


compression_type = 2


def _compress_chunk_data(chunk_data):
    if compression_type == 2:
        return zlib.compress(chunk_data)
    raise ValueError("Unknown compression type reference provided")


def _decompress_chunk_data(compressed_data, compression_type):
    if compression_type == 2:
        return zlib.decompress(compressed_data)
    raise ValueError("Unknown compression type reference provided")


@dataclass
class Chunk(MCPhys, aabb=CHUNK_BASE_AABB):
    blocks: Tuple[Block, ...] = field(default_factory=lambda: [Block(i) for i in _CHUNK_BLOCK_ID_RANGE])
    entities: List[Entity] = field(default_factory=list)
    tile_entities: List[TileEntity] = field(default_factory=list)
    height_map: Tuple[int] = field(default_factory=lambda: copy(_CHUNK_NULL_HEIGHT_MAP))
    biome_data: Tuple[Biome] = field(default_factory=lambda: [None for _ in range(CHUNK_NUM_BIOMES)])

    timestamp: Optional[int] = 0
    region: Optional[Region] = None

    def __post_init__(self):
        super().__post_init__()
        self.uninitialized: bool = True

        self.octree: Optional[Octree] = None

        self._solid_dict: Dict[int, Block] = {}
        self._solid_bitmask: int = 0
        self._solid_list_cache: List[Block] = []
        self._num_solid_cache: int = 0
        self._solid_cache_valid: bool = False

        self._renderable_dict: Dict[int, Block] = {}
        self._renderable_bitmask: int = 0
        self._renderable_list_cache: List[Block] = []
        self._num_renderable_cache: int = 0
        self._renderable_cache_valid: bool = False

        self.status = "empty"
        self.last_update: int = 0

        if self.region:
            self.initialize()

    def initialize(self, region: Optional[Region] = None):
        if region:
            if self.region:
                raise RuntimeError("Attempted to set region of a chunk already set in a region")
            self.region = region

        self.pos += self.region.pos

        self.octree = Octree(self.bounds)

        self.pos -= CHUNK_BLOCK_EXTENTS / 2
        for block in self.blocks:
            block.initialize(self)
            self.update_block_in_lists(block)
        self.pos += CHUNK_BLOCK_EXTENTS / 2

        self.uninitialized = False
        self.octree.update()

    def _is_block_solid(self, block: Block):
        return self._solid_bitmask & _BLOCK_MASKS[block.index]

    def _block_is_solid(self, block: Block):
        if not self._is_block_solid(block):  # Check if Block was not solid
            self._solid_dict[block.index] = block
            self._solid_bitmask |= _BLOCK_MASKS[block.index]
            self._solid_cache_valid = False

            if self.uninitialized:
                self.octree.queue_insert(block)
            else:
                self.octree.insert(block)

    def _block_is_not_solid(self, block: Block):
        if self._is_block_solid(block):  # Check if Block was solid
            self._solid_dict.pop(block.index)
            self._solid_bitmask &= ~_BLOCK_MASKS[block.index]
            self._solid_cache_valid = False

            if self.uninitialized:
                self.octree.queue_remove(block)
            else:
                self.octree.remove(block)

    def _is_block_renderable(self, block: Block):
        return self._renderable_bitmask & _BLOCK_MASKS[block.index]

    def _block_is_renderable(self, block: Block):
        if not self._is_block_renderable(block):  # Check if Block was not renderable
            self._renderable_dict[block.index] = block
            self._renderable_bitmask |= _BLOCK_MASKS[block.index]
            self._renderable_cache_valid = False

    def _block_is_not_renderable(self, block: Block):
        if self._is_block_renderable(block):  # Check if Block was renderable
            self._renderable_dict.pop(block.index)
            self._renderable_bitmask &= ~_BLOCK_MASKS[block.index]
            self._renderable_cache_valid = False

    def update_block_in_lists(self, block: Block):
        if block.is_solid:
            self._block_is_solid(block)
        else:
            self._block_is_not_solid(block)

        if block.is_renderable:
            self._block_is_renderable(block)
        else:
            self._block_is_not_renderable(block)

    def _validate_solid_cache(self):
        self._solid_list_cache = self._solid_dict.values()
        self._num_solid_cache = len(self._solid_list_cache)
        self._solid_cache_valid = True

    @property
    def solid_blocks(self):
        if not self._solid_cache_valid:
            self._validate_solid_cache()
        return self._solid_list_cache

    @property
    def num_solid_blocks(self):
        if not self._solid_cache_valid:
            self._validate_solid_cache()
        return self._num_solid_cache

    @property
    def is_solid(self) -> bool:
        return bool(self._num_solid_cache)

    def _validate_renderable_cache(self):
        self._renderable_list_cache = self._renderable_dict.values()
        self._num_renderable_cache = len(self._renderable_list_cache)
        self._renderable_cache_valid = True

    @property
    def renderable_blocks(self):
        if not self._renderable_cache_valid:
            self._validate_renderable_cache()
        return self._renderable_list_cache

    @property
    def num_renderable_blocks(self):
        if not self._renderable_cache_valid:
            self._validate_renderable_cache()
        return self._num_renderable_cache

    @property
    def is_renderable(self) -> bool:
        return bool(self._num_renderable_cache)

    def to_nbt(self) -> nbtlib.Compound:
        return nbtlib.Compound({
            "index": nbtlib.Int(self.index),
            "DataVersion": nbtlib.Int(1),
            "xPos": nbtlib.Int(self.pos.x),  # x position of chunk in chunks not relative to region
            "zPos": nbtlib.Int(self.pos.z),  # z position of chunk in chunks not relative to region
            "yPos": nbtlib.Int(CHUNK_LOWEST_Y),  # lowest Y section position in the chunk
            "Status": nbtlib.String(self.status),
            "LastUpdate": nbtlib.Long(self.last_update),
            "sections": nbtlib.List(
                [
                    nbtlib.Compound({
                        "Y": nbtlib.Byte(y - 128),
                        "block_states": block_state_compound(
                            self.blocks[y * SECTION_NUM_BLOCKS: y * SECTION_NUM_BLOCKS + SECTION_NUM_BLOCKS]
                        ),
                        # "biomes": biome_compound(
                        #    None#self.blocks[y * SECTION_NUM_BIOMES: y * SECTION_NUM_BIOMES + SECTION_NUM_BIOMES]
                        # ),
                        "BlockLight": nbtlib.ByteArray(length=2048),
                        "SkyLight": nbtlib.ByteArray(length=2048),
                    }) for y in range(CHUNK_BLOCK_HEIGHT)
                ]
            ),
            "block_entities": nbtlib.List(
                []
            ),
            # TODO: save octree to chunk data
            "octree": nbtlib.ByteArray(None),
            "CarvingMasks": nbtlib.Compound({
                "AIR": nbtlib.ByteArray(),
                "LIQUID": nbtlib.ByteArray(),
            }),
            "Heightmaps": nbtlib.Compound({
                "MOTION_BLOCKING": nbtlib.LongArray(),
                "MOTION_BLOCKING_NO_LEAVES": nbtlib.LongArray(),
                "OCEAN_FLOOR": nbtlib.LongArray(),
                "OCEAN_FLOOR_WG": nbtlib.LongArray(),
                "WORLD_SURFACE": nbtlib.LongArray(),
                "WORLD_SURFACE_WG": nbtlib.LongArray(),
            }),
            "Lights": nbtlib.List(
                []
            ),
            "Entities": nbtlib.List(
                []
            ),
            "fluid_ticks": nbtlib.List(
                []
            ),
            "block_ticks": nbtlib.List(
                []
            ),
            "InhabitedTime": nbtlib.Long(),
            "PostProcessing": nbtlib.List(),
            "structures": nbtlib.Compound({
                "References": nbtlib.Compound({
                    "Structure Name": nbtlib.LongArray(
                        []
                    )
                }),
                "starts": nbtlib.Compound({
                    "Structure Name": nbtlib.Compound({
                        # TODO: Structure to nbt
                    })
                })
            })
        })

    @classmethod
    def from_nbt(cls, nbt_data: nbtlib.Compound) -> Chunk:
        blocks = []
        section_block_range = SECTION_BLOCK_RANGE  # Cache SECTION_BLOCK_RANGE
        _Block = Block  # Cache Block class
        for section in nbt_data["sections"]:
            y = section["Y"] + 128  # Reverse the Y calculation
            block_states = section["block_states"]  # Call function to parse block states
            block_states_palette = block_states["palette"]
            block_states_data = block_states.get("data")
            if block_states_data is not None:
                # blocks.extend([
                #    _Block(block_states_palette[palette_index]["BlockID"])
                #    for palette_index in enumerate(block_states_data)
                # ])
                palette_block_ids = [block_states_palette[entry]["BlockID"] for entry in block_states_data]
                blocks.extend(_Block(block_id) for block_id in palette_block_ids)
            else:
                default_block_id = block_states_palette[0]["BlockID"]
                blocks.extend(repeat(_Block(default_block_id), len(section_block_range)))
            # biomes = parse_biomes(section["biomes"])  # Call function to parse biomes
            # block_light = section["BlockLight"]  # Byte array, can be used directly or processed
            # sky_light = section["SkyLight"]

            # # Append the section information into the chunk sections
            # chunk.sections.append({
            #     "Y": y,
            #     "block_states": block_states,
            #     "biomes": biomes,
            #     "block_light": block_light,
            #     "sky_light": sky_light
            # })

        chunk = Chunk(nbt_data["index"], blocks)
        chunk.pos = glm.vec3(nbt_data["xPos"], CHUNK_BLOCK_HEIGHT / 2,
                             nbt_data["zPos"])  # Assuming Position is a class that holds x and z
        chunk.status = nbt_data["Status"]
        chunk.last_update = nbt_data["LastUpdate"]
        # Parse block entities
        # chunk.block_entities = parse_block_entities(nbt_data["block_entities"])

        # Parse carving masks
        # chunk.carving_masks = {
        #     "AIR": nbt_data["CarvingMasks"]["AIR"],
        #     "LIQUID": nbt_data["CarvingMasks"]["LIQUID"]
        # }

        # Parse heightmaps
        # chunk.heightmaps = {
        #     "MOTION_BLOCKING": nbt_data["Heightmaps"]["MOTION_BLOCKING"],
        #     "MOTION_BLOCKING_NO_LEAVES": nbt_data["Heightmaps"]["MOTION_BLOCKING_NO_LEAVES"],
        #     "OCEAN_FLOOR": nbt_data["Heightmaps"]["OCEAN_FLOOR"],
        #     "OCEAN_FLOOR_WG": nbt_data["Heightmaps"]["OCEAN_FLOOR_WG"],
        #     "WORLD_SURFACE": nbt_data["Heightmaps"]["WORLD_SURFACE"],
        #     "WORLD_SURFACE_WG": nbt_data["Heightmaps"]["WORLD_SURFACE_WG"]
        # }

        # # Parse lights, entities, and ticks (fluid/block)
        # chunk.lights = nbt_data["Lights"]
        # chunk.entities = nbt_data["Entities"]
        # chunk.fluid_ticks = nbt_data["fluid_ticks"]
        # chunk.block_ticks = nbt_data["block_ticks"]

        # # Parse other fields
        # chunk.inhabited_time = nbt_data["InhabitedTime"]
        # chunk.post_processing = nbt_data["PostProcessing"]

        # # Parse structures
        # chunk.structures = {
        #     "references": nbt_data["structures"]["References"],
        #     "starts": nbt_data["structures"]["starts"]
        # }

        return chunk  # Return the populated chunk object

    def serialize(self, compress: bool = False) -> bytes:
        with BytesIO() as buffer:
            # Convert the chunk to NBT and write to the buffer
            nbtlib.File(self.to_nbt()).write(buffer)

            # Return the serialized data directly
            chunk_data = buffer.getvalue()
        if compress:
            return _compress_chunk_data(chunk_data)  # Compress chunk data
        else:
            return chunk_data

    @classmethod
    def deserialize(cls, chunk_data: bytes, compression_type: Optional[int] = None) -> Chunk:
        if compression_type is not None:
            chunk_data = _decompress_chunk_data(chunk_data, compression_type)

        # Parse the NBT data to recreate the chunk
        with BytesIO(chunk_data) as f:
            chunk_nbt = nbtlib.File.parse(f)
        return cls.from_nbt(chunk_nbt)


class _Region:
    pos: glm.vec3 = glm.vec3()


if __name__ == "__main__":
    a: Chunk = Chunk(0)
    a.initialize(_Region())


    def test():
        pass


    def test2():
        pass


    num_tests = 10
    print(timeit(test, number=num_tests))
    print(timeit(test2, number=num_tests))

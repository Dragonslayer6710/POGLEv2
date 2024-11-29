from __future__ import annotations

from io import BytesIO
import zlib
from copy import deepcopy

from MineClone.Entity import *
from MineClone.Biome import *
from MineClone.Generation import TerrainGenerator

from dataclasses import dataclass, field
from collections import deque

if TYPE_CHECKING:
    from MineClone.Region import Region


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
    palette: List[BiomeID] = []
    nbt_palette: List[nbtlib.Compound] = []
    data: List[int] = []
    for biome in biomes:
        try:
            index = palette.index(biome.id)
        except ValueError:
            index = palette.__len__()
            palette.append(biome.id)
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


NULL_CHUNK_BLOCK_NDARRAY: np.ndarray = np.empty(CHUNK.NUM_BLOCKS, dtype=np.ndarray)
NULL_CHUNK_BLOCK_LIST: List = [None] * CHUNK.NUM_BLOCKS
NULL_CHUNK_BLOCK_FACE_LIST: List = [None] * CHUNK.NUM_BLOCKS * 6

chunk_gen = TerrainGenerator(42)

@dataclass
class Chunk(MCPhys, aabb=CHUNK.AABB):
    blocks: np.ndarray = field(
        default_factory=lambda: np.empty((CHUNK.HEIGHT, CHUNK.WIDTH, CHUNK.WIDTH), dtype=object)
    )
    entities: List[Entity] = field(default_factory=list)
    tile_entities: List[TileEntity] = field(default_factory=list)
    height_map: List[List[int]] = field(default_factory=lambda: CHUNK.NULL_HEIGHT_MAP.copy())
    biomes: List[List[List[Optional[Biome], ...], ...]] = field(
        default_factory=lambda: np.empty((CHUNK.HEIGHT, CHUNK.WIDTH, CHUNK.WIDTH), dtype=object)
    )

    _from_nbt: bool = False
    timestamp: Optional[int] = 0
    region: Optional[Region] = None

    def __post_init__(self):
        if self.index == 0:
            self.index = glm.ivec2((REGION.WIDTH + 1) // 2)
        super().__post_init__()
        self.chunk_seed: Optional[int] = None
        self.initialized: bool = False

        self._renderable_dict: Dict[int, Block] = {}
        self._renderable_bitmask: int = 0
        self._renderable_list_cache: List[Block] = []
        self._num_renderable_cache: int = 0
        self._renderable_cache_valid: bool = False

        self.status = "empty"
        self.last_update: int = 0

        self._update_deque: deque[Block] = deque()
        self._updated_deque: deque[Block] = deque()
        self._awaiting_update: bool = False

        self.initialized: bool = False

        self.block_face_instance_data: List[glm.ivec3] = deepcopy(NULL_CHUNK_BLOCK_FACE_LIST)
        # self.block_face_ids: np.ndarray = deepcopy(NULL_CHUNK_BLOCK_NDARRAY)
        # self.block_face_tex_ids: np.ndarray = deepcopy(NULL_CHUNK_BLOCK_NDARRAY)
        # self.block_face_tex_size_ids: np.ndarray = deepcopy(NULL_CHUNK_BLOCK_NDARRAY)

        # self.block_instance_data: np.ndarray[np.ndarray[np.ndarray[np.float32]]] = np.empty((CHUNK.NUM_BLOCKS,),
        #                                                                                     dtype=np.ndarray)
        self.block_instance_data: List[glm.mat4] = deepcopy(NULL_CHUNK_BLOCK_LIST)
        self.neighbours: Dict[glm.vec2, Chunk] = {}
        self.block_query_cache: Dict[glm.vec3, Block] = {}

        if self.region:
            self.initialize()

    def initialize(self, region: Optional[Region] = None):
        if region:
            if self.region:
                raise RuntimeError("Attempted to set region of a chunk already set in a region")
            self.region = region

        offset_xz = REGION.CHUNK_OFFSETS[self.index[1]][self.index[0]]
        self.pos = self.region.pos.xyz + glm.vec3(offset_xz[0], 0, offset_xz[1])
        self.region.chunk_query_cache[self.pos.xz] = self
        chunk_gen.gen_chunk(self)
        # for y, section in enumerate(self.blocks):
        #     y_index = y * CHUNK.SECTION_NUM_BLOCKS
        #     for z, yz_plane in enumerate(section):
        #         z_index = z * CHUNK.WIDTH
        #         for x, block in enumerate(yz_plane):
        #             index: int = y_index + z_index + x
        #             if not self._from_nbt:
        #                 height = self.height_map[z][x]
        #                 if height is None:
        #                     height = self.height_map[z][x] = self.generate_height(self.pos.x + x, self.pos.z + z)
        #
        #                 if y <= height:
        #                     if y < height - 4:
        #                         block.set(BlockID.Stone)
        #                     elif y < height - 1:
        #                         block.set(BlockID.Dirt)
        #                     else:
        #                         block.set(BlockID.Grass)
        #
        #             self.block_face_ids[index] = block.face_ids
        #             self.block_face_tex_ids[index] = block.face_tex_ids
        #             self.block_face_tex_size_ids[index] = block.face_tex_sizes
        #             block.pos += glm.vec3(x, y, z)
        #             block.initialize(self)
        #             self.block_instances[index] = np.array(NMM(block.pos, s=glm.vec3(0.5)).to_list())
        #             # self.block_instances[block.index] = NMM(block.pos, s=glm.vec3(0.5))
        #     # quit()
        #     # if z == 1 and x == 1:
        #     #    quit()

        self.enqueue_update()
        self.initialized = True

    def set_block(self, x: int, y: int, z: int, block_id: Union[int, BlockID]):
        self.blocks[y][z][x].set(block_id)

    def enqueue_update(self):
        if self._awaiting_update:
            raise RuntimeError("Attempted to put Chunk into update queue whilst already awaiting an update")
        self._awaiting_update = True
        self.region.enqueue_chunk(self)

    def stack_update(self):
        if self._awaiting_update:
            raise RuntimeError("Attempted to put Chunk into update queue whilst already awaiting an update")
        self._awaiting_update = True
        self.region.stack_chunk(self)

    def enqueue_block(self, block: Block):
        if not block in self._updated_deque:
            self._update_deque.append(block)

    def stack_block(self, block: Block):
        if not block in self._updated_deque:
            self._update_deque.appendleft(block)

    def update(self):
        if not self._awaiting_update:
            return  # Shouldn't be possible
        total_updates = len(self._update_deque)
        cnt = 1
        print(f"Chunk ({self.index.x}, {self.index.y}) has {total_updates} blocks to update")
        while self._update_deque:
            block = self._update_deque.popleft()
            block.update()
            self._updated_deque.append(block)
            # if not cnt % 16:
            #     print(
            #         f"Chunk ({self.index.x}, {self.index.y}) Updated: {cnt}/{total_updates} Blocks!\nCurrent Queue Length: {len(self._update_deque)}")
            # cnt += 1
        self._updated_deque.clear()
        self._awaiting_update = False

    def get_block(self, block_pos: Union[glm.ivec3, glm.vec3]) -> Optional[Block]:
        # If index vec3
        if isinstance(block_pos, glm.ivec3):  # get from nested list
            return self.blocks[block_pos.y][block_pos.z][block_pos.x]

        # Otherwise check cache
        cached_block = self.block_query_cache.get(block_pos)
        if cached_block is not None:  # get from cache
            return cached_block

        # Convert vec3 to index vec3
        local_pos = w_to_cb(block_pos)

        # If in bounds
        if self.bounds.intersect_point(block_pos):
            return self.get_block(local_pos)  # Recurse and get from nested list

        # Out of bounds so get from neighbour
        # Get Chunk index of neighbour
        neighbour_chunk_pos = w_to_rc(block_pos)
        if self.index == neighbour_chunk_pos:
            return None  # Intersects on x and z but not y

        # Get offset in terms of chunks
        chunk_offset = neighbour_chunk_pos - self.index

        neighbour = self.neighbours.get(chunk_offset)
        if neighbour is None:
            neighbour = self.region.get_chunk(neighbour_chunk_pos)
            if neighbour is None:
                return None
            if neighbour.index == self.index:
                raise Exception("Self cannot be neighbour")
            self.neighbours[chunk_offset] = neighbour
        block = neighbour.get_block(local_pos)
        if block is not None:
            if block_pos != block.pos:
                print(self.pos)
                print(self.index)
                print(block_pos)
                print(local_pos)
                quit()
                raise Exception("Wrong Block!")
        self.block_query_cache[block_pos] = block
        return block

    def query(self, bounds: AABB) -> List[Block]:
        blocks = []
        min_block = self.get_block(bounds.min)
        valid_block = min_block is not None
        if valid_block:
            if not min_block.is_solid:
                valid_block = False
            else:
                blocks.append(min_block)
        if not valid_block:
            min_block = self.get_block(bounds.pos)
            valid_block = min_block is not None
            if valid_block:
                if not min_block.is_solid:
                    valid_block = False
                else:
                    blocks.append(min_block)
            if not valid_block:
                max_block = self.get_block(bounds.max)
                valid_block = max_block is not None
                if valid_block:
                    if not max_block.is_solid:
                        valid_block = False
                    else:
                        blocks.append(max_block)
                else:
                    pass#raise RuntimeError("No blocks found for bounds: " + str(bounds))
            return blocks

        max_block = self.get_block(bounds.max)
        if max_block is not min_block:
            for y in range(min_block.index.x + 1, max_block.index.x + 1):
                for z in range(min_block.index.y + 1, max_block.index.y + 1):
                    for x in range(min_block.index.z + 1, max_block.index.z + 1):
                        print((x, y, z))
                        blocks.append(self.blocks[y][z][x])
        return blocks


    def to_nbt(self) -> nbtlib.Compound:
        return nbtlib.Compound({
            "index": nbtlib.Int(self.index),
            "DataVersion": nbtlib.Int(1),
            "xPos": nbtlib.Int(self.pos.x),  # x position of chunk in chunks not relative to region
            "zPos": nbtlib.Int(self.pos.z),  # z position of chunk in chunks not relative to region
            "yPos": nbtlib.Int(CHUNK.LOWEST_Y),  # lowest Y section position in the chunk
            "Status": nbtlib.String(self.status),
            "LastUpdate": nbtlib.Long(self.last_update),
            "sections": nbtlib.List(
                [
                    nbtlib.Compound({
                        "Y": nbtlib.Byte(y - 128),
                        "block_states": block_state_compound(
                            self.blocks[
                            y * CHUNK.SECTION_NUM_BLOCKS: y * CHUNK.SECTION_NUM_BLOCKS + CHUNK.SECTION_NUM_BLOCKS]
                        ),
                        # "biomes": biome_compound(
                        #    None#self.blocks[y * SECTION_NUM_BIOMES: y * SECTION_NUM_BIOMES + SECTION_NUM_BIOMES]
                        # ),
                        "BlockLight": nbtlib.ByteArray(length=2048),
                        "SkyLight": nbtlib.ByteArray(length=2048),
                    }) for y in range(CHUNK.HEIGHT)
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
        section_block_range = CHUNK.SECTION_BLOCK_RANGE  # Cache SECTION_BLOCK_RANGE
        _Block = Block  # Cache Block class
        block_in_chunk_index = 0
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
                blocks.extend(
                    _Block.from_nbt(block_in_chunk_index, block_states_palette[entry]) for entry in block_states_data
                )
            else:
                default_block_nbt = block_states_palette[0]
                blocks.extend(_Block.from_nbt(block_in_chunk_index, default_block_nbt) for _ in section_block_range)
            block_in_chunk_index += 1
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

        chunk = Chunk(int(nbt_data["index"]), blocks, _from_nbt=True)
        chunk.pos = glm.vec3(nbt_data["xPos"], CHUNK.EXTENTS_HALF.y,
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

    def get_shape(self) -> BlockShape:
        return BlockShape(
            self.block_instance_data,
            self.block_face_instance_data
        )

    def get_mesh(self) -> BlockShapeMesh:
        return BlockShapeMesh(self.get_shape())


if __name__ == "__main__":
    class _Region(MCPhys, aabb=REGION.AABB):
        chunks: List[Chunk] = []

        def enqueue_chunk(self, chunk: Chunk):
            self.chunks.append(chunk)

        def update(self):
            for chunk in self.chunks:
                chunk.update()

        def get_block(self, block_pos: glm.vec3):
            return None

    from timeit import timeit
    a: Chunk = Chunk(0)
    a.initialize(_Region())


    def test():
        pass


    def test2():
        pass


    num_tests = 10
    print(timeit(test, number=num_tests))
    print(timeit(test2, number=num_tests))

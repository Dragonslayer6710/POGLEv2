from __future__ import annotations

import pstats

import os.path
import pickle

import glm
import nbtlib
import numpy as np

from Region import *

WORLD_REGION_EXTENTS: glm.vec3 = glm.vec3(WORLD_REGION_WIDTH, 1, WORLD_REGION_WIDTH)
WORLD_NUM_REGIONS: int = int(np.prod(WORLD_REGION_EXTENTS))
WORLD_REGION_NONE_LIST: List[None] = [None] * WORLD_NUM_REGIONS
_WORLD_REGION_ID_RANGE: range = range(WORLD_NUM_REGIONS)

WORLD_CHUNK_WIDTH: int = WORLD_REGION_WIDTH * REGION_CHUNK_WIDTH
WORLD_CHUNK_EXTENTS: glm.vec3 = WORLD_REGION_EXTENTS * REGION_CHUNK_EXTENTS
WORLD_NUM_CHUNKS: int = WORLD_NUM_REGIONS * REGION_NUM_CHUNKS

WORLD_BLOCK_WIDTH: int = WORLD_CHUNK_WIDTH * CHUNK_BLOCK_WIDTH
WORLD_BLOCK_WIDTH_HALF: int = REGION_BLOCK_WIDTH * WORLD_REGION_WIDTH_HALF
WORLD_BLOCK_EXTENTS: glm.vec3 = WORLD_CHUNK_EXTENTS * CHUNK_BLOCK_EXTENTS
WORLD_NUM_BLOCKS: int = WORLD_NUM_CHUNKS * CHUNK_NUM_BLOCKS

WORLD_AABB: AABB = AABB.from_pos_size(size=WORLD_BLOCK_EXTENTS)

WORLD_SPAWN_CHUNK_WIDTH = 19
WORLD_NUM_SPAWN_CHUNKS = WORLD_SPAWN_CHUNK_WIDTH ** 2

if not (WORLD_REGION_WIDTH % 2):
    raise RuntimeError(f"World Spawn Chunk Width: {WORLD_REGION_WIDTH} is not acceptable."
                       f" It must be an odd number!")
WORLD_SPAWN_CHUNK_WIDTH_HALF = WORLD_SPAWN_CHUNK_WIDTH // 2
WORLD_SPAWN_CHUNK_RANGE: range = range(-WORLD_SPAWN_CHUNK_WIDTH_HALF, WORLD_SPAWN_CHUNK_WIDTH_HALF + 1)
WORLD_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS: Tuple[int] = tuple(a * CHUNK_BLOCK_WIDTH for a in WORLD_SPAWN_CHUNK_RANGE)

INITIAL_SPAWN_CHUNK_WIDTH = 3
INITIAL_NUM_SPAWN_CHUNKS = INITIAL_SPAWN_CHUNK_WIDTH ** 2

INITIAL_SPAWN_CHUNK_WIDTH_HALF = INITIAL_SPAWN_CHUNK_WIDTH // 2
INITIAL_SPAWN_CHUNK_RANGE: Tuple[glm.vec3] = range(-INITIAL_SPAWN_CHUNK_WIDTH_HALF, INITIAL_SPAWN_CHUNK_WIDTH_HALF + 1)
INITIAL_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS: Tuple[int] = tuple(a * CHUNK_BLOCK_WIDTH for a in INITIAL_SPAWN_CHUNK_RANGE)

INITIAL_SPAWN_CHUNK_POSITIONS: Tuple[glm.vec3] = tuple(
    glm.vec3(x, 0, z)
    for x in INITIAL_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS
    for z in INITIAL_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS
)

WORLD_SPAWN_CHUNK_POSITIONS: Tuple[glm.vec3] = tuple(
    sorted(
        tuple(
            filter(
                lambda x: x not in INITIAL_SPAWN_CHUNK_POSITIONS,
                [
                    glm.vec3(x, 0, z)
                    for x in WORLD_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS
                    for z in WORLD_SPAWN_CHUNK_BLOCK_WIDTH_OFFSETS
                ]
            )
        ),
        key=lambda pos: glm.length(pos)  # Sort by distance from origin
    )
)

gen_options_buffet = nbtlib.Compound({
    "biome_source": nbtlib.Compound({
        "options": nbtlib.Compound({
            "biomes": nbtlib.List(
                [
                    # A biome ID as a string
                ]
            ),
            "size": nbtlib.Byte(2)  # Size of the biomes
        }),  # Ignored if the biome source ID is vanilla layered
        "type": nbtlib.String("")  # An applicable biome source ID
    }),
    "chunk_generator": nbtlib.Compound({
        "options": nbtlib.Compound({
            "default_block": nbtlib.String("minecraft:stone"),  # A Block ID
            "default_fluid": nbtlib.String("minecraft:water"),  # A Fluid ID
        }),
        "type": nbtlib.String("")  # An applicable chunk generator ID
    })
})

gen_options_superflat = nbtlib.Compound({
    "structures": nbtlib.Compound({
        # <structure name>: nbtlib.Compound({
        #   <parameter name>: nbtlib.String(),  The parameter value is a number represented by a string
        # }),  An empty compound named as the structure
    }),
    "layers": nbtlib.List(
        [
            # Block: str,  The block ID for the layer
            # height: int/byte/Short  The height of the layer
        ]  # A Layer
    ),
    "biome": nbtlib.String(""),  # The biome ID
    "flat_world_options": nbtlib.String(""),  # The unescaped "generator-settings" string.
})

_REGION_MASKS = tuple(1 << i for i in _WORLD_REGION_ID_RANGE)


@dataclass
class World(MCPhys, aabb=WORLD_AABB):
    player: Player = field(default_factory=Player)
    world_name: str = "world"
    _from_file: bool = False
    file_location: str = os.getcwd()

    def __post_init__(self):
        super().__post_init__()

        self.file_path = f"{self.file_location}\\{self.world_name}"
        del self.file_location

        self._renderable_dict: Dict[int, Region] = {}
        self._renderable_bitmask: int = 0
        self._renderable_list_cache: List[Region] = []
        self._num_renderable_cache: int = 0
        self._renderable_cache_valid: bool = False

        self.regions = copy(WORLD_REGION_NONE_LIST)

        self._update_deque: deque[Region] = deque()

        if self._from_file:
            for region_file in os.listdir(f"{self.file_path}\\region"):
                wr_coords = glm.vec2(tuple(int(i) for i in region_file.split("r.")[1].split(".mcr")[0].split(".")))
                wrid = wr_coords_to_wrid(wr_coords)
                self.region_from_file(wrid)
        else:
            # Initialise Spawn Region
            self.init_region()
            print(f"Generating {INITIAL_NUM_SPAWN_CHUNKS} Initial spawn chunks...")
            cnt = 1
            for i, w_coords in enumerate(INITIAL_SPAWN_CHUNK_POSITIONS):
                self.generate_chunk(w_coords)
                print(f"{cnt}/{INITIAL_NUM_SPAWN_CHUNKS} Initial Spawn Chunks Generated")
                cnt += 1
            self.update()
            self.to_file()

    def enqueue_region(self, region: Region):
        self._update_deque.append(region)

    def stack_region(self, region: Region):
        self._update_deque.appendleft(region)

    def update(self):
        while self._update_deque:
            region = self._update_deque.popleft()
            region.update()
            self.update_region_in_lists(region)

    def generate_chunk(self, wrid: Union[int, glm.vec3], rcid: Optional[Union[int, glm.vec3]] = None) -> Chunk:
        if isinstance(wrid, int):
            if rcid is None:
                raise ValueError(f"wrid of {wrid} provided without an rcid value!")
            elif isinstance(rcid, glm.vec3):
                rcid: int = rc_coords_to_rcid(rcid)
        elif isinstance(wrid, glm.vec3):
            if rcid is None:
                rcid: int = w_coords_to_rcid(wrid)
            wrid: int = w_coords_to_wrid(wrid)
        else:
            raise TypeError("wrid provided as invalid type")
        region = self.get_region(wrid)
        region.init_chunk(rcid)

    def _get_region_id(self, x: int, z: int) -> int:
        return (z * WORLD_REGION_WIDTH) + x

    def init_region(self, wr_coords: Optional[glm.vec2] = None):
        if wr_coords is None:
            wr_coords = glm.vec2()
        wrid = wr_coords_to_wrid(wr_coords)
        region_pos = wr_coords_to_w_coords(wr_coords)
        self.set_region(wrid, Region(wrid))
        region = self.get_region(wrid)
        region.initialize(region_pos, self)

    def _is_index_out_of_bounds(self, region_index):
        if not -1 < region_index < WORLD_NUM_REGIONS:
            raise ValueError(f"Region Index: {region_index} is out of world bounds")

    def set_region(self, region_index: int, region: Region):
        self._is_index_out_of_bounds(region_index)
        self.regions[region_index] = region
        if not region._awaiting_update:
            self.enqueue_region(region)

    def set_region_from_w_coords(self, w_coords: glm.vec3, region: Region):
        try:
            self.set_region(w_coords_to_wrid(w_coords), region)
        except ValueError:
            raise ValueError(f"World Region Coords: {w_coords} are out of bounds")

    def get_region(self, region_index: int) -> Optional[Region]:
        # self._is_index_out_of_bounds(region_index)
        return self.regions[region_index]

    def get_region_from_w_coords(self, w_coords: glm.vec3) -> Region:
        try:
            return self.get_region(w_coords_to_wrid(w_coords))
        except ValueError:
            raise ValueError(f"World Region Coords: {w_coords} are out of bounds")

    def region_to_file(self, region: Region):
        World.save_region_to_file(region, self.file_path)

    def region_from_file(self, wrid: int) -> Region:
        region = World.load_region_from_file(wrid, self.file_path)
        region.initialize(wrid_to_w_coords(wrid), self)
        self.update_region_in_lists(region)
        self.regions[wrid] = region

    def _is_region_renderable(self, region: Region):
        return self._renderable_bitmask & _REGION_MASKS[region.index]

    def _region_is_renderable(self, region: Region):
        if not self._is_region_renderable(region):  # Check if Chunk was not renderable
            self._renderable_dict[region.index] = region
            self._renderable_bitmask |= _REGION_MASKS[region.index]
            self._renderable_cache_valid = False

    def _region_is_not_renderable(self, region: Region):
        if self._is_region_renderable(region):  # Check if Chunk was renderable
            self._renderable_dict.pop(region.index)
            self._renderable_bitmask &= ~_REGION_MASKS[region.index]
            self._renderable_cache_valid = False

    def update_region_in_lists(self, region: Region):
        if region.is_renderable:
            self._region_is_renderable(region)
        else:
            self._region_is_not_renderable(region)

    def _validate_renderable_cache(self):
        self._renderable_list_cache = self._renderable_dict.values()
        self._num_renderable_cache = len(self._renderable_list_cache)
        self._renderable_cache_valid = True

    @property
    def renderable_regions(self):
        if not self._renderable_cache_valid:
            self._validate_renderable_cache()
        return self._renderable_list_cache

    @property
    def num_renderable_regions(self):
        if not self._renderable_cache_valid:
            self._validate_renderable_cache()
        return self._num_renderable_cache

    @property
    def is_renderable(self) -> bool:
        return bool(self._num_renderable_cache)

    @property
    def spawn_region(self) -> Region:
        return self.regions[(len(self.regions) - 1) // 2]

    @property
    def spawn_chunks(self) -> Tuple[Chunk]:
        return self.spawn_region.non_none_chunks

    def get_block(self, block_pos: glm.vec3) -> Optional[Block]:
        if self.bounds.intersectPoint(block_pos):
            local_pos = [int(i) for i in block_pos.xz - self.pos]
            region = self.get_region(self._get_region_id(*local_pos))
            if region is None:
                return None
            return region.get_block(block_pos)
        else:
            return None

    @staticmethod
    def create_world_directory(world_file_path: str, ):
        if not os.path.exists(f"{world_file_path}\\region"):
            if not os.path.exists(f"{world_file_path}"):
                os.mkdir(f"{world_file_path}")
            os.mkdir(f"{world_file_path}\\region")

    @staticmethod
    def save_region_to_file(region: Region, world_file_path: str = os.getcwd() + "\\world"):
        orel_wr_coords = wrid_to_wr_coords(region.index) - WORLD_REGION_WIDTH_HALF
        region_file_path = world_file_path + f"\\region\\r.{int(orel_wr_coords[0])}.{int(orel_wr_coords[1])}.mcr"
        with open(region_file_path, "wb") as f:
            f.write(region.serialize())

    @staticmethod
    def load_region_from_file(wrid: int, world_file_path: str = os.getcwd() + "\\world\\") -> Region:
        orel_wr_coords = wrid_to_wr_coords(wrid) - WORLD_REGION_WIDTH_HALF
        region_file_path = world_file_path + f"\\region\\r.{int(orel_wr_coords[0])}.{int(orel_wr_coords[1])}.mcr"
        with open(region_file_path, "rb") as f:
            return Region.deserialize(
                f.read(),
                wrid
            )

    def to_file(self):
        World.create_world_directory(self.file_path)  # Make the World directory folder
        level_compound = nbtlib.Compound({
            "allowCommands": nbtlib.Byte(False),
            "BorderCenterX": nbtlib.Double(0),  # Center of the world border on the X coordinate.
            "BorderCenterZ": nbtlib.Double(0),  # Center of the world border on the Z coordinate
            "BorderCenterDamagePerBlock": nbtlib.Double(0.2),
            "BorderSize": nbtlib.Double(60000000),  # Width and length of the border of the border
            "BorderSafeZone": nbtlib.Double(5),
            "BorderSizeLerpTarget": nbtlib.Double(60000000),
            "BorderSizeLerpTime": nbtlib.Long(0),
            "BorderWarningBlocks": nbtlib.Double(5),
            "BorderWarningTime": nbtlib.Double(15),
            "clearWeatherTime": nbtlib.Int(0),  # Number of ticks until "clear weather" has ended
            "CustomBossEvents": nbtlib.Compound({
                "ID": nbtlib.Compound({
                    "Players": nbtlib.List(
                        [
                            # Int Array for player UUIDs called UUID
                        ]
                    ),  # List of players that may see this boss bar
                    "Color": nbtlib.String(""),  # ID of color of bossbar
                    "CreateWorldFog": nbtlib.Byte(False),  # If bossbar should create fog
                    "DarkenScreen": nbtlib.Byte(False),  # If bossbar should darken the sky
                    "Max": nbtlib.Int(0),  # Max health of the bossbar
                    "Value": nbtlib.Int(0),  # Current health of the bossbar
                    "Name": nbtlib.String(""),  # Display name of the bossbar as a JSON text component
                    # ID of the overlay shown over the health bar out of progress, notched_6, notched_10, notched_12 and notched_20
                    "Overlay": nbtlib.String(""),
                    "PlayBossMusic": nbtlib.Byte(False),  # If bossbar should initiate boss music
                    "Visible": nbtlib.Byte(False)  # If bossbar should be visible to listed players
                })  # The ID of a bossbar
            }),  # A Collection of bossbars
            "DataPacks": nbtlib.Compound({}),  # Options for datapacks
            "DataVersion": nbtlib.Int(0),
            # Time of day. 0 is sunrise, 12000 is sunset, 18000 is midnight, 24000 is next day, time does not reset to 0
            "DayTime": nbtlib.Long(0),
            # Current difficulty setting. 0 is Peaceful, 1 is Easy, 2 is Normal and 3 is Hard
            "Difficulty": nbtlib.Byte(2),
            "DifficultyLocked": nbtlib.Byte(False),  # True if the difficulty has been locked.
            "DimensionData": nbtlib.Compound({}),  # This tag contains level data specific to certain dimensions
            "GameRules": nbtlib.Compound({}),  # The gamerules for the world
            "WorldGenSettings": nbtlib.Compound({
                "bonus_chest": nbtlib.Byte(False),
                "seed": nbtlib.Long(0),  # The numerical seed for the world
                "generate_features": nbtlib.Byte(False),  # Whether structures should be generated or not
                "dimensions": nbtlib.Compound({
                    "Dimension ID": nbtlib.Compound({})  # The root for generator settings
                })  # Contains all the dimensions
            }),  # Generation settings for each dimension
            "GameType": nbtlib.Int(0),  # Gamemode
            "generatorName": nbtlib.String(""),  # Name of generator such as default, flat, largeBiomes etc.
            "generatorOptions": nbtlib.Compound({}),  # See Generator Options objects
            "generatorVersion": nbtlib.Int(0),
            "hardcore": nbtlib.Byte(False),
            # Normally true after a world has been initialized properly after creation
            "initialized": nbtlib.Byte(False),
            "LastPlayed": nbtlib.Long(0),  # Unix time in milliseconds when the level was last loaded
            "LevelName": nbtlib.String(self.world_name),
            "MapFeatures": nbtlib.Byte(True),  # True if map generator should place structures such as Villages
            "Player": self.player.to_nbt(),  # State of Singleplayer player
            "raining": nbtlib.Byte(False),  # True if the level is currently experiencing rain, snow and cloud cover
            # The number of ticks before "raining" is toggled and this value gets set to another random value
            "rainTime": nbtlib.Int(0),
            "RandomSeed": nbtlib.Long(0),  # The random level seed used to generate consistent terrain
            "SizeOnDisk": nbtlib.Long(0),  # Estimated size in bytes of the level
            "SpawnX": nbtlib.Int(0),  # X coordinate of world spawn
            "SpawnY": nbtlib.Int(256),  # Y coordinate of world spawn
            "SpawnZ": nbtlib.Int(0),  # Z coordinate of world spawn
            "thundering": nbtlib.Byte(False),
            # The number of ticks before "thundering" is toggled and this value gets set to another random value
            "thunderTime": nbtlib.Int(0),
            "Time": nbtlib.Long(0),  # The number of ticks since the start of the level
            "version": nbtlib.Int(0),  # NBT version of level
            "Version": nbtlib.Compound({
                "Id": nbtlib.Int(0),  # An integer displaying the data version
                "Name": nbtlib.String(""),  # The version name as a string
                "Series": nbtlib.String(""),  # Developing series
                "Snapshot": nbtlib.Byte(False)  # Whether the version is a snapshot or not
            }),  # Information about the Minecraft version the world was saved in
            # The UUID of the current wandering trader in the world saved as four ints
            "WanderingTraderId": nbtlib.IntArray([0, 0, 0, 0]),
            # The current chance of the wandering trader spawning next attempt;
            # this value is the percentage and will be divided by 10 when loaded by the game,
            # for example a value of 50 means 5.0% chance
            "WanderingTraderSpawnChance": nbtlib.Int(0),
            # The amount of ticks until another wandering trader is attempted to spawn
            "WanderingTraderSpawnDelay": nbtlib.Int(0),
            "WasModded": nbtlib.Byte(False)  # True if the world was opened in a modified version
        })

        with open(f"{self.file_path}\\level.dat", "wb") as f:
            nbtlib.File(level_compound).write(f)  # Write to level.dat
        for region in self.regions:
            if region is not None:
                self.region_to_file(region)  # Write all notnull Regions to files

    @classmethod
    def from_file(cls, world_file_path: str = os.getcwd() + "\\world\\"):
        with open(f"{world_file_path}\\level.dat", "rb") as f:
            level_compound = nbtlib.File.parse(f)
        player = Player()  # TODO: something to do with Player from level_compound
        split_subs = world_file_path.split("\\")
        return World(
            player=player,
            world_name=split_subs[-2] if world_file_path[-1:] == "\\" else split_subs[-1],
            _from_file=True
        )

    def get_shape(self):
        return BlockShape(self._block_instances, self._face_instances)
        #mesh = ShapeMesh(bs)
do_profile = True
load_from_file = False
if __name__ == "__main__":
    def profile_test():
        if load_from_file:
            world = World.from_file()
        else:
            world = World(0)
        return world


    def test():
        world = profile_test()

        # spawn_region: Region = world.regions[4]
        # solid_blocks_in_spawn_chunks = list(
        #    filter(
        #        lambda x: x is not None,
        #        [
        #            f"{chunk.index}: {chunk.num_solid_blocks}" if chunk is not None else None
        #            for chunk in spawn_region.chunks
        #        ]
        #    )
        # )
        # print(solid_blocks_in_spawn_chunks)


    if do_profile:
        # Run the main profiling function
        profiler = cProfile.Profile()
        profiler.enable()
        profile_test()  # your main function to profile
        profiler.disable()

        # Create a Stats object
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')  # or 'time', 'calls', etc.
        stats.dump_stats("world.prof")  # Print the profiling results

    else:
        num_tests = 1
        test_time = timeit("test()", globals=globals(), number=num_tests)
        print(f"Time taken for test(): {test_time:.6f} seconds")

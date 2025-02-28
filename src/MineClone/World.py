from __future__ import annotations

import pstats

import os.path

from MineClone.Region import *

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


@dataclass
class World(MCPhys, aabb=WORLD.AABB):
    player: Player = field(default_factory=Player)
    world_name: str = "world"
    _from_file: bool = False
    file_location: str = os.getcwd()

    def __post_init__(self):
        super().__post_init__()

        self.file_path = f"{self.file_location}\\{self.world_name}"
        del self.file_location

        self.regions = WORLD.NONE_LIST.copy()

        self._update_deque: deque[Region] = deque()

        self.region_query_cache: Dict[glm.vec2, Region] = {}

        if self._from_file:
            for region_file in os.listdir(f"{self.file_path}\\region"):
                wr_coords = glm.vec2(tuple(int(i) for i in region_file.split("r.")[1].split(".mcr")[0].split(".")))
                self.region_from_file(wr_coords)
        else:
            # Initialise Spawn Region
            self.init_region()
            print(f"Generating {WORLD.INITIAL_NUM_SPAWN_CHUNKS} Initial spawn chunks...")
            cnt = 1
            for i, w_coords in enumerate(WORLD.INITIAL_SPAWN_CHUNK_POSITIONS):
                self.generate_chunk(w_coords)
                print(f"{cnt}/{WORLD.INITIAL_NUM_SPAWN_CHUNKS} Initial Spawn Chunks Generated")
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

    def generate_chunk(self, w: glm.ivec2, rcid: Optional[glm.ivec2] = None) -> Chunk:
        if isinstance(w, glm.ivec2):
            if rcid is None:
                wrid: glm.ivec2 = w_to_wr(w)
                rcid: glm.ivec2 = w_to_rc(w)
            elif not isinstance(rcid, glm.ivec2):
                raise TypeError(f"rcid of type glm.ivec2 is required!")
        else:
            raise TypeError("w must be of type glm.ivec2")
        region = self.get_region(wrid)
        region.init_chunk(rcid)

    def init_region(self, wrid: Optional[glm.ivec2] = None):
        if wrid is None:
            wrid = self.spawn_region_index
        w_coords = wrid - WORLD.WIDTH // 2

        region_pos = WORLD.WIDTH * glm.vec3(w_coords[0], 0, w_coords[1]) + glm.vec3(0, CHUNK.EXTENTS_HALF.y, 0)
        self.set_region(wrid, Region(wrid))
        region = self.get_region(wrid)
        region.initialize(region_pos, self)

    def _is_index_out_of_bounds(self, region_index: glm.ivec2):
        if not np.prod(glm.ivec2(-1) < region_index < glm.ivec2(WORLD.NUM_REGIONS)):
            raise ValueError(f"Region Index: {region_index} is out of world bounds")

    def set_region(self, region_index: glm.ivec2, region: Region):
        self._is_index_out_of_bounds(region_index)
        self.regions[region_index[0]][region_index[1]] = region
        if not region._awaiting_update:
            self.enqueue_region(region)

    def get_region(self, region_pos: Union[glm.ivec2, glm.vec2]) -> Optional[Region]:
        # if index vec2
        if isinstance(region_pos, glm.ivec2):
            return self.regions[region_pos[0]][region_pos[1]]

        # Otherwise check cache
        cached_region = self.region_query_cache.get(region_pos)
        if cached_region is not None:  # get from cache
            return cached_region

        # If in bounds
        if self.bounds.intersect_point(region_pos, "y"):
            region = self.get_region(w_to_wr(region_pos))
            if region is not None:
                self.region_query_cache[region_pos] = region
                return region

    def get_chunk(self, chunk_pos: Union[glm.ivec2, glm.vec2]) -> Optional[Chunk]:
        region = self.get_region(w_to_wr(chunk_pos))
        if region is not None:
            return region.get_chunk(chunk_pos)

    def get_block(self, block_pos: glm.vec3) -> Optional[Block]:
        chunk = self.get_chunk(block_pos.xz)
        if chunk is not None:
            return chunk.get_block(block_pos)

    def region_to_file(self, region: Region):
        World.save_region_to_file(region, self.file_path)

    def region_from_file(self, wrid: glm.ivec2) -> Region:
        region = World.load_region_from_file(wrid, self.file_path)
        region.initialize(glm.vec3(wrid[0], CHUNK.HEIGHT_HALF_INT, wrid[1]), self)
        region.update()
        wrid += WORLD.WIDTH // 2
        #self.update_region_in_lists(region)
        self.regions[int(wrid[0])][int(wrid[1])] = region

    @property
    def spawn_region_index(self) -> glm.ivec2:
        return glm.ivec2(WORLD.WIDTH // 2, WORLD.WIDTH // 2)

    @property
    def spawn_region(self) -> Region:
        wrid = self.spawn_region_index
        return self.regions[wrid[0]][wrid[1]]

    @property
    def spawn_chunks(self) -> Tuple[Chunk]:
        return self.spawn_region.non_none_chunks

    def get_block(self, block_pos: glm.vec3) -> Optional[Block]:
        if self.bounds.intersect_point(block_pos):
            local_pos = [int(i) for i in w_to_wr(block_pos)]
            region = self.get_region(self._get_region_id(*local_pos))
            if region is None:
                return None
            block = region.get_block(block_pos)
        else:
            block = None
        if block is not None:
            pos_delta = block_pos - block.pos
            if glm.round(pos_delta) != glm.vec3(0):
                raise Exception(f"Wrong Block!:\n\t- Expected: {block_pos}\n\t- Obtained: {block.pos}")
        return block

    @staticmethod
    def create_world_directory(world_file_path: str, ):
        if not os.path.exists(f"{world_file_path}\\region"):
            if not os.path.exists(f"{world_file_path}"):
                os.mkdir(f"{world_file_path}")
            os.mkdir(f"{world_file_path}\\region")

    @staticmethod
    def save_region_to_file(region: Region, world_file_path: str = os.getcwd() + "\\world"):
        orel_wr_coords = region.index - WORLD.EXTENTS_HALF_INT  # wrid_to_wr_coords(region.index) - WORLD_REGION_WIDTH_HALF
        region_file_path = world_file_path + f"\\region\\r.{int(orel_wr_coords[0])}.{int(orel_wr_coords[1])}.mcr"
        with open(region_file_path, "wb") as f:
            f.write(region.serialize())

    @staticmethod
    def load_region_from_file(wrid: int, world_file_path: str = os.getcwd() + "\\world\\") -> Region:
        # orel_wr_coords = wrid_to_wr_coords(wrid) - WORLD.EXTENTS_HALF_INT
        #region_file_path = world_file_path + f"\\region\\r.{int(orel_wr_coords[0])}.{int(orel_wr_coords[1])}.mcr"
        region_file_path = world_file_path + f"\\region\\r.{int(wrid[0])}.{int(wrid[1])}.mcr"
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
        for region_axis in self.regions:
            for region in region_axis:
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
        # mesh = ShapeMesh(bs)

    def query(self, bounds: AABB) -> List[Region]:
        regions = []
        min_region = self.get_region(bounds.min.xz)
        if min_region is not None:
            regions.append(min_region)
        else:
            min_region = self.get_region(bounds.pos.xz)
            if min_region is not None:
                regions.append(min_region)
            else:
                max_region = self.get_region(bounds.max.xz)
                if max_region is not None:
                    regions.append(max_region)
                else:
                    raise RuntimeError("No regions found for bounds: " + str(bounds))
            return regions

        max_region = self.get_region(bounds.max.xz)
        if max_region is not min_region:
            for x in range(min_region.index.x + 1, max_region.index.x + 1):
                for z in range(min_region.index.z + 1, max_region.index.z+1):
                    regions.append(self.regions[x][z])
        return regions

    def query(self, bounds: AABB) -> List[Region]:
        if not self.bounds.contains(bounds):
            return []

        min_point = glm.floor(bounds.min)
        max_point = glm.floor(bounds.max)

        if min_point == max_point:
            return [self.get_region(min_point.xz)]

        min_region = self.get_region(min_point.xz)
        max_region = self.get_region(max_point.xz)

        if max_region is not min_region:
            regions = []
            for z in range(min_region.index[1], max_region.index[1] + 1):
                for x in range(min_region.index[0], max_region.index[0] + 1):
                    regions.append(self.regions[z][x])
            return regions
        else:
            return [min_region]


@dataclass
class ChunkRange:
    world: World
    size: int
    origin: glm.ivec2 = glm.ivec2()


    def __post_init__(self):
        self.chunks: Optional[List[List[Chunk]]] = None
        self.setup_chunk_grid()

    def setup_chunk_grid(self):
        self.chunks = []
        half_size = self.size // 2
        for z in range(-half_size, half_size + 1):
            self.chunks.append([])
            for x in range(-half_size, half_size + 1):
                offset = glm.ivec2(x, z)
                array_pos = offset + half_size
                pos = (self.origin + offset) * CHUNK.WIDTH

                region_pos = w_to_wr(pos)
                chunk_pos = w_to_rc(pos)
                self.chunks[array_pos[1]][array_pos[0]] = self.world.get_region(region_pos).get_chunk(chunk_pos)


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
        import cProfile
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

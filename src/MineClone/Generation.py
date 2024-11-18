from __future__ import annotations

from dataclasses import dataclass
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

from pyfastnoiselite.pyfastnoiselite import (
    FastNoiseLite, NoiseType, FractalType
)

from Block import *
from Biome import *

if TYPE_CHECKING:
    from Chunk import Chunk


def get_temperature_level(temperature: float) -> int:
    if temperature <= -0.45:
        return 0
    elif temperature <= -0.15:
        return 1
    elif temperature <= 0.2:
        return 2
    elif temperature <= 0.55:
        return 3
    else:
        return 4


def get_humidity_level(humidty: float) -> int:
    if humidty <= -0.35:
        return 0
    elif humidty <= -0.1:
        return 1
    elif humidty <= 0.1:
        return 2
    elif humidty <= 0.3:
        return 3
    else:
        return 4


class Continent(Renum):
    MushroomFields = 0
    DeepOcean = 1
    Ocean = 2
    Coast = 3
    NearInland = 4
    MidInland = 5
    FarInland = 6


def get_continentalness_level(continentalness: float) -> Continent:
    if continentalness <= -1.05:
        return Continent.MushroomFields
    elif continentalness <= -0.455:
        return Continent.DeepOcean
    elif continentalness <= -0.19:
        return Continent.Ocean
    elif continentalness <= -0.11:
        return Continent.Coast
    elif continentalness <= 0.03:
        return Continent.NearInland
    elif continentalness <= 0.3:
        return Continent.MidInland
    else:
        return Continent.FarInland


calculate_continent_base_height = UnivariateSpline(
    [-1, -0.455, -0.19, -0.11, 0.03, 0.3, 1.0],
    [62, 30, 50, 62, 64, 70, 100]
)


def get_erosion_level(erosion: float) -> int:
    if erosion <= -0.78:
        return 0
    elif erosion <= -0.375:
        return 1
    elif erosion <= -0.2225:
        return 2
    elif erosion <= 0.05:
        return 3
    elif erosion <= 0.45:
        return 4
    elif erosion <= 0.55:
        return 5
    else:
        return 6


calculate_erosion_modifier = UnivariateSpline(
    [-1, -0.78, -0.375, -0.2225, 0.05, 0.45, 0.55, 1.0],
    [1, 0.98, 0.94, 0.92, 0.9, 0.87, 0.85, 0.8]
)


class PeakValley(Renum):
    Valleys = 0
    Low = 1
    Mid = 2
    High = 3
    Peaks = 4


PV = PeakValley


def calculate_peak_valley_value(weirdness_value: float):
    return 1 - abs((3 * abs(weirdness_value)) - 2)


def get_peak_valley_level(peak_valley_value: float) -> PV:
    if peak_valley_value <= -0.85:
        return PV.Valleys
    elif peak_valley_value <= -0.6:
        return PV.Low
    elif peak_valley_value <= 0.2:
        return PV.Mid
    elif peak_valley_value <= 0.7:
        return PV.High
    else:
        return PV.Peaks


calculate_peak_valley_modifier = UnivariateSpline(
    [-1, -0.85, -0.6, 0.2, 0.7, 1.0],
    [-30, -20, -10, 10, 40, 120]
)


def calculate_terrain_height(
        continentalness: float,
        erosion: float,
        peak_valley_value: float) -> float:
    return (64#calculate_continent_base_height(continentalness)
            * calculate_erosion_modifier(erosion)
            + calculate_peak_valley_modifier(peak_valley_value))

@dataclass
class BiomeParams:
    temperature: float
    humidity: float
    continentalness: float
    erosion: float
    weirdness: float
    depth: float

    def __post_init__(self):
        self.value_peak_valley: float = calculate_peak_valley_value(self.weirdness)

        self.level_temperature: int = get_temperature_level(self.temperature)
        self.level_humidty: int = get_humidity_level(self.humidity)
        self.level_erosion: int = get_erosion_level(self.erosion)

        self.continent: Continent = get_continentalness_level(self.continentalness)
        self.peak_valley: PeakValley = get_peak_valley_level(self.value_peak_valley)

def get_grid_coords(origin: Union[glm.vec2, glm.vec3],
                    extents: Union[glm.ivec2, glm.ivec3]) -> Tuple[np.ndarray, Tuple[int, ...]]:
    if not isinstance(origin, Union[glm.vec2, glm.vec3]):
        raise TypeError("origin must be a glm.vec2 or glm.vec3")

    if not isinstance(extents, Union[glm.ivec2, glm.ivec3]):
        if isinstance(extents, Union[glm.vec2, glm.vec3]):
            if isinstance(extents, glm.vec2):
                extents = glm.ivec2(extents)
            else:
                extents = glm.ivec3(extents)
        else:
            raise TypeError("extents must be a glm.ivec2 or glm.ivec3")
    if isinstance(origin, glm.vec2):
        vec = glm.vec2
    else:
        vec = glm.vec3

    # Compute the half spread for all axes
    half_extents = extents / 2

    min_point = origin - vec(half_extents)
    max_point = origin + vec(half_extents)

    # Generate coordinate ranges for each axis
    ranges = [np.linspace(min_point[i], max_point[i], extents[i]) for i in range(len(origin))]

    # Generate a meshgrid and flatten it
    grids = np.meshgrid(*ranges, indexing='ij')
    return np.stack([g.ravel() for g in grids]).astype(np.float32), tuple(extent for extent in extents)


@dataclass
class NoiseGenerator:
    _frequency: float = 0.010
    _octaves: int = 1
    _lacunarity: float = 2.0
    _gain: float = 0.5
    _weighted_strength: float = 0.0
    _scale: float = 1.0
    _seed: Optional[int] = None

    def __post_init__(self):
        if self._seed is None:
            self._seed = random.randint(0, 1_000_000)

        self._fnl: FastNoiseLite = FastNoiseLite()
        self._fnl.noise_type = NoiseType.NoiseType_Perlin
        self._fnl.fractal_type = FractalType.FractalType_FBm

        self._fnl.seed = self._seed
        self._fnl.frequency = self._scale * self._frequency

        self._fnl.fractal_octaves = self._octaves
        self._fnl.fractal_lacunarity = self._lacunarity
        self._fnl.fractal_gain = self._gain
        self._fnl.fractal_weighted_strength = self._weighted_strength

    def sample(self, x: float, y: float, z: Optional[float] = None) -> float:
        return self._fnl.get_noise(x, y, z)

    def grid_sample(self, coords: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        # Reshape the samples into the grid dimensions
        return self._fnl.gen_from_coords(coords).reshape(shape)


class ContinentalNoiseGenerator(NoiseGenerator):
    def __init__(self, seed: Optional[int] = None, _scale: float = 1.0):
        super().__init__(
            _frequency=0.0075,
            _octaves=4,
            _lacunarity=2.0,
            _gain=0.75,
            _weighted_strength=0.7,
            _scale=_scale,
            _seed=seed
        )


class TemperatureNoiseGenerator(NoiseGenerator):
    def __init__(self, seed: Optional[int] = None, _scale: float = 1.0):
        super().__init__(
            _frequency=0.003,  # Lower frequency for smoother transitions
            _octaves=2,  # Fewer octaves for broad patterns
            _lacunarity=1.75,
            _gain=0.13,
            _weighted_strength=0.0,
            _scale=_scale,
            _seed=seed
        )


class HumidityNoiseGenerator(NoiseGenerator):
    def __init__(self, seed: Optional[int] = None, _scale: float = 1.0):
        super().__init__(
            _frequency=0.005,  # Higher frequency than temperature for more localized variability
            _octaves=3,  # More octaves for finer detail
            _lacunarity=2.82,
            _gain=0.18,
            _weighted_strength=0,
            _scale=_scale,
            _seed=seed
        )


class ErosionNoiseGenerator(NoiseGenerator):
    def __init__(self, seed: Optional[int] = None, _scale: float = 1.0):
        super().__init__(
            _frequency=0.0075,  # Higher frequency for rugged detail
            _octaves=5,  # High octaves for more jagged features
            _lacunarity=3.39,  # Slightly increased lacunarity for variability in _scale
            _gain=0.4,  # Lower gain to create a more eroded, rugged look
            _weighted_strength=0.72,
            _scale=_scale,
            _seed=seed
        )


class WeirdnessNoiseGenerator(NoiseGenerator):
    def __init__(self, seed: Optional[int] = None, _scale: float = 1.0):
        super().__init__(
            _frequency=0.010,  # Higher frequency for rapid changes
            _octaves=3,  # More octaves to add layers of detail
            _lacunarity=3,  # High lacunarity for fractal-like patterns
            _gain=0.16,  # Moderate gain to keep some structure
            _weighted_strength=-0.94,  # Moderate weighted strength for variance
            _scale=_scale,
            _seed=seed
        )


class DensityNoiseGenerator(NoiseGenerator):
    def __init__(self, seed: Optional[int] = None, _scale: float = 1.0):
        super().__init__(
            _frequency=0.01,  # Higher frequency for rapid changes
            _octaves=4,  # More octaves to add layers of detail
            _lacunarity=2.0,  # High lacunarity for fractal-like patterns
            _gain=0.5,  # Moderate gain to keep some structure
            _weighted_strength=-0.2,  # Moderate weighted strength for variance
            _scale=_scale,
            _seed=seed
        )

    def sample(self, x: float, y: float, z: Optional[float] = None) -> float:
        if z is None:
            raise ValueError("Density is a 3D noise generator!")
        return super().sample(x, y, z)


def calculate_density(density_noise: float, y: float, base_height: float = 0, height_bias: float = 1) -> float:
    if y == base_height:
        return density_noise
    if y > base_height:
        adjustment: float = -(height_bias * (y - base_height) / base_height)
    else:
        adjustment: float = height_bias * (base_height - y) / base_height
    return density_noise + adjustment


# Alias for convenience
NoiseGen = NoiseGenerator
ConNoiseGen = ContinentalNoiseGenerator
TempNoiseGen = TemperatureNoiseGenerator
HumidNoiseGen = HumidityNoiseGenerator
EroNoiseGen = ErosionNoiseGenerator
WeirdNoiseGen = WeirdnessNoiseGenerator
DenseNoiseGen = DensityNoiseGenerator

# Step 1: Define the biomes and categories
biome_categories = {
    "Null": (-1, "black"),  # Unassigned or default
    # Cave Biomes
    "CavesDripstone": (0, "brown"),
    "CavesLush": (1, "limegreen"),
    "DeepDark": (2, "darkgray"),
    # Non-Inland
    "MushroomFields": (3, "purple"),
    "OceanDeepFrozen": (4, "lightblue"),
    "OceanDeepCold": (5, "blue"),
    "OceanDeepLukewarm": (6, "teal"),
    "OceanDeep": (7, "darkblue"),
    "OceanFrozen": (8, "cyan"),
    "OceanCold": (9, "dodgerblue"),
    "Ocean": (10, "navy"),
    "OceanLukewarm": (11, "mediumaquamarine"),
    "OceanWarm": (12, "gold"),
    # Inland Surface
    "River": (13, "steelblue"),
    "RiverFrozen": (14, "skyblue"),
    "StonyShore": (15, "gray"),
    "Beach": (16, "khaki"),
    "BeachSnowy": (17, "whitesmoke"),
    "Desert": (18, "yellow"),
    "Savanna": (19, "lightgoldenrodyellow"),
    "SavannaWindsept": (20, "goldenrod"),
    "SavannaPlateau": (21, "darkkhaki"),
    "Swamp": (22, "darkolivegreen"),
    "SwampMangrove": (23, "forestgreen"),
    "SnowySlope": (24, "white"),
    "Grove": (25, "lightgreen"),
    "IceSpikes": (26, "aliceblue"),
    "Badlands": (27, "sienna"),
    "BadlandsEroded": (28, "peru"),
    "BadlandsWooded": (29, "rosybrown"),
    "Plains": (30, "wheat"),
    "PlainsSnowy": (31, "mintcream"),
    "PlainsSunflower": (32, "gold"),
    "Forest": (33, "green"),
    "ForestFlower": (34, "lightpink"),
    "ForestBirch": (35, "beige"),
    "ForestOldGrowthBirch": (36, "darkseagreen"),
    "ForestDark": (37, "darkgreen"),
    "ForestWindswept": (38, "seagreen"),
    "Taiga": (39, "mediumspringgreen"),
    "TaigaSnowy": (40, "snow"),
    "TaigaOldGrowthSpruce": (41, "darkslategray"),
    "TaigaOldGrowthPine": (42, "olivedrab"),
    "Jungle": (43, "chartreuse"),
    "JungleSparse": (44, "yellowgreen"),
    "JungleBamboo": (45, "springgreen"),
    "Meadow": (46, "greenyellow"),
    "CherryGrove": (47, "pink"),
    "PaleGarden": (48, "lavender"),
    "PeaksJagged": (49, "slategray"),
    "PeaksFrozen": (50, "lightskyblue"),
    "PeaksStony": (51, "dimgrey"),
    "HillsWindswept": (52, "lightsteelblue"),
    "HillsWindsweptGravelly": (53, "silver"),
}

# Step 2: Create colormap and boundaries
biome_ids = [v[0] for v in biome_categories.values()]
colors = [v[1] for v in biome_categories.values()]
boundaries = list(range(len(biome_ids) + 1))  # Create discrete boundaries
biome_cmap = ListedColormap(colors)
biome_norm = BoundaryNorm(boundaries, biome_cmap.N)


def plot_noise_grid(noise_values: np.ndarray, cmap: str = "gray", title: str = "Perlin Noise Grid",
                    normalized: bool = False) -> None:
    """
    Visualizes a 2D grid of Perlin noise values as an image using matplotlib.

    :param noise_values: A 2D numpy array of noise values.
    :param cmap: The color map to use for visualization (default is 'gray').
    :param title: The title of the plot (default is "Perlin Noise Grid").
    :param normalized: Whether to normalize the color range to [-1, 1] (default is False).
    """
    # Dynamically set figure size to match the screen
    fig = plt.figure()
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.state("zoomed")  # Switch to fullscreen mode

    # Apply normalization if requested
    if normalized:
        norm = Normalize(vmin=-1, vmax=1)
        plt.imshow(noise_values, cmap=cmap, norm=norm)
    elif title == "Biome":  # Assume `biome_cmap` and `biome_norm` are predefined
        plt.imshow(noise_values, cmap=biome_cmap, norm=biome_norm)
    else:
        plt.imshow(noise_values, cmap=cmap)

    # Add a colorbar and title
    plt.colorbar()
    plt.title(title)

    # Automatically adjust the layout for better fit
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_cross_section(array_3d, slice_type="xy", slice_index=0):
    """
    Plots a cross-section of the 3d array.

    Parameters:
    - d_values: 3D array of density values.
    - slice_type: A string ("xy", "xz", or "yz") specifying the cross-sectional plane.
    - slice_index: The index along the third axis (z-axis for xy, x-axis for yz, or y-axis for xz).
    """
    if slice_type == "xy":
        # Slice along the z-axis, use slice_index to pick the slice
        data = array_3d[:, slice_index, :]
        xlabel, ylabel = "X-axis", "Y-axis"
        title = f"Cross-section at Z = {slice_index}"

    elif slice_type == "xz":
        # Slice along the y-axis, use slice_index to pick the slice
        data = array_3d[slice_index, :, :]
        xlabel, ylabel = "X-axis", "Z-axis"
        title = f"Cross-section at Y = {slice_index}"

    elif slice_type == "yz":
        # Slice along the x-axis, use slice_index to pick the slice
        data = array_3d[slice_index, :, :]
        xlabel, ylabel = "Y-axis", "Z-axis"
        title = f"Cross-section at X = {slice_index}"

    else:
        raise ValueError("Invalid slice_type. Choose from 'xy', 'xz', or 'yz'.")

    # Plotting the 2D slice
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='gray_r', aspect='auto', origin='lower')
    plt.colorbar(label="Density")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_3d_isometric(array_3d):
    """
    Plots the whole 3D array in an isometric view with colors based on the distance
    of each point from the origin or based on the x, y, z positions.

    Parameters:
    - array_3d: 3D numpy array of data (density values).
    """
    # Create a meshgrid for the coordinates
    y, z, x = array_3d.nonzero()  # Get the coordinates of non-zero points
    values = array_3d[y, z, x]  # Extract the corresponding density values

    # Calculate a gradient for coloring based on x, y, z positions
    # This can be a simple Euclidean distance from the origin (0, 0, 0)
    gradient = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points as a scatter plot with coloring based on gradient
    sc = ax.scatter(x, z, y, c=gradient, cmap='viridis', marker='o', alpha=0.6)

    # Set the labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Z-axis")
    ax.set_zlabel("Y-axis")
    ax.set_title("3D Isometric View of Array with Gradient Coloring")

    # Set the aspect ratio to be equal to ensure the isometric view
    ax.view_init(azim=45, elev=30)  # Adjust view angle to give isometric perspective

    # Add a color bar to show the gradient scale
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)

    plt.show()


class TerrainGenerator:
    def __init__(self, seed: int):
        self.seed: int = seed
        self.c_ngen = ContinentalNoiseGenerator(self.seed)
        self.t_ngen = TempNoiseGen(self.seed + 1)
        self.h_ngen = HumidNoiseGen(self.seed + 2)
        self.e_ngen = EroNoiseGen(self.seed + 3)
        self.w_ngen = WeirdNoiseGen(self.seed + 4)
        self.d_ngen = DenseNoiseGen(self.seed + 5)

        self.chunk_size = CHUNK.EXTENTS.yzx

    def gen_chunk(self, chunk: Chunk):
        grid_coords_2d, grid_shape_2d = get_grid_coords(chunk.pos.yz, self.chunk_size.yz)
        grid_coords_3d, grid_shape_3d = get_grid_coords(chunk.pos.yzx, self.chunk_size)
        chunk.pos -= CHUNK.EXTENTS_HALF

        c_noise_values: np.ndarray = self.c_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
        t_noise_values: np.ndarray = self.t_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
        h_noise_values: np.ndarray = self.h_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
        e_noise_values: np.ndarray = self.e_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
        w_noise_values: np.ndarray = self.w_ngen.grid_sample(grid_coords_2d, grid_shape_2d)

        pv_values: np.ndarray = np.zeros(grid_shape_2d)
        th_values: np.ndarray = np.zeros(grid_shape_2d)

        d_noise_values = self.d_ngen.grid_sample(grid_coords_3d, grid_shape_3d)
        d_values = np.zeros(grid_shape_3d)

        c_levels = np.empty(grid_shape_2d)
        t_levels = np.empty(grid_shape_2d)
        h_levels = np.empty(grid_shape_2d)
        e_levels = np.empty(grid_shape_2d)
        pv_levels = np.empty(grid_shape_2d)

        tallest_height = 0
        lowest_height = CHUNK.HEIGHT

        index = 0
        for x in CHUNK.WIDTH_RANGE:
            for z in CHUNK.WIDTH_RANGE:
                continentalness: float = c_noise_values[z][x]
                temperature: float = t_noise_values[z][x]
                humidity: float = h_noise_values[z][x]
                erosion: float = e_noise_values[z][x]
                weirdness: float = w_noise_values[z][x]
                peak_valley_value = pv_values[z][x] = calculate_peak_valley_value(weirdness)
                base_height = th_values[z][x] = calculate_terrain_height(
                    continentalness, erosion, peak_valley_value
                )
                # new_limits = False
                # if base_height > tallest_height:
                #     tallest_height = base_height
                #     new_limits = True
                # if tallest_height >= CHUNK.HEIGHT:
                #     raise RuntimeError(f"Base Terrain Height ({base_height}) Too High!")
                #
                # if base_height < lowest_height:
                #     lowest_height = base_height
                #     new_limits = True
                # if lowest_height < 0:
                #     raise RuntimeError(f"Base Terrain Height ({base_height}) Too Low!")

                # continentalness_level = c_levels[z][x] = get_continentalness_level(continentalness)
                # temperature_level = t_levels[z][x] = get_temperature_level(temperature)
                # humidty_level = h_levels[z][x] = get_humidity_level(humidity)
                # erosion_level = e_levels[z][x] = get_erosion_level(erosion)
                # peak_valley_level = pv_levels[z][x] = get_peak_valley_level(peak_valley_value)

                # biome_params = BiomeParams(
                #     temperature,
                #     humidity,
                #     continentalness,
                #     erosion,
                #     weirdness,
                #     1.0
                # )
                # biome_values[z][x] = get_biome_id(biome_params)

                for y in CHUNK.HEIGHT_RANGE:
                    density = d_noise_values[y][z][x]

                    d_values[y][z][x] = density_adjusted = calculate_density(
                        density,
                        y,
                        base_height
                    )
                    if density_adjusted > 0:  # Solid Block
                        chunk.set_block(x, y, z, BlockID.Stone)
                    else:
                        chunk.set_block(x, y, z, BlockID.Air)

                    block = chunk.blocks[y][z][x]
                    chunk.block_face_ids[index] = block.face_ids
                    chunk.block_face_tex_ids[index] = block.face_tex_ids
                    chunk.block_face_tex_size_ids[index] = block.face_tex_sizes
                    block.pos += glm.vec3(x, y, z)
                    block.initialize(chunk)
                    chunk.block_instances[index] = np.array(NMM(block.pos, s=glm.vec3(0.5)).to_list())
                    index += 1
        chunk.pos += CHUNK.EXTENTS_HALF

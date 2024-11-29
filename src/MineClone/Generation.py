from __future__ import annotations
from scipy.interpolate import UnivariateSpline

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

from pyfastnoiselite.pyfastnoiselite import (
    FastNoiseLite, NoiseType, FractalType
)

from MineClone.Block import *
from MineClone.Biome import *

if TYPE_CHECKING:
    from Chunk import Chunk


calculate_continent_base_height = UnivariateSpline(
    [-1, -0.455, -0.19, -0.11, 0.03, 0.3, 1.0],
    [62, 30, 50, 62, 64, 70, 100]
)


calculate_erosion_modifier = UnivariateSpline(
    [-1, -0.78, -0.375, -0.2225, 0.05, 0.45, 0.55, 1.0],
    [1, 0.98, 0.94, 0.92, 0.9, 0.87, 0.85, 0.8]
)


calculate_peak_valley_modifier = UnivariateSpline(
    [-1, -0.85, -0.6, 0.2, 0.7, 1.0],
    [-30, -20, -10, 10, 40, 120]
)


def calculate_terrain_height(
        continentalness: float,
        erosion: float,
        peak_valley_value: float) -> float:
    return (calculate_continent_base_height(continentalness)
            * calculate_erosion_modifier(erosion)
            + calculate_peak_valley_modifier(peak_valley_value))




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


def plot_3d_isometric(array_3ds: Union[np.ndarray, List[np.ndarray]]):
    """
    Plots one or more 3D arrays in an isometric view with colors based on the distance
    of each point from the origin or based on the x, y, z positions.

    Parameters:
    - array_3ds: A single 3D numpy array or a list of 3D numpy arrays of data (density values).
    """
    # If a single array is provided, convert it to a list for uniform processing
    if isinstance(array_3ds, np.ndarray):
        array_3ds = [array_3ds]
        # If a single array is provided, convert it to a list for uniform processing
        if isinstance(array_3ds, np.ndarray):
            array_3ds = [array_3ds]

        # Set up the figure and axis grid based on the number of arrays
        num_arrays = len(array_3ds)
        cols = int(np.ceil(np.sqrt(num_arrays)))
        rows = int(np.ceil(num_arrays / cols))
        fig, axs = plt.subplots(rows, cols, subplot_kw={'projection': '3d'}, figsize=(6 * cols, 6 * rows))

        # Flatten axs to simplify indexing
        if num_arrays == 1:
            axs = [axs]  # Ensure axs is a list
        else:
            axs = axs.flatten()

        for idx, (array_3d, ax) in enumerate(zip(array_3ds, axs)):
            # Get non-zero points and their values
            y, z, x = array_3d.nonzero()
            values = array_3d[y, z, x]

            # Calculate a gradient for coloring based on x, y, z positions
            gradient = np.sqrt(x ** 2 + y ** 2 + z ** 2)

            # Plot the points as a scatter plot with coloring based on gradient
            sc = ax.scatter(x, z, y, c=gradient, cmap='viridis', marker='o', alpha=0.6)

            # Set the labels and title
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Z-axis")
            ax.set_zlabel("Y-axis")
            ax.set_title(f"Array {idx + 1}")

            # Adjust view angle to give isometric perspective
            ax.view_init(azim=45, elev=30)

        # Hide unused subplots
        for ax in axs[num_arrays:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()


def plot_3d_isometric(
        array_3ds: Union[np.ndarray, List[np.ndarray]],
        show_separately: bool = False
):
    """
    Plots one or more 3D arrays in an isometric view with colors based on the distance
    of each point from the origin or based on the x, y, z positions.

    Parameters:
    - array_3ds: A single 3D numpy array or a list of 3D numpy arrays of data (density values).
    - show_separately: If True, each array is plotted in a separate figure. Otherwise, they are
      shown as subplots in a single figure.
    """
    # If a single array is provided, convert it to a list for uniform processing
    if isinstance(array_3ds, np.ndarray):
        array_3ds = [array_3ds]

    if show_separately:
        # Create a separate figure for each array
        for idx, array_3d in enumerate(array_3ds):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            _plot_single_3d_array(array_3d, ax, title=f"Array {idx + 1}")
            plt.show()
    else:
        # Set up a single figure with subplots for all arrays
        num_arrays = len(array_3ds)
        cols = int(np.ceil(np.sqrt(num_arrays)))
        rows = int(np.ceil(num_arrays / cols))
        fig, axs = plt.subplots(rows, cols, subplot_kw={'projection': '3d'}, figsize=(6 * cols, 6 * rows))

        # Flatten axs to simplify indexing
        if num_arrays == 1:
            axs = [axs]  # Ensure axs is a list
        else:
            axs = axs.flatten()

        for idx, (array_3d, ax) in enumerate(zip(array_3ds, axs)):
            _plot_single_3d_array(array_3d, ax, title=f"Array {idx + 1}")

        # Hide unused subplots
        for ax in axs[num_arrays:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()


def _plot_single_3d_array(array_3d, ax, title="3D Isometric View"):
    """
    Helper function to plot a single 3D array on the provided axis.
    """
    # Get non-zero points and their values
    y, z, x = array_3d.nonzero()
    values = array_3d[y, z, x]

    # Calculate a gradient for coloring based on x, y, z positions
    gradient = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Plot the points as a scatter plot with coloring based on gradient
    sc = ax.scatter(x, z, y, c=gradient, cmap='viridis', marker='o', alpha=0.6)

    # Set the labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Z-axis")
    ax.set_zlabel("Y-axis")
    ax.set_title(title)

    # Adjust view angle to give isometric perspective
    ax.view_init(azim=-90, elev=30)

    max_y, max_z, max_x = array_3d.shape
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_z)
    ax.set_zlim(0, max_y)


def get_biome_block_id(biome: Biome, y: int, surface_y: int) -> BlockID:
    # TODO: Bedrock level
    if surface_y - 4 < y < surface_y:
        return BlockID.Dirt
    elif y == surface_y:
        return BlockID.Grass


    return BlockID.Stone


class TerrainGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.seed: int = seed if seed is not None else random.randint(0, 2 ** 16 - 1)
        self.c_ngen = ContinentalNoiseGenerator(self.seed)
        self.t_ngen = TempNoiseGen(self.seed + 1)
        self.h_ngen = HumidNoiseGen(self.seed + 2)
        self.e_ngen = EroNoiseGen(self.seed + 3)
        self.w_ngen = WeirdNoiseGen(self.seed + 4)
        self.d_ngen = DenseNoiseGen(self.seed + 5)

        self.chunk_size = CHUNK.EXTENTS.yzx
        self.chunk_size_zx = CHUNK.EXTENTS.zx

    def gen_chunk(self, chunk: Optional[Chunk] = None, chunk_pos: Optional[glm.vec3] = None) -> Optional[np.ndarray]:
        if chunk is not None:
            chunk_pos = chunk.pos
        elif chunk_pos is None:
            raise ValueError("Either 'chunk' or 'chunk_pos' must be provided.")
        grid_coords_2d, grid_shape_2d = get_grid_coords(chunk_pos.zx, self.chunk_size_zx)
        grid_coords_3d, grid_shape_3d = get_grid_coords(chunk_pos.yzx, self.chunk_size)
        # print(f"z: {grid_coords_2d[0][0]}, x:{grid_coords_2d[1][0]}")
        # print(f"z: {grid_coords_2d[0][-1]}, x:{grid_coords_2d[1][-1]}")
        if chunk_pos is not None:
            biome_values: np.ndarray = np.zeros(grid_shape_2d)
            solid_values: np.ndarray = np.zeros(grid_shape_3d)
        chunk_pos -= CHUNK.EXTENTS_HALF

        c_noise_values: np.ndarray = self.c_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
        t_noise_values: np.ndarray = self.t_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
        h_noise_values: np.ndarray = self.h_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
        e_noise_values: np.ndarray = self.e_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
        w_noise_values: np.ndarray = self.w_ngen.grid_sample(grid_coords_2d, grid_shape_2d)

        pv_values: np.ndarray = np.zeros(grid_shape_2d)
        th_values: np.ndarray = np.zeros(grid_shape_2d)

        d_noise_values = self.d_ngen.grid_sample(grid_coords_3d, grid_shape_3d)
        d_values = np.zeros(grid_shape_3d)

        # c_levels = np.empty(grid_shape_2d)
        # t_levels = np.empty(grid_shape_2d)
        # h_levels = np.empty(grid_shape_2d)
        # e_levels = np.empty(grid_shape_2d)
        # pv_levels = np.empty(grid_shape_2d)
        #
        # tallest_height = 0
        # lowest_height = CHUNK.HEIGHT

        index = 0
        biome_talley: Dict[BiomeID, int] = {}
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

                surface_biome = Biome(
                    BiomeParams(
                        temperature,
                        humidity,
                        continentalness,
                        erosion,
                        weirdness,
                        0
                    )
                )
                if biome_talley.get(surface_biome.id) is None:
                    biome_talley[surface_biome.id] = 1
                else:
                    biome_talley[surface_biome.id] += 1
                surface_height = None
                for y in CHUNK.HEIGHT_RANGE_REVERSE:
                    block_depth = base_height - y
                    depth = min(0, block_depth) * (1/128) # increases by this per block below surface
                    if depth not in [0.0, 1.0]:
                        biome = Biome(
                            BiomeParams(
                                temperature,
                                humidity,
                                continentalness,
                                erosion,
                                weirdness,
                                depth
                            )
                        )
                    else:
                        biome = surface_biome

                    init_density = d_noise_values[y][z][x]

                    d_values[y][z][x] = density = calculate_density(
                        init_density,
                        y,
                        base_height
                    )
                    if chunk is not None:
                        chunk.biomes[y][z][x] = biome
                        chunk.blocks[y][z][x] = Block(glm.ivec3(y, z, x))
                        block = chunk.blocks[y][z][x]

                        if density > 0:  # Solid Block
                            if surface_height is None:
                                surface_height = y
                            block_id = get_biome_block_id(biome, y, surface_height)
                            chunk.set_block(x, y, z, block_id)
                        face_index = index * 6
                        chunk.block_face_instance_data[face_index: face_index+6] = block.face_instance_data
                        block.pos += glm.vec3(x, y, z)
                        block.initialize(chunk)

                        if x > 0:
                            block.neighbours[Side.West] = chunk.blocks[y][z][x - 1]
                            chunk.blocks[y][z][x - 1].neighbours[Side.East] = block

                        if y < CHUNK.HEIGHT - 1:
                            block.neighbours[Side.Top] = chunk.blocks[y + 1][z][x]
                            chunk.blocks[y + 1][z][x].neighbours[Side.Bottom] = block

                        if z > 0:
                            block.neighbours[Side.North] = chunk.blocks[y][z - 1][x]
                            chunk.blocks[y][z - 1][x].neighbours[Side.South] = block

                        chunk.block_query_cache[glm.vec3(x, y, z)] = block

                        chunk.block_instance_data[index] = block.instance_data
                        index += 1
                    else:
                        biome_values[y][z][x] = biome.id
                        if density > 0:
                            solid_values[y][z][x] = True

        chunk_pos += CHUNK.EXTENTS_HALF
        print(f"\nBiome distribution for position {chunk_pos}: {biome_talley}")
        if chunk is None:
            return solid_values


if __name__ == "__main__":
    t_gen = TerrainGenerator(42)
    a = [t_gen.gen_chunk(chunk_pos=glm.vec3(x, CHUNK.HEIGHT / 2, 0)) for x in [-16, 0, 16]]
    #
    # [plot_cross_section(i) for i in a]
    plot_3d_isometric(a)

import random
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Union, Optional, Dict, Tuple
from pyfastnoiselite.pyfastnoiselite import (
    FastNoiseLite, NoiseType, FractalType
)

from MineClone.Biome import get_temperature_level, get_erosion_level

_grids_cache: Dict[Tuple, Tuple[np.ndarray, Tuple]] = {}

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

    def sample_grid(self, x: float, x_size: int, y: float, y_size: int, z: Optional[float] = None,
                    z_size: Optional[int] = None):

        cache_key = (x, x_size, y, y_size, z, z_size)
        cache_value = _grids_cache.get(cache_key, None)
        if cache_value is None:
            # Initialize base coordinates and sizes
            base_coords = [x, y] if z is None else [x, y, z]
            sizes = [x_size, y_size] if z is None else [x_size, y_size, z_size]

            # Compute the half spread for all axes
            spread = 1.0
            half_spread = spread / 2

            # Generate coordinate ranges for each axis
            ranges = [np.linspace(coord - half_spread, coord + half_spread, size) for coord, size in
                      zip(base_coords, sizes)]

            # Generate a meshgrid and flatten it
            grids = np.meshgrid(*ranges, indexing='ij')
            flattened_coords = np.stack([g.ravel() for g in grids]).astype(np.float32)

            _grids_cache[cache_key] = flattened_coords, sizes
        else:
            flattened_coords, sizes = cache_value
        # Generate samples from the coordinates
        samples = self._fnl.gen_from_coords(flattened_coords)

        # Reshape the samples into the grid dimensions
        return samples.reshape(sizes)

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

from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm

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

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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
    Plots the whole 3D array in an isometric view using scatter or surface plot.

    Parameters:
    - array_3d: 3D numpy array of data (density values).
    """
    # Create a meshgrid for the coordinates
    z, y, x = array_3d.nonzero()  # Get the coordinates of non-zero points
    values = array_3d[z, y, x]  # Extract the corresponding density values
    gradient = np.sqrt(x**2 + y**2 + z**2)


    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(x, y, z, projection='3d')

    # Plot the points as a scatter plot (you can also use other plot types like surface or wireframe)
    ax.scatter(x, y, z, c=gradient, cmap='viridis', marker='o', alpha=0.1)

    # Set the labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Isometric View of Array")

    # Set the aspect ratio to be equal to ensure the isometric view
    ax.view_init(azim=45, elev=30)  # Adjust view angle to give isometric perspective

    plt.show()

def plot_3d_isometric(array_3d):
    """
    Plots the whole 3D array in an isometric view with colors based on the distance
    of each point from the origin or based on the x, y, z positions.

    Parameters:
    - array_3d: 3D numpy array of data (density values).
    """
    # Create a meshgrid for the coordinates
    z, y, x = array_3d.nonzero()  # Get the coordinates of non-zero points
    values = array_3d[z, y, x]  # Extract the corresponding density values

    # Calculate a gradient for coloring based on x, y, z positions
    # This can be a simple Euclidean distance from the origin (0, 0, 0)
    gradient = np.sqrt(x**2 + y**2 + z**2)

    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points as a scatter plot with coloring based on gradient
    sc = ax.scatter(x, y, z, c=gradient, cmap='viridis', marker='o', alpha=0.6)

    # Set the labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Isometric View of Array with Gradient Coloring")

    # Set the aspect ratio to be equal to ensure the isometric view
    ax.view_init(azim=45, elev=30)  # Adjust view angle to give isometric perspective

    # Add a color bar to show the gradient scale
    fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)

    plt.show()

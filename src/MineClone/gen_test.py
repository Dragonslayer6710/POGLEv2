import random

import numpy as np
from Biome import *

size_2d = glm.vec2(256)
size_2d_int = glm.ivec2(size_2d)

size_3d = glm.vec3(384, size_2d)
size_3d_int = glm.ivec3(size_3d)

known_seed = True
seed = 42 if known_seed else random.randint(0, 1_000_000)




c_levels = np.empty(size_2d_int)
t_levels = np.empty(size_2d_int)
h_levels = np.empty(size_2d_int)
e_levels = np.empty(size_2d_int)
pv_levels = np.empty(size_2d_int)

th_values = np.empty(size_2d_int)

d_values = np.empty(size_3d_int)
is_solid_values = np.empty(size_3d_int)

biome_values = np.empty(size_2d_int)

def one_at_a_time():
    c_noise_values = np.empty(size_2d_int)
    t_noise_values = np.empty(size_2d_int)
    h_noise_values = np.empty(size_2d_int)
    e_noise_values = np.empty(size_2d_int)
    w_noise_values = np.empty(size_2d_int)
    d_noise_values = np.empty(size_3d_int)

    tallest_height = 0
    lowest_height = size_3d_int[0]
    for x in range(size_3d_int[1]):
        for z in range(size_3d_int[1]):
            continentalness = c_noise_values[z][x] = c_ngen.sample(x, z)
            temperature = t_noise_values[z][x] = t_ngen.sample(x, z)
            humidity = h_noise_values[z][x] = h_ngen.sample(x, z)
            erosion = e_noise_values[z][x] = e_ngen.sample(x, z)
            weirdness = w_noise_values[z][x] = w_ngen.sample(x, z)
            peak_valley_value = pv_noise_values[z][x] = calculate_peak_valley_value(weirdness)
            base_height = th_values[z][x] = calculate_terrain_height(
                continentalness, erosion, peak_valley_value
            )

            new_limits = False
            if base_height > tallest_height:
                tallest_height = base_height
                new_limits = True
            if tallest_height >= size_3d_int[0]:
                raise RuntimeError(f"Base Terrain Height ({base_height}) Too High!")

            if base_height < lowest_height:
                lowest_height = base_height
                new_limits = True
            if lowest_height < 0:
                raise RuntimeError(f"Base Terrain Height ({base_height}) Too Low!")

            c_level_values[z][x] = get_continentalness_level(continentalness)
            t_level_values[z][x] = get_temperature_level(temperature)
            h_level_values[z][x] = get_humidity_level(humidity)
            e_level_values[z][x] = get_erosion_level(erosion)
            pv_level_values[z][x] = get_peak_valley_level(peak_valley_value)

            biome_params = BiomeParams(
                temperature,
                humidity,
                continentalness,
                erosion,
                weirdness,
                1.0
            )
            biome_values[z][x] = get_biome_id(biome_params)

            for y in range(size_3d_int[0]):
                d_noise_values[y][z][x] = d_ngen.sample(x, y, z)

                d_values[y][z][x] = density = calculate_density(
                    d_noise_values[y][z][x],
                    y,
                    base_height
                )
                is_solid_values[y][z][x] = density > 0


def all_at_once(offset: glm.vec3 = glm.vec3(0)):
    global biome_values
    if offset.y != 0:
        raise RuntimeError("Offset Y must be 0!")
    origin_2d = glm.vec2(size_3d_int[1] / 2) + offset.xz
    extents_2d = glm.ivec2(size_3d_int[1])
    grid_coords_2d, grid_shape_2d = get_grid_coords(origin_2d, extents_2d)

    origin_3d = glm.vec3(size_3d_int[0] / 2, origin_2d) + offset
    extents_3d = glm.ivec3(size_3d_int[0], extents_2d)
    grid_coords_3d, grid_shape_3d = get_grid_coords(origin_3d, extents_3d)

    c_noise_values: np.ndarray = c_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
    t_noise_values: np.ndarray = t_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
    h_noise_values: np.ndarray = h_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
    e_noise_values: np.ndarray = e_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
    w_noise_values: np.ndarray = w_ngen.grid_sample(grid_coords_2d, grid_shape_2d)

    d_noise_values = d_ngen.grid_sample(grid_coords_3d, grid_shape_3d)

    tallest_height = 0
    lowest_height = size_3d_int[0]
    for x in range(size_3d_int[1]):
        for z in range(size_3d_int[1]):
            continentalness: float = c_noise_values[z][x]
            temperature: float = t_noise_values[z][x]
            humidity: float = h_noise_values[z][x]
            erosion: float = e_noise_values[z][x]
            weirdness: float = w_noise_values[z][x]
            peak_valley_value = pv_noise_values[z][x] = calculate_peak_valley_value(weirdness)
            base_height = th_values[z][x] = calculate_terrain_height(
                continentalness, erosion, peak_valley_value
            )
            new_limits = False
            if base_height > tallest_height:
                tallest_height = base_height
                new_limits = True
            if tallest_height >= size_3d_int[0]:
                raise RuntimeError(f"Base Terrain Height ({base_height}) Too High!")

            if base_height < lowest_height:
                lowest_height = base_height
                new_limits = True
            if lowest_height < 0:
                raise RuntimeError(f"Base Terrain Height ({base_height}) Too Low!")

            c_level_values[z][x] = get_continentalness_level(continentalness)
            t_level_values[z][x] = get_temperature_level(temperature)
            h_level_values[z][x] = get_humidity_level(humidity)
            e_level_values[z][x] = get_erosion_level(erosion)
            pv_level_values[z][x] = get_peak_valley_level(peak_valley_value)

            # biome_params = BiomeParams(
            #     temperature,
            #     humidity,
            #     continentalness,
            #     erosion,
            #     weirdness,
            #     1.0
            # )
            # biome_values[z][x] = get_biome_id(biome_params)

            for y in range(size_3d_int[0]):
                density = d_noise_values[y][z][x]

                d_values[y][z][x] = density_adjusted = calculate_density(
                    density,
                    y,
                    base_height
                )
                is_solid_values[y][z][x] = is_solid = density_adjusted > 0

def plot_all_at_once(offset: glm.vec3 = glm.vec3(0)):
    all_at_once(offset)
    plot_3d_isometric(is_solid_values)

# from timeit import timeit
# num_tests = 10
# print(timeit(one_at_a_time, number=num_tests))
# print(timeit(all_at_once, number=num_tests))
# quit()

chunks = 16
end_value = (16 * 16) // 2
start_value = -end_value
for z in range(start_value, end_value, 16):
    for x in range(start_value, end_value, 16):
        plot_all_at_once(glm.vec3(x, 0, z))
quit()
# plot_noise_grid(c_noise_values, title="Continentalness", normalized=True)
# plot_noise_grid(c_level_values, title="Continentalness Leveled")
#
# plot_noise_grid(t_noise_values, title="Temperature", normalized=True)
# plot_noise_grid(t_level_values, title="Temperature Leveled")
#
# plot_noise_grid(h_noise_values, title="Humidity", normalized=True)
# plot_noise_grid(h_level_values, title="Humidity Leveled")
#
# plot_noise_grid(e_noise_values, title="Erosion", normalized=True)
# plot_noise_grid(e_level_values, title="Erosion Leveled")
#
# plot_noise_grid(w_noise_values, title="Weirdness", normalized=True)
# plot_noise_grid(pv_noise_values, title="Peak/Valley", normalized=True)
# plot_noise_grid(pv_level_values, title="Peak/Valley")

# plot_noise_grid(biome_values, title="Biome")

# plot_cross_section(d_noise_values)
# plot_cross_section(d_values)

# for z in range(100):
#     plot_cross_section(is_solid_values, slice_index=z)


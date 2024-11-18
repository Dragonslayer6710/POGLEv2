import random

import numpy as np
from Biome import *

width, height = 16, 384

known_seed = True
seed = 42 if known_seed else random.randint(0, 1_000_000)

c_ngen = ContinentalNoiseGenerator(seed)
t_ngen = TempNoiseGen(seed + 1)
h_ngen = HumidNoiseGen(seed + 2)
e_ngen = EroNoiseGen(seed + 3)
w_ngen = WeirdNoiseGen(seed + 4)
d_ngen = DenseNoiseGen(seed + 5)

c_level_values = np.empty((width, width))

t_level_values = np.empty((width, width))

h_level_values = np.empty((width, width))

e_level_values = np.empty((width, width))

pv_noise_values = np.empty((width, width))
pv_level_values = np.empty((width, width))

th_values = np.empty((width, width))

d_values = np.empty((height, width, width))
is_solid_values = np.empty((height, width, width))


def one_at_a_time():
    c_noise_values = np.empty((width, width))
    t_noise_values = np.empty((width, width))
    h_noise_values = np.empty((width, width))
    e_noise_values = np.empty((width, width))
    w_noise_values = np.empty((width, width))
    d_noise_values = np.empty((height, width, width))

    tallest_height = 0
    lowest_height = height
    biome_values = np.empty((width, width))
    for x in range(width):
        for z in range(width):
            c_noise_values[z][x] = continentalness = c_ngen.sample(x, z)
            c_level_values[z][x] = get_continentalness_level(c_noise_values[z][x])

            t_noise_values[z][x] = t_ngen.sample(x, z)
            t_level_values[z][x] = get_temperature_level(t_noise_values[z][x])

            h_noise_values[z][x] = h_ngen.sample(x, z)
            h_level_values[z][x] = get_humidity_level(h_noise_values[z][x])

            e_noise_values[z][x] = erosion = e_ngen.sample(x, z)
            e_level_values[z][x] = get_erosion_level(e_noise_values[z][x])

            w_noise_values[z][x] = w_ngen.sample(x, z)

            pv_noise_values[z][x] = peak_valley_value = calculate_peak_valley_value(w_noise_values[z][x])
            pv_level_values[z][x] = get_peak_valley_level(pv_noise_values[z][x])

            th_values[z][x] = base_height = calculate_terrain_height(
                continentalness, erosion, peak_valley_value
            )

            # new_limits = False
            # if base_height > tallest_height:
            #     tallest_height = base_height
            #     new_limits = True
            # if tallest_height >= 384:
            #     raise RuntimeError(f"Base Terrain Height ({base_height}) Too High!")
            #
            # if base_height < lowest_height:
            #     lowest_height = base_height
            #     new_limits = True
            # if lowest_height < 0:
            #     raise RuntimeError(f"Base Terrain Height ({base_height}) Too Low!")

            # biome_params = BiomeParams(
            #     t_noise_values[z][x],
            #     h_noise_values[z][x],
            #     c_noise_values[z][x],
            #     e_noise_values[z][x],
            #     w_noise_values[z][x],
            #     1.0
            # )
            # biome_values[z][x] = get_biome_id(biome_params)

            def normalize(x, min, max):
                # Normalize y to [0, 1]
                return (x - min) / (max - min)

            for y in range(height):
                d_noise_values[y][z][x] = d_ngen.sample(x, y, z)
                continue
                # bh_normal = normalize(base_height - y, 0, height-1)
                d_values[y][z][x] = density = calculate_density(
                    d_noise_values[y][z][x],
                    y,
                    base_height
                )
                is_solid_values[y][z][x] = density > 0


def all_at_once():
    origin_2d = glm.vec2(width / 2)
    extents_2d = glm.ivec2(width)
    grid_coords_2d, grid_shape_2d = get_grid_coords(origin_2d, extents_2d)

    c_noise_values = c_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
    t_noise_values = t_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
    h_noise_values = h_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
    e_noise_values = e_ngen.grid_sample(grid_coords_2d, grid_shape_2d)
    w_noise_values = w_ngen.grid_sample(grid_coords_2d, grid_shape_2d)

    grid_coords_3d, grid_shape_3d = get_grid_coords(
        glm.vec3(height / 2, origin_2d), glm.ivec3(height, extents_2d)
    )
    d_noise_values = d_ngen.grid_sample(grid_coords_3d, grid_shape_3d)
    for x in range(width):
        for z in range(width):
            c_level_values[z][x] = get_continentalness_level(c_noise_values[z][x])
            t_level_values[z][x] = get_temperature_level(t_noise_values[z][x])
            h_level_values[z][x] = get_humidity_level(h_noise_values[z][x])
            e_level_values[z][x] = get_erosion_level(e_noise_values[z][x])

from timeit import timeit
num_tests = 10
print(timeit(one_at_a_time, number=num_tests))
print(timeit(all_at_once, number=num_tests))
quit()
# plot_3d_isometric(is_solid_values)

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


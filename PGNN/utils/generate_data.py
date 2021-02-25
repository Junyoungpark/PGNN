import numpy as np
import ray
from GraphFloris.WindFarm import WindFarm


def prepare_data(n_samples,
                 wind_speed_range: list = [6.0, 15.0],
                 wind_direction_range: list = [0.0, 360.0],
                 num_turbine_list: list = [5, 10, 15, 20],
                 farm_config: dict = None,
                 verbose: bool = False):
    if farm_config is None:
        farm_config = {'x_grid_size': 2000, 'y_grid_size': 2000, 'min_distance_factor': 2.0,
                       'dist_cutoff_factor': 25.0}
    farm = WindFarm(5, **farm_config)

    nts = np.random.choice(num_turbine_list,
                           size=n_samples)
    wds = np.random.uniform(low=wind_direction_range[0],
                            high=wind_direction_range[1],
                            size=n_samples)
    wss = np.random.uniform(low=wind_speed_range[0],
                            high=wind_speed_range[1],
                            size=n_samples)

    gs, us = [], []
    for i, (n, wd, ws) in enumerate(zip(nts, wds, wss)):
        if i % 50 == 0 and verbose:
            print("sampling [{}]/[{}] layouts".format(i + 1, n_samples))
        farm.sample_layout(n)
        farm.update_graph(wind_speed=ws, wind_direction=wd)
        g, u = farm.observe()
        gs.append(g)
        us.append(u.view(-1))

    return gs, us


def prepare_data_mp(n_samples: int,
                    num_procs: int,
                    wind_speed_range: list = [6.0, 15.0],
                    wind_direction_range: list = [0.0, 360.0],
                    num_turbine_list: list = [5, 10, 15, 20],
                    farm_config: dict = None):
    if farm_config is None:
        farm_config = {'x_grid_size': 2000, 'y_grid_size': 2000, 'min_distance_factor': 2.0,
                       'dist_cutoff_factor': 25.0}

    nts = np.random.choice(num_turbine_list, size=n_samples)
    wds = np.random.uniform(low=wind_direction_range[0],
                            high=wind_direction_range[1],
                            size=n_samples)
    wss = np.random.uniform(low=wind_speed_range[0],
                            high=wind_speed_range[1],
                            size=n_samples)

    sub_nts = np.array_split(nts, num_procs)
    sub_wds = np.array_split(wds, num_procs)
    sub_wss = np.array_split(wss, num_procs)
    map = [generate_gs_ray.remote(nts, wds, wss, farm_config) for nts, wds, wss in zip(sub_nts, sub_wds, sub_wss)]
    rets = ray.get(map)

    gs, us = [], []
    for g, u in rets:
        gs.extend(g)
        us.extend(u)
    return gs, us


@ray.remote
def generate_gs_ray(nts, wds, wss, farm_config):
    gs, us = [], []
    farm = WindFarm(5, **farm_config)
    for nt, wd, ws in zip(nts, wds, wss):
        farm.sample_layout(nt)
        farm.update_graph(wind_speed=ws, wind_direction=wd)
        g, u = farm.observe()
        gs.append(g)
        us.append(u.view(-1))
    return gs, us

import os

import dgl
import numpy as np
import ray
import torch

from PGNN.utils.generate_data import generate_gs_ray


def calc_mape(pred, target):
    with torch.no_grad():
        rel_error = (target - pred) / (target + 1e-10)
        mape = rel_error.abs().mean()
        return mape


def test_over_speed_and_direction(model,
                                  use_xy,
                                  use_ws_only,
                                  wind_speeds,
                                  wind_directions,
                                  device: str = None,
                                  num_turbines: int = 20,
                                  num_samples_per_setup: int = 20,
                                  n_procs: int = None,
                                  farm_config: dict = None):
    ray.init()
    if farm_config is None:
        farm_config = {'x_grid_size': 2000,
                       'y_grid_size': 2000,
                       'min_distance_factor': 2.0,
                       'dist_cutoff_factor': 25.0}
    if n_procs is None:
        n_procs = int(os.cpu_count() * 0.5)

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    results = []
    for ws in wind_speeds:
        results_fixed_ws = []
        for wd in wind_directions:
            sub_nts = np.array_split(np.array([num_turbines] * num_samples_per_setup), n_procs)
            sub_wds = np.array_split(np.array([wd] * num_samples_per_setup), n_procs)
            sub_wss = np.array_split(np.array([ws] * num_samples_per_setup), n_procs)

            map = [generate_gs_ray.remote(nts, wds, wss, farm_config) for nts, wds, wss in
                   zip(sub_nts, sub_wds, sub_wss)]
            rets = ray.get(map)

            gs, us = [], []
            for g, u in rets:
                gs.extend(g)
                us.extend(u)

            gs = dgl.batch(gs).to(device)
            us = torch.stack(us).to(device)

            if use_ws_only:
                us = us[:, 0].view(-1, 1)

            with torch.no_grad():
                nf, ef = gs.ndata['feat'], gs.edata['feat']
                if use_xy:
                    nf = torch.cat([nf, gs.ndata['x'], gs.ndata['y']], dim=-1)
                pred = model(gs, nf, ef, us)
                ret = torch.nn.functional.mse_loss(pred,
                                                   gs.ndata['power']).cpu().numpy()
            results_fixed_ws.append(ret)
        results.append(results_fixed_ws)
    ray.shutdown()
    return results

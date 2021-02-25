import argparse
from os.path import join

import dgl
import ray
import torch
import torch.nn as nn
from adamp import AdamP
from box import Box
from dgl.data.utils import load_graphs

import wandb
from PGNN.nets.pgnn import PGNN
from PGNN.utils.generate_data import prepare_data_mp


def get_config():
    nf_dim = 3
    ef_dim = 2
    u_dim = 1

    cfg = Box({
        'model': {
            'edge_in_dim': ef_dim,
            'node_in_dim': nf_dim,
            'global_in_dim': u_dim,
            'n_pgn_layers': 3,
            'edge_hidden_dim': 50,
            'node_hidden_dim': 50,
            'global_hidden_dim': 50,
            'residual': True,
            'input_norm': True,
            'pgn_mlp_params': {'num_neurons': [256, 128],
                               'hidden_act': 'ReLU',
                               'out_act': 'ReLU'},
            'reg_mlp_params': {'num_neurons': [64, 32, 16],
                               'hidden_act': 'ReLU',
                               'out_act': 'ReLU'},
            'pgn_params': {'edge_aggregator': 'mean',
                           'global_node_aggr': 'mean',
                           'global_edge_aggr': 'mean'}
        },
        'train': {
            'batch_size': 512,
            'reset_g_every': 64,
            'num_procs': 6,
            'log_every': 100,
            'train_steps': 20000,
        }
    })
    return cfg


def main(device):
    ray.init()

    # if use_ws_only is true, global feature 'u' only contains wind speed
    # if use_ws_only is false, global feature 'u' contains wind speed and direction
    # we experimentally re-confirmed that using only wind speed as the global feature
    # results in better prediction results.
    use_ws_only = True

    config = get_config()

    # prepare validation data
    val_gs, labels = load_graphs('val_gs3.bin')
    val_us = labels['global_feat']
    val_us = val_us.to(device)
    val_gs = dgl.batch(val_gs).to(device)
    if use_ws_only:
        val_us = val_us[:, 0].view(-1, 1)

    m = PGNN(**config.model).to(device)
    print(m)

    crit = nn.MSELoss()
    opt = AdamP(m.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50)

    # setup wandb logger
    run = wandb.init(project='PGNN', config=config.to_dict())

    # save config
    config.to_yaml(join(wandb.run.dir, "model_config.yaml"))

    n_update = 0
    for epoch in range(config.train.train_steps):
        if n_update % config.train.reset_g_every == 0:
            gs, us = prepare_data_mp(config.train.batch_size,
                                     num_procs=config.train.num_procs)
            gs = dgl.batch(gs)
            us = torch.stack(us)

        gs = gs.to(device)
        us = us.to(device)
        if use_ws_only:
            us = us[:, 0].view(-1, 1)

        nf, ef = gs.ndata['feat'], gs.edata['feat']
        # augment input node feature to have Euclidean coordinates.
        # we found that this augmentation helps for better generalization.
        nf = torch.cat([nf, gs.ndata['x'], gs.ndata['y']], dim=-1)

        pred = m(gs, nf, ef, us)
        loss = crit(pred, gs.ndata['power'])
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

        # logging
        log_dict = dict()
        log_dict['lr'] = opt.param_groups[0]['lr']
        log_dict['loss'] = loss

        n_update += 1

        if n_update % config.train.log_every == 0:
            with torch.no_grad():
                m.eval()
                val_nf, val_ef = val_gs.ndata['feat'], val_gs.edata['feat']
                val_nf = torch.cat([val_nf, val_gs.ndata['x'], val_gs.ndata['y']], dim=-1)

                val_pred = m(val_gs, val_nf, val_ef, val_us)
                val_loss = crit(val_pred, val_gs.ndata['power'])
                log_dict['val_loss'] = val_loss
                m.train()

        wandb.log(log_dict)
        torch.save(m.state_dict(), join(wandb.run.dir, "model.pt"))
    run.finish()
    ray.shutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-device', type=str, default='cuda:0', help='fitting device')

    args = p.parse_args()
    main(args.device)

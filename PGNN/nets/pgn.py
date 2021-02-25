from functools import partial

import dgl
import dgl.function as dgl_fn
import torch
import torch.nn as nn

from PGNN.nets.Attention import PhysicsInducedAttention


class PGN(nn.Module):
    """
    Pytorch-DGL implementation of the physics-induced Graph Network Layer
    "https://www.sciencedirect.com/science/article/pii/S0360544219315555"
    """

    def __init__(self,
                 edge_model: nn.Module,
                 node_model: nn.Module,
                 global_model: nn.Module,
                 residual: bool,
                 use_attention: bool,
                 edge_aggregator: str = 'mean',
                 global_node_aggr: str = 'mean',
                 global_edge_aggr: str = 'mean'):
        super(PGN, self).__init__()

        # trainable models
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        if use_attention:
            self.attention_model = PhysicsInducedAttention(use_approx=False)

        # residual hook
        self.residual = residual
        self.use_attention = use_attention

        # aggregators
        self.edge_aggr = getattr(dgl_fn, edge_aggregator)('m', 'agg_m')
        self.global_node_aggr = global_node_aggr
        self.global_edge_aggr = global_edge_aggr

    def forward(self, g, nf, ef, u):
        """
        :param g: graphs
        :param nf: node features
        :param ef: edge features
        :param u: global features
        :return:

        """
        with g.local_scope():
            g.ndata['_h'] = nf
            g.edata['_h'] = ef

            # update edges
            repeated_us = u.repeat_interleave(g.batch_num_edges(), dim=0)
            edge_update = partial(self.edge_update, repeated_us=repeated_us)
            g.apply_edges(func=edge_update)

            # update nodes
            repeated_us = u.repeat_interleave(g.batch_num_nodes(), dim=0)
            node_update = partial(self.node_update, repeated_us=repeated_us)
            g.pull(g.nodes(),
                   message_func=dgl_fn.copy_e('m', 'm'),
                   reduce_func=self.edge_aggr,
                   apply_node_func=node_update)

            # update global features
            node_readout = dgl.readout_nodes(g, 'uh', op=self.global_node_aggr)
            edge_readout = dgl.readout_edges(g, 'uh', op=self.global_edge_aggr)
            gm_input = torch.cat([node_readout, edge_readout, u], dim=-1)
            updated_u = self.global_model(gm_input)

            updated_nf = g.ndata['uh']
            updated_ef = g.edata['uh']

            if self.residual:
                updated_nf = updated_nf + nf
                updated_ef = updated_ef + ef
                updated_u = updated_u + u

            return updated_nf, updated_ef, updated_u

    def edge_update(self, edges, repeated_us):
        sender_nf = edges.src['_h']
        receiver_nf = edges.dst['_h']
        ef = edges.data['_h']

        # update edge features
        em_input = torch.cat([sender_nf, receiver_nf, ef, repeated_us], dim=-1)
        e_updated = self.edge_model(em_input)

        if self.use_attention:
            # compute edge weights
            dd = edges.data['down_stream_dist']
            rd = edges.data['radial_dist']
            ws = edges.data['wind_speed']
            attn_input = torch.cat([dd, rd, ws], dim=-1)
            weights = self.attention_model(attn_input)
            updated_ef = weights * e_updated
        else:
            updated_ef = e_updated
        return {'m': updated_ef, 'uh': updated_ef}

    def node_update(self, nodes, repeated_us):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['_h']
        nm_input = torch.cat([agg_m, nf, repeated_us], dim=-1)
        updated_nf = self.node_model(nm_input)
        return {'uh': updated_nf}

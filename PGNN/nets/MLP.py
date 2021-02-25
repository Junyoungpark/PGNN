from typing import List

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: List[int] = [64, 32],
                 hidden_act: str = 'LeakyReLU',
                 out_act: str = 'LeakyReLU',
                 input_norm: str = None,
                 dropout_prob: float = 0.0):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)

        if dropout_prob > 0.0:
            self.dropout = nn.Dropout(dropout_prob)

        if input_norm is not None:
            if input_norm == 'batch':
                self.input_norm = nn.BatchNorm1d(input_dim)
            elif input_norm == 'layer':
                self.input_norm = nn.LayerNorm(input_dim)
            else:
                raise RuntimeError

    def forward(self, xs):
        if hasattr(self, 'input_norm'):
            xs = self.input_norm(xs)

        for i, layer in enumerate(self.layers):
            if i != 0 and hasattr(self, 'dropout'):
                xs = self.dropout(xs)
            xs = layer(xs)
        return xs

    def __repr__(self):
        msg = "MLP \n"
        if hasattr(self, 'input_norm'):
            msg += "Input Norm : {} \n".format(self.input_norm)
        msg += "Dimensions : {} \n".format([self.input_dim] + self.num_neurons + [self.output_dim])
        msg += "Hidden Act. : {} \n".format(self.hidden_act)
        msg += "Out Act. : {} \n".format(self.out_act)
        return msg

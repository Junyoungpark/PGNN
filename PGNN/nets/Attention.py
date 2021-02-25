import torch
import torch.nn as nn


class Normalizer(nn.Module):

    def __init__(self, num_feature, alpha=0.9):
        super(Normalizer, self).__init__()
        self.alpha = alpha
        self.num_feature = num_feature
        self.register_buffer('mean', torch.zeros(num_feature))
        self.register_buffer('var', torch.zeros(num_feature))
        self.register_buffer('std', torch.zeros(num_feature))
        self.w = nn.Parameter(torch.ones(num_feature))
        self.b = nn.Parameter(torch.zeros(num_feature))
        self.reset_stats()

    def forward(self, xs):
        xs = xs.view(-1, self.num_feature)  # Handling 1-batch case

        if self.training:
            mean_update = torch.mean(xs, dim=0)  # Get mean-values along the batch dimension
            self.mean = self.alpha * self.mean + (1 - self.alpha) * mean_update.data
            var_update = (1 - self.alpha) * torch.mean(torch.pow((xs - self.mean), 2), dim=0)
            self.var = self.alpha * self.var + var_update.data
            self.std = torch.sqrt(self.var + 1e-10)

        standardized = xs / self.std
        affined = standardized * torch.nn.functional.relu(self.w)

        return affined

    def reset_stats(self):
        self.mean.zero_()
        self.var.fill_(1)
        self.std.fill_(1)


class PhysicsInducedAttention(nn.Module):

    def __init__(self,
                 input_dim=3,
                 use_approx=True,
                 degree=5):
        """
        :param input_dim: (int) input_dim for PhysicsInducedBias layer
        :param use_approx: (bool) If True, exp() will be approximated with power series
        :param degree: (int) degree of power series approximation
        """
        super(PhysicsInducedAttention, self).__init__()
        self.input_dim = input_dim
        self.use_approx = use_approx
        self.degree = degree
        self.alpha = nn.Parameter(torch.zeros(1))
        self.r0 = nn.Parameter(torch.zeros(1))
        self.k = nn.Parameter(torch.zeros(1))

        self.alpha.data.fill_(1.0)
        self.r0.data.fill_(1.0)
        self.k.data.fill_(1.0)

        self.norm = Normalizer(self.input_dim)

    def forward(self, xs, degree=None):
        if degree is None:
            degree = self.degree
        interacting_coeiff = self.get_scaled_bias(xs, degree)
        interacting_coeiff = nn.functional.relu(interacting_coeiff)
        return interacting_coeiff

    @staticmethod
    def power_approx(fx, degree=5):
        ret = torch.ones_like(fx)
        fact = 1
        for i in range(1, degree + 1):
            fact = fact * i
            ret += torch.pow(fx, i) / fact
        return ret

    def get_scaled_bias(self, xs, degree=5):
        xs = self.norm(xs)
        eps = 1e-10
        inp = torch.split(xs, 1, dim=1)
        x, r, ws = inp[0], inp[1], inp[2]

        r0 = nn.functional.relu(self.r0 + eps)
        alpha = nn.functional.relu(self.alpha + eps)
        k = nn.functional.relu(self.k + eps)

        denom = r0 + k * x
        down_stream_effect = alpha * torch.pow((r0 / denom), 2)
        radial_input = -torch.pow((r / denom), 2)

        if self.use_approx:
            radial_effect = self.power_approx(radial_input, degree=degree)
        else:
            radial_effect = torch.exp(-torch.pow((r / denom), 2))

        interacting_coeiff = down_stream_effect * radial_effect

        return interacting_coeiff

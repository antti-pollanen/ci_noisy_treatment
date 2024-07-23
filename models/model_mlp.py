import torch
import torch.nn as nn
import torch.nn.functional as F

import models.model_common as model_common


class Mlp(nn.Module):
    def __init__(self, z_dim, hidden_sizes, uses_t_instead_of_w: bool):
        super().__init__()
        self.z_dim = z_dim
        self.y_mean_nn = model_common.FullyConnected(
            [self.z_dim + 1] + hidden_sizes + [1], final_activation=None
        )
        self.uses_t_instead_of_w = uses_t_instead_of_w

    def get_y_sd_estimate(
        self, z: torch.Tensor, t_or_w: torch.Tensor, y: torch.Tensor
    ) -> float:
        assert len(y.shape) == 2
        assert y.shape[1] == 1

        with torch.no_grad():
            y_mean = self.forward(z, t_or_w)
            return torch.sqrt(
                1 / (y.shape[0] - 1) * torch.sum(torch.square(y - y_mean))
            ).item()  # not unbiased, but will do

    def forward(self, z: torch.Tensor, t_or_w: torch.Tensor):
        return self.y_mean_nn(torch.cat((z, t_or_w), dim=1))

    def loss(self, batch, qw_term_weight=None):
        z, t, w, y, _ = batch
        t_or_w = t if self.uses_t_instead_of_w else w

        return F.mse_loss(self.forward(z, t_or_w), y)

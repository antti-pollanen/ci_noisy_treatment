import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

import models.model_common as model_common


class DecoderPars:
    def __init__(self, t_dist, w_dist, y_dist):
        assert len(t_dist.loc.shape) == 2, t_dist.loc.shape  # n_data x n_features
        assert (
            len(w_dist.loc.shape) == 3
        ), w_dist.loc.shape  # n_iw x n_data x n_features
        assert (
            len(y_dist.loc.shape) == 3
        ), y_dist.loc.shape  # n_iw x n_data x n_features
        self.t_dist, self.w_dist, self.y_dist = t_dist, w_dist, y_dist


class Decoder(nn.Module):
    def __init__(self, z_dim: int, hidden_sizes, known_w_sd: float | None):
        super().__init__()
        self.z_dim = z_dim

        if known_w_sd is not None:
            if known_w_sd == 0.0:
                known_w_sd = 1e-10
                print("Obtained known_w_sd of 0.0, setting to 1e-10 instead.")
            self.uses_constant_w_sd = True
            self.known_w_sd = torch.tensor(known_w_sd, dtype=torch.float32)
        else:
            self.uses_constant_w_sd = False
            self.w_sd_par = nn.Parameter(
                torch.tensor(0.1, dtype=torch.float32)
            )  # parameter that can be nonpositive

        self.y_mean_nn = model_common.FullyConnected(
            [self.z_dim + 1] + hidden_sizes + [1], final_activation=None
        )
        self.y_sd_par = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32)
        )  # parameter that can be nonpositive
        self.t_mean_nn = model_common.FullyConnected(
            [self.z_dim] + hidden_sizes + [1], final_activation=None
        )
        self.t_sd_nn = model_common.FullyConnected(
            [self.z_dim] + hidden_sizes + [1], final_activation=None
        )

    @property
    def w_sd(self):
        if self.uses_constant_w_sd:
            return self.known_w_sd
        return torch.exp(self.w_sd_par)

    @property
    def y_sd(self):
        return torch.exp(self.y_sd_par)

    def forward(self, z, q):
        """
        z -- (n_data x n_features) tensor
        q -- (n_iw x n_data x n_features) tensor
        """
        assert len(z.shape) == 2, z.shape
        assert len(q.shape) == 3 and q.shape[2] == 1, q.shape

        t_mean, log_t_sd = self.t_mean_nn(z), self.t_sd_nn(z)

        n_iw = q.shape[0]

        z = z.tile((n_iw, 1, 1))
        y_mean = self.y_mean_nn(torch.cat((z, q), dim=2))

        t_dist = dist.Normal(loc=t_mean, scale=torch.exp(log_t_sd))
        return DecoderPars(
            t_dist=t_dist,
            w_dist=dist.Normal(loc=q, scale=self.w_sd),
            y_dist=dist.Normal(loc=y_mean, scale=self.y_sd),
        )

    def y_mean_estimate(self, z, t):
        # z -- n_data x n_features tensor
        # t -- n_data x n_features tensor
        assert len(z.shape) == 2, z.shape
        assert len(t.shape) == 2, t.shape
        with torch.no_grad():
            return self.y_mean_nn(torch.cat((z, t), dim=1))

    def t_mean_estimate(self, z):
        # z -- n_data x n_feature tensor
        assert len(z.shape) == 2, z.shape
        return self.t_mean_nn(z)

    def t_sd_estimate(self, z):
        # z -- n_data x n_feature tensor
        assert len(z.shape) == 2, z.shape
        return self.t_sd_nn(z)


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_sizes):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = 1
        self.y_dim = 1
        self.q_mean_nn = model_common.FullyConnected(
            [self.z_dim + self.w_dim + self.y_dim] + hidden_sizes + [1],
            final_activation=None,
        )
        self.q_sd_nn = model_common.FullyConnected(
            [self.z_dim + self.w_dim + self.y_dim] + hidden_sizes + [1],
            final_activation=None,
        )

    # from observed variables to latent t
    def forward(self, z, w, y):
        input = torch.cat([z, w, y], dim=1)
        q_mean, q_sd = self.q_mean_nn(input), torch.exp(self.q_sd_nn(input))
        return dist.Normal(loc=q_mean, scale=q_sd)


def term_weights(qw_term_weight: float) -> tuple[float, float]:
    term_weights_sum = 2 * qw_term_weight + 2
    qw_term_weight = qw_term_weight / term_weights_sum
    ty_term_weight = 1 / term_weights_sum
    return qw_term_weight, ty_term_weight


class Ceme(nn.Module):
    def __init__(
        self,
        z_dim,
        n_importance_samples,
        n_likelihood_samples,
        encoder_hidden_sizes,
        decoder_hidden_sizes,
        known_w_sd=None,
    ):
        super().__init__()
        self.encoder = Encoder(z_dim, encoder_hidden_sizes)
        self.decoder = Decoder(z_dim, decoder_hidden_sizes, known_w_sd)
        self.n_importance_samples = n_importance_samples
        self.n_likelihood_samples = n_likelihood_samples

    # z: covariates, conditional model
    # out: full factorized p as well as q, for the values given as input, and for both t=0 and t=1
    def forward(self, z, w, y, n_iw=None):
        if n_iw is None:
            n_iw = self.n_importance_samples
        q_dist = self.encoder(z, w, y)
        q = q_dist.rsample(torch.Size([n_iw]))
        gen_model_pars = self.decoder(z, q)
        return q, q_dist, gen_model_pars

    def likelihood(self, z, w, y):
        with torch.no_grad():
            q, q_dist, gpars = self(z=z, w=w, y=y, n_iw=self.n_likelihood_samples)
            likelihood = -self.elbo_loss(
                w=w, y=y, q=q, q_dist=q_dist, gpars=gpars, qw_term_weight=1.0
            )
        return likelihood

    def elbo_loss(self, w, y, q, q_dist, gpars, qw_term_weight: float = 1.0):
        """
        The loss is the negative importance weighted ELBO.
        """

        # assumes q is rsampled from q_dist

        # assert len(z.shape) == 2 and z.shape[1] == 1, z.shape # n_data x n_features
        assert len(w.shape) == 2 and w.shape[1] == 1, w.shape  # n_data x n_features
        assert len(y.shape) == 2 and y.shape[1] == 1, y.shape  # n_data x n_features
        assert (
            len(q.shape) == 3 and q.shape[2] == 1
        ), q.shape  # n_iw x n_data x n_features

        n_iw = q.shape[0]
        n_data = q.shape[1]
        n_features = q.shape[2]
        assert n_features == 1  # haven't checked that this works for more features

        assert w.shape == torch.Size([n_data, n_features])
        assert y.shape == torch.Size([n_data, n_features])

        assert q_dist.loc.shape == torch.Size([n_data, n_features])
        assert q_dist.scale.shape == torch.Size([n_data, n_features])
        assert gpars.t_dist.loc.shape == torch.Size([n_data, n_features])
        assert gpars.t_dist.scale.shape == torch.Size([n_data, n_features])
        assert gpars.w_dist.loc.shape == torch.Size([n_iw, n_data, n_features])
        assert gpars.w_dist.scale.shape == torch.Size([n_iw, n_data, n_features])
        assert gpars.y_dist.loc.shape == torch.Size([n_iw, n_data, n_features])
        assert gpars.y_dist.scale.shape == torch.Size([n_iw, n_data, n_features])

        qw_term_weight, ty_term_weight = term_weights(qw_term_weight)

        # so here for q input has more shapes than dist
        # q: n_iw x n_data x n_features
        # t_dist: n_data x n_features

        # w: n_data x n_features
        # w_dist: n_iw x n_data x n_features

        # for y same as for w

        # and then for w and y input has less shapes than dist ()
        # check that both behave as expected?
        t_term = ty_term_weight * gpars.t_dist.log_prob(
            q
        )  # t_term gets at first shape n_iw
        w_term = qw_term_weight * gpars.w_dist.log_prob(w)
        y_term = ty_term_weight * gpars.y_dist.log_prob(y)

        assert len(t_term.shape) == 3, t_term.shape  # n_iw x n_data x n_features
        assert len(w_term.shape) == 3, w_term.shape  # n_iw x n_data x n_features
        assert len(y_term.shape) == 3, y_term.shape  # n_iw x n_data x n_features

        assert t_term.shape == torch.Size([n_iw, n_data, n_features])
        assert w_term.shape == torch.Size([n_iw, n_data, n_features])
        assert y_term.shape == torch.Size([n_iw, n_data, n_features])

        q_term = qw_term_weight * q_dist.log_prob(q)
        assert q_term.shape == torch.Size([n_iw, n_data, n_features])

        iw = (w_term + y_term + t_term - q_term).sum(
            axis=2
        )  # sum across features, assume here they are independent, for now there is only one anyway
        assert iw.shape == torch.Size([n_iw, n_data])

        elbo = iw.logsumexp(dim=0) - np.log(iw.shape[0])
        assert elbo.shape == torch.Size([n_data])

        elbo = elbo.mean()

        return -elbo

    def loss(self, batch, qw_term_weight=1.0):
        z, _, w, y, _ = batch
        q, q_dist, gpars = self(z=z, w=w, y=y)
        return self.elbo_loss(
            w=w, y=y, q=q, q_dist=q_dist, gpars=gpars, qw_term_weight=qw_term_weight
        )

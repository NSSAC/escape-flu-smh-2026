"""Compute seasonality."""

import random
import operator
import pickle
from functools import reduce
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tqdm
import polars as pl

import torch
import torch.nn as nn
import torch.distributions as dist
import torch.distributions.constraints as constr
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def prod(x):
    return reduce(operator.mul, x, 1)


def xscale(x):
    return (x - 1.0) / 365.0


def xscale_std(x):
    return x / 365.0


@dataclass
class SeasonalityDataset:
    num_years: int
    num_regions: int
    years: list[int]
    regions: list[str]
    year_index: dict[int, int]
    region_index: dict[str, int]

    X: dict[tuple[int, str], torch.Tensor]  # shape(n_samples)
    y: dict[tuple[int, str], torch.Tensor]  # shape(n_samples)


def make_seasonality_dataset(target_data: pl.DataFrame) -> SeasonalityDataset:
    X = defaultdict(list)
    y = defaultdict(list)

    target_data = target_data.filter(
        pl.col("age_group") == "0-130", pl.col("target") == "inc hosp"
    )

    region_col = target_data["abbreviation"]
    date = target_data["date"]
    year_col = date.dt.year()
    ordinal_day_col = date.dt.ordinal_day()
    observation_col = target_data["observation"]

    for region, year, ordinal_day, observation in zip(
        region_col, year_col, ordinal_day_col, observation_col
    ):
        if observation > 0 and region != "US":
            X[year, region].append(ordinal_day)
            y[year, region].append(observation)

    years = set()
    regions = set()
    for (year, region), x in X.items():
        if len(x) > 1:
            years.add(year)
            regions.add(region)

    years = sorted(years)
    regions = sorted(regions)
    year_index = {y: i for i, y in enumerate(years)}
    region_index = {r: i for i, r in enumerate(regions)}

    nX = dict()
    ny = dict()
    for (year, region), x in X.items():
        if year in years and region in regions:
            nX[year, region] = torch.tensor(x, dtype=torch.float)
            ny[year, region] = torch.tensor(y[year, region], dtype=torch.float)

    return SeasonalityDataset(
        num_years=len(years),
        num_regions=len(regions),
        years=years,
        regions=regions,
        year_index=year_index,
        region_index=region_index,
        X=nX,
        y=ny,
    )


# Seasonaility model: 2 level hierarchial model
# Bi Modal: sinusoidal curves
# Day indexing == polars ordinal day (Jan 1 = 1)
#
# Winter peak prior mean: 7.33
#   [Jan 3, Jan 4, Jan 6, Jan 7, Jan 15, Jan 9]
#   Prior standard dev of mean: 30 / 3 (+- 3 sigma = 1 month)
# Summer peak prior mean: 238.0
#   [Sep 6, Sep 9, July 30, Aug 28]
#   Prior standard dev of mean: 60 / 3 (+- 3 sigma = 2 months)
# Prior mean is obtained by taking the mean of prior years peak days
# for whole us for "inc hosp" target date


class SeasonalityModel(nn.Module):
    def __init__(self, num_years: int, num_regions: int):
        super().__init__()

        self.num_years = num_years
        self.num_regions = num_regions

        # 2 x phase, 2 x gamma, mu = 5
        self.num_latents_per_level = 5

        # latents
        # fmt: off
        self.latent_shape = dict(
            g_phase_w = (1, 1),
            g_phase_s = (1, 1),
            g_gamma_w = (1, 1),
            g_gamma_s = (1, 1),
            g_mu = (1, 1),

            l1_phase_w = (1, self.num_regions),
            l1_phase_s = (1, self.num_regions),
            l1_gamma_w = (1, self.num_regions),
            l1_gamma_s = (1, self.num_regions),
            l1_mu = (1, self.num_regions),

            l2_phase_w = (self.num_years, self.num_regions),
            l2_phase_s = (self.num_years, self.num_regions),
            l2_gamma_w = (self.num_years, self.num_regions),
            l2_gamma_s = (self.num_years, self.num_regions),
            l2_mu = (self.num_years, self.num_regions),

            epsilon = (1,),
        )

        pos_transform = dist.biject_to(constr.positive)

        self.latent_transform = dict(
            g_gamma_w = pos_transform,
            g_gamma_s = pos_transform,

            l1_gamma_w = pos_transform,
            l1_gamma_s = pos_transform,

            l2_gamma_w = pos_transform,
            l2_gamma_s = pos_transform,

            epsilon = pos_transform,
        )

        self.latent_parent = dict(
            l1_phase_w = "g_phase_w",
            l1_phase_s = "g_phase_s",
            l1_gamma_w = "g_gamma_w",
            l1_gamma_s = "g_gamma_s",
            l1_mu = "g_mu",

            l2_phase_w = "l1_phase_w",
            l2_phase_s = "l1_phase_s",
            l2_gamma_w = "l1_gamma_w",
            l2_gamma_s = "l1_gamma_s",
            l2_mu = "l1_mu",
        )
        # fmt: on

        self.latent_indices = [0]
        for shape in self.latent_shape.values():
            size = prod(shape)
            self.latent_indices.append(self.latent_indices[-1] + size)

        self.num_latents = self.latent_indices[-1]

        self.q_loc = nn.Parameter(torch.randn(self.num_latents))
        self.q_cov_unconstrained = nn.Parameter(torch.eye(self.num_latents))

    def forward(
        self,
        dataset: SeasonalityDataset,
        groups: list[tuple[int, str]],
        num_particles: int = 30,
    ):
        device = self.q_loc.device
        psd_transform = dist.PositiveDefiniteTransform()

        elbo = torch.tensor(0.0, device=device)

        q_cov = psd_transform(self.q_cov_unconstrained)
        q_latents = dist.MultivariateNormal(self.q_loc, q_cov).rsample((num_particles,))
        elbo -= dist.MultivariateNormal(self.q_loc, q_cov).log_prob(q_latents).sum()

        latents = dict()
        for i, (var, shape) in enumerate(self.latent_shape.items()):
            begin, end = self.latent_indices[i], self.latent_indices[i + 1]
            latents[var] = q_latents[:, begin:end].reshape(num_particles, *shape)

        for var, transform in self.latent_transform.items():
            latents[var] = transform(latents[var])

        # \phase_w ~ Normal(7.33, 30.0 / 3)
        elbo += (
            dist.Normal(xscale(7.33), xscale_std(30.0 / 3))
            .log_prob(latents["g_phase_w"])
            .sum()
        )
        # \phase_s ~ Normal(238.0, 60.0 / 3)
        elbo += (
            dist.Normal(xscale(238.0), xscale_std(60.0 / 3))
            .log_prob(latents["g_phase_s"])
            .sum()
        )
        # \gamma_w, \gamma_s ~ Gamma(shape=1.0, rate=1.0)
        elbo += dist.Gamma(1.0, 1.0).log_prob(latents["g_gamma_w"]).sum()
        elbo += dist.Gamma(1.0, 1.0).log_prob(latents["g_gamma_s"]).sum()
        # \mu ~ Normal(0.0, 1.0)
        elbo += dist.Normal(0, 1.0).log_prob(latents["g_mu"]).sum()
        # \epsilon ~ HalfCauchy(10.0)
        elbo += dist.HalfCauchy(5.0).log_prob(latents["epsilon"]).sum()

        # Hierarchial Scale = Normal(0, 2.0)
        for child, parent in self.latent_parent.items():
            elbo += dist.Normal(latents[parent], 2.0).log_prob(latents[child]).sum()

        epsilon = latents["epsilon"][:, 0].unsqueeze(-1)
        for year, region in groups:
            year_i = dataset.year_index[year]
            region_i = dataset.region_index[region]

            # These have shape (num_particles, 1)
            phase_w = latents["l2_phase_w"][:, year_i, region_i].unsqueeze(-1)
            phase_s = latents["l2_phase_s"][:, year_i, region_i].unsqueeze(-1)
            gamma_w = latents["l2_gamma_w"][:, year_i, region_i].unsqueeze(-1)
            gamma_s = latents["l2_gamma_s"][:, year_i, region_i].unsqueeze(-1)
            mu = latents["l2_mu"][:, year_i, region_i].unsqueeze(-1)

            X = xscale(dataset.X[year, region]).unsqueeze(0)  # (1, num_samples)
            y_true = dataset.y[year, region].unsqueeze(0)  # (1, num_samples)
            y_true /= y_true.max()
            num_samples = X.shape[0]

            # y_w = \gamma_w cos(2 \pi (x - \phase_w) / 365)
            # y_s = \gamma_s cos(2 \pi (x - \phase_s) / 365)
            y_w = gamma_w * torch.cos(2 * torch.pi * (X - phase_w))
            y_s = gamma_s * torch.cos(2 * torch.pi * (X - phase_s))
            y_mean = mu + y_w + y_s

            # y = exp{\mu + y_w + y_s + epsilon}
            y_true = torch.log(y_true)
            elbo += dist.Normal(y_mean, epsilon).log_prob(y_true).sum() / num_samples

        return elbo / num_particles

    def predict(
        self,
        dataset: SeasonalityDataset,
        num_particles: int = 100,
    ):
        with torch.no_grad():
            psd_transform = dist.PositiveDefiniteTransform()

            q_cov = psd_transform(self.q_cov_unconstrained)
            q_latents = dist.MultivariateNormal(self.q_loc, q_cov).rsample(
                (num_particles,)
            )

            latents = dict()
            for i, (var, shape) in enumerate(self.latent_shape.items()):
                begin, end = self.latent_indices[i], self.latent_indices[i + 1]
                latents[var] = q_latents[:, begin:end].reshape(num_particles, *shape)

            for var, transform in self.latent_transform.items():
                latents[var] = transform(latents[var])

            ret_mean = dict()
            ret_std = dict()

            # epsilon = latents["epsilon"][:, 0].unsqueeze(-1)
            for year, region in dataset.X:
                year_i = dataset.year_index[year]
                region_i = dataset.region_index[region]
                # These have shape (num_particles, 1)
                phase_w = latents["l2_phase_w"][:, year_i, region_i].unsqueeze(-1)
                phase_s = latents["l2_phase_s"][:, year_i, region_i].unsqueeze(-1)
                gamma_w = latents["l2_gamma_w"][:, year_i, region_i].unsqueeze(-1)
                gamma_s = latents["l2_gamma_s"][:, year_i, region_i].unsqueeze(-1)
                mu = latents["l2_mu"][:, year_i, region_i].unsqueeze(-1)

                X = xscale(dataset.X[year, region]).unsqueeze(0)  # (1, num_samples)
                y_true = dataset.y[year, region].unsqueeze(0)  # (1, num_samples)

                # y_w = \gamma_w cos(2 \pi (x - \phase_w) / 365)
                # y_s = \gamma_s cos(2 \pi (x - \phase_s) / 365)
                y_w = gamma_w * torch.cos(2 * torch.pi * (X - phase_w))
                y_s = gamma_s * torch.cos(2 * torch.pi * (X - phase_s))
                y_mean = torch.exp(mu + y_w + y_s) * y_true.max()

                # y_mean.shape = (num_particles, num_samples)

                ret_mean[year, region] = y_mean.mean(dim=0)
                ret_std[year, region] = y_mean.std(dim=0)

            return ret_mean, ret_std


def maximize_elbo(
    model: SeasonalityModel,
    dataset: SeasonalityDataset,
    writer: SummaryWriter,
    output_dir: Path,
    num_particles: int = 100,
    num_epochs: int = 20000,
    lr: float = 0.001,
    grad_clip: float = 10.0,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    for k in dataset.X:
        dataset.X[k] = dataset.X[k].to(device)
        dataset.y[k] = dataset.y[k].to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm.trange(1, num_epochs + 1):
        groups = list(dataset.X)
        num_groups = len(groups)
        sampled_groups = random.sample(groups, num_groups // 10)

        optimizer.zero_grad()
        loss = -model.forward(dataset, sampled_groups, num_particles)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        writer.add_scalar("loss", loss.item(), epoch)

        if epoch % 1000 == 0:
            with open(output_dir / f"model-{epoch}.pkl", "wb") as fobj:
                pickle.dump(model, fobj, protocol=pickle.HIGHEST_PROTOCOL)

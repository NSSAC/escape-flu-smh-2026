"""BoTorch Optimizer Helpers."""

import math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Any

from numpy._core import dtype
from numpy._core.numeric import ndarray
import pandas as pd
import numpy as np
from scipy.stats import qmc

import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import (
    qLogExpectedImprovement,
    qProbabilityOfImprovement,
    PosteriorMean,
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.optim import optimize_acqf

PARAMS = ["baseline_tau", "tau_seasonality_scale"]
TARGET = "hosp_rmse"
PARAM_RANGE = {
    "baseline_tau": (1e-4, 1e-1),
    "tau_seasonality_scale": (1e-4, 1e-1),
}

NUM_REPLICATES = 1

NUM_INIT_POINTS = 2**5
NUM_STEP_POINTS = 5
NUM_STEPS = 10

MC_SAMPLES = 512
NUM_RESTARTS = 10
RAW_SAMPLES = 4096


def standardize_params(X: np.ndarray) -> np.ndarray:
    for i, param in enumerate(PARAMS):
        x_min, x_max = PARAM_RANGE[param]
        x_log_min, x_log_max = np.log(x_min), np.log(x_max)
        x_log_min, x_log_max = np.log(x_min), np.log(x_max)
        shift = x_log_min
        scale = x_log_max - x_log_min

        X[:, i] = np.clip(X[:, i], a_min=x_min, a_max=x_max)
        X[:, i] = np.log(X[:, i])
        X[:, i] -= shift
        X[:, i] /= scale
    return X


def unstandardize_params(X: np.ndarray) -> np.ndarray:
    for i, param in enumerate(PARAMS):
        x_min, x_max = PARAM_RANGE[param]
        x_log_min, x_log_max = np.log(x_min), np.log(x_max)
        x_log_min, x_log_max = np.log(x_min), np.log(x_max)
        shift = x_log_min
        scale = x_log_max - x_log_min

        X[:, i] *= scale
        X[:, i] += shift
        X[:, i] = np.exp(X[:, i])

    return X


def make_init_points() -> np.ndarray:
    sampler = qmc.Sobol(d=len(PARAMS), scramble=True)
    X = sampler.random_base2(int(math.log2(NUM_INIT_POINTS)))
    X = unstandardize_params(X)
    return X


@dataclass
class Model:
    gp: SingleTaskGP
    train_size: int
    best_f: float


def train_model(data: list[dict[str, Any]]):
    X, y = [], []
    for row in data:
        X.append([row[param] for param in PARAMS])
        y.append(row[TARGET])
    X = torch.tensor(X, dtype=torch.float64)
    y = -torch.tensor(y, dtype=torch.float64)

    input_transform = Normalize(d=len(PARAMS))
    outcome_transform = Standardize(m=1)
    gp = SingleTaskGP(
        train_X=X,
        train_Y=y,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    return Model(gp=gp, train_size=len(data), best_f=y.max().item())


def make_step_points(model: Model):
    configs: list[dict[str, float]] = []

    dim = len(PARAMS)
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(torch.double)
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    qEI = qLogExpectedImprovement(
        model=model.gp,
        best_f=model.best_f,
        sampler=qmc_sampler,
    )
    next_x_qei, _ = optimize_acqf(
        qEI,
        bounds=bounds,
        q=NUM_STEP_POINTS,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    next_x_qei = next_x_qei.detach().cpu().numpy()
    next_x_qei = unstandardize_params(next_x_qei)

    for i in next_x_qei.shape[0]:
        row = {}
        for j in next_x_qei.shape[1]:
            row[PARAMS[j]] = next_x_qei[i, j].item()
        configs.append(row)

    qPI = qProbabilityOfImprovement(
        model=model.gp,
        best_f=model.best_f,
        sampler=qmc_sampler,
    )
    next_x_qpi, _ = optimize_acqf(
        qPI,
        bounds=bounds,
        q=NUM_STEP_POINTS,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )
    next_x_qpi = next_x_qpi.detach().cpu().numpy()
    next_x_qpi = unstandardize_params(next_x_qpi)

    for i in next_x_qpi.shape[0]:
        row = {}
        for j in next_x_qpi.shape[1]:
            row[PARAMS[j]] = next_x_qpi[i, j].item()
        configs.append(row)

    return configs


def make_post_mean_point(model: Model):
    dim = len(X_COLS)
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim]).double()

    posterior_transform = ScalarizedPosteriorTransform(Y_WEIGHTS)

    pmean = PosteriorMean(model=model.gp, posterior_transform=posterior_transform)
    pmean_x, _ = optimize_acqf(
        pmean,
        bounds=bounds,
        q=1,
        num_restarts=16,
        raw_samples=64,
    )
    pmean_x = pmean_x.detach().cpu().numpy()
    pmean_x = unstandardize_params(pmean_x)
    pmean_x["sample_type"] = "pmean"
    return pmean_x


def minimize_par(
    func: Callable,
    func_kwargs: dict,
    results_file: str | Path,
    chunksize: int,
    ray_options_kwargs: dict,
):
    next_trial_id = 0

    print("Generating initial points.")

    round_start = datetime.now()
    init_X = make_init_points()
    round_time_s = (datetime.now() - round_start).total_seconds()

    init_X["round"] = 0
    kwargs_list = []
    for x in init_X.to_dict(orient="records"):
        for replicate in range(NUM_REPLICATES):
            kwargs = {}
            kwargs["trial_id"] = next_trial_id
            kwargs["replicate_id"] = replicate
            kwargs["calib_configs"] = x
            kwargs.update(func_kwargs)

            kwargs_list.append(kwargs)
            next_trial_id += 1

    print("Waiting for initial points.")
    results = ray_map_kwargs(
        func, kwargs_list, chunksize=chunksize, **ray_options_kwargs
    )
    results = pd.DataFrame(results)
    results["round_time_s"] = round_time_s

    data = pd.concat([init_X, results], axis=1)
    data.to_csv(results_file, index=False)

    for round in range(1, NUM_STEPS + 1):
        model = train_model(data)
        print("Best observed f:", model.best_f)
        print(f"Generating points for step {round}.")

        round_start = datetime.now()
        step_X = make_step_points(model)
        round_time_s = (datetime.now() - round_start).total_seconds()

        step_X["round"] = round
        kwargs_list = []
        for x in step_X.to_dict(orient="records"):
            for replicate in range(NUM_REPLICATES):
                kwargs = {}
                kwargs["trial_id"] = next_trial_id
                kwargs["replicate_id"] = 0
                kwargs["calib_configs"] = x
                kwargs.update(func_kwargs)

                kwargs_list.append(kwargs)
                next_trial_id += 1

        print(f"Waiting for points from step {round}.")
        results = ray_map_kwargs(
            func, kwargs_list, chunksize=NUM_STEP_POINTS, **ray_options_kwargs
        )
        results = pd.DataFrame(results)
        results["round_time_s"] = round_time_s

        step_data = pd.concat([step_X, results], axis=1)
        data = pd.concat([data, step_data], axis=0, ignore_index=True)
        data.to_csv(results_file, index=False)

    model = train_model(data)
    print("Best observed f", model.best_f)

    return model, data


def explore_par(
    func: Callable,
    func_kwargs: dict,
    results_file: str | Path,
    chunksize: int,
    ray_options_kwargs: dict,
):
    next_trial_id = 0

    round_start = datetime.now()
    init_X = make_init_points()
    round_time_s = (datetime.now() - round_start).total_seconds()

    init_X["round"] = 0
    kwargs_list = []
    for x in init_X.to_dict(orient="records"):
        for replicate in range(1):
            kwargs = {}
            kwargs["trial_id"] = next_trial_id
            kwargs["replicate_id"] = replicate
            kwargs["calib_configs"] = x
            kwargs.update(func_kwargs)

            kwargs_list.append(kwargs)
            next_trial_id += 1

    print("Waiting for initial points.")
    results = ray_map_kwargs(
        func, kwargs_list, chunksize=chunksize, **ray_options_kwargs
    )
    results = pd.DataFrame(results)
    results["round_time_s"] = round_time_s

    data = pd.concat([init_X, results], axis=1)
    data.to_csv(results_file, index=False)

    return data

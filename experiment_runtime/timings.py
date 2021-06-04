#! /usr/bin/env python
#SBATCH --gres=gpu:1 -p gpu
#SBATCH --time=00:05:00
#SBATCH --mem=4GB
#SBATCH --exclude=gpu[062-083] --exclude=gpu[060-061] --exclud=gpu[036-037]

import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time


class PeriodicBasis(torch.nn.Module):
    
    def forward(self, x):
        return torch.cat([
            (2 * np.pi * x).cos(),
            (2 * np.pi * x).sin()
        ], dim=-1)
    

from bgflow import (
    DenseNet,
    
    WrapCDFTransformerWithInverse,
    GridInversion,
    AffineSigmoidComponentInitGrid,
    
    MixtureCDFTransformer,
    
    AffineSigmoidComponents,   
    MoebiusComponents,
    NonCompactAffineSigmoidComponents,
    
    ConditionalSplineTransformer,
    
    SmoothRamp,
    SmoothRampWithTrainableExponent,
    BisectionRootFinder,
    
    ConstrainedBoundaryCDFTransformer,
    
    SequentialFlow,
    CouplingFlow,
    SplitFlow,
    InverseFlow,
    SwapFlow,
    
)

        
def make_net(d_in, d_hidden, d_out, activation, periodic=False, init_weight=1.):
    return torch.nn.Sequential(
        PeriodicBasis() if periodic else torch.nn.Identity(),
        DenseNet([d_in * (2 if periodic else 1), d_hidden, d_hidden, d_out], activation, weight_scale=init_weight, bias_scale=init_weight)
    )

def make_transformer(
    d_in,
    d_out,
    d_hidden,
    n_components,
    transformer_type="spline",
    inverse="approx",
    periodic=False,
    zero_boundary_left=False,
    zero_boundary_right=False,
    activation=torch.nn.SiLU(),
    smoothness_type="type1",
    init_weight=1.5,
    min_density=1e-6,
    verbose=False,
    n_grid_inversion=10
):
    offset = 1 if not periodic else 0
    if transformer_type == "spline":
        t = ConditionalSplineTransformer(
            params_net=make_net(d_in, d_hidden, d_out * (n_components * 3 + offset), activation, periodic=periodic, init_weight=init_weight),
            is_circular=periodic
        )
    elif transformer_type == "bump":
        t = MixtureCDFTransformer(
                compute_weights=make_net(d_in, d_hidden, d_out * n_components, activation, periodic=periodic),
                compute_components=AffineSigmoidComponents(
                    conditional_ramp=SmoothRamp(
                        compute_alpha=make_net(d_in, d_hidden, d_out * n_components, activation, periodic=periodic),
                        unimodal=True,
                        ramp_type=smoothness_type
                    ),
                    log_sigma_bound=torch.tensor(1.),
                    compute_params=make_net(d_in, d_hidden, d_out * (3 * n_components), activation, periodic=periodic),
                    min_density=torch.tensor(min_density),
                    periodic=periodic,
                    zero_boundary_left=zero_boundary_left,
                    zero_boundary_right=zero_boundary_right

                ),
        )
    else:
        raise ValueError(f"unknown transformer_type={transformer_type}")
        
    if inverse=="approx":
        return WrapCDFTransformerWithInverse(
            transformer=t,        
            oracle=GridInversion(
                transformer=t,
                compute_init_grid=lambda x,y: torch.linspace(0, 1, n_grid_inversion, dtype=y.dtype, device=y.device).view(-1, 1, 1).repeat(1, *y.shape),
                verbose=verbose,
                abs_tol=1e-6
            )
        )
    elif inverse=="exact":
        return t
    else:
        raise ValueError(f"unknown inverse={inverse}")

        


def run_benchmark(
    n_batch,
    dim_x,
    dim_y,
    n_hidden,
    n_components,
    n_runs,
    inverse_flow,
    transformer_type,
    inverse_algo,
    n_grid_inversion,
    backward
):   
    x = torch.zeros(n_batch, dim_x).cuda()
    y = torch.rand(n_batch, dim_y).cuda()
    t = make_transformer(
        dim_x, 
        dim_y, 
        n_hidden, 
        n_components, 
        transformer_type=transformer_type, 
        verbose=False, 
        inverse=inverse_algo,
        n_grid_inversion=n_grid_inversion
    ).cuda()


    if backward:
        t1 = time()
        for i in range(n_runs):
            a = t(x, y, inverse=inverse_flow)
            (a[0].sum() + a[1].sum()).backward()
        t2 = time()
    else:
        with torch.no_grad():
            t1 = time()
            for i in range(n_runs):
                t(x, y, inverse=inverse_flow)
            t2 = time()
    
    avg_diff_ms = (t2 - t1) * 1000 / n_runs
    
    return avg_diff_ms


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batch", default=128, type=int)
    parser.add_argument("--dim_x", default=8, type=int)
    parser.add_argument("--dim_y", default=8, type=int)
    parser.add_argument("--n_hidden", default=64, type=int)
    parser.add_argument("--n_components", default=8, type=int)
    parser.add_argument("--n_runs", default=10, type=int)
    parser.add_argument("--inverse_flow", default=False, type=bool)
    parser.add_argument("--transformer_type", default="spline", type=str, choices=["spline", "bump"])
    parser.add_argument("--inverse_algo", default="approx", type=str, choices=["approx", "exact"])
    parser.add_argument("--n_grid_inversion", default=10, type=int)
    parser.add_argument("--backward", default=False, type=bool)
    
    args = parser.parse_args()    
    print(run_benchmark(**vars(args)))

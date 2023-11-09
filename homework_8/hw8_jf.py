"""Functions and plotting for UQ HW 8.

Usage:
```shell
# If in dir containing hw8_jf.py, run the below
python hw8_jf.py --alpha 5 --n-samples 10_000 --burnin 1000
```
"""


from typing import Callable, Dict, Optional, Tuple

from argparse import ArgumentParser

import matplotlib.pyplot as plt

from numpy import (
    zeros, ndarray, exp, meshgrid, linspace, absolute, power, mean, array,
    diagonal, log, identity
)

# 'as' naming follows R conventions
from numpy.random import ( 
    uniform as runif,               # sample random uniform
    normal as rnorm,                # sample random normal
    multivariate_normal as mvrnorm, # sample multivariate random normal
    seed,
    random
)

from scipy.stats import norm, multivariate_normal
dnorm = norm.pdf      
mvdnorm = multivariate_normal.pdf

from tqdm import tqdm


def multivariate_metropolis_hastings(
    x0: ndarray, 
    n_samples: int,
    target_pdf: Callable, 
    alpha: Optional[float] = None,
    verbose: bool = True) -> Tuple[ndarray, ndarray]:
    """Samples from multivariate target distribution through MCMC.

    Args:
        x0: Initial sample array. This is unlikely to come from the target 
            distribution, though over the course of the algorithm, samples
            will converge to being drawn from the target distribution.
        n_samples: Number of samples you wish to draw from the target 
            distribution.
        target_pdf: Probability density function from which you wish to sample.
        alpha: Covariance matrix alpha. 
        verbose: True to print tqdm progress, false otherwise.

    Returns:
        A tuple where the first element is an array of samples drawn from the 
        target distribution and the second element are the acceptance
        probabilities (aka density ratios) throughout the course of the 
        algorithm.

    References:
        [1] : Smith, R.C. (2014). Chapter 8.3.1 and 8.3.3 in "Uncertainty 
            Quantification: Theory, Implementation, and Applications." SIAM
        [2] : Ravenzwaaij, D.v. et al. A simple introduction to Markov 
            Chain Monte–Carlo sampling. Psychon Bull Rev (2018) 25:143-154.
        [3] : Xiu, D. (2010). Chapter 8.2 in "Numerical Methods for Stochastic 
            Computations: A Spectral Method Approach". Princeton University 
            Press.
        [4] : Stephens, M. "The Metropolis Hastings Algorithm"
            url: https://stephens999.github.io/fiveMinuteStats/MH_intro.html
        [5] : Lecture 9 Slides, UvA Uncertainty Quantification.
    """ 
    # Force x0 to be vector
    if not isinstance(x0, ndarray):
        raise TypeError(f"`x0` must be `ndarray` but is type {type(x0)}")
    else:
        if len(x0.shape) != 1:
            raise ValueError(f"`x0` must be vector but is shape {x0.shape}")
        
    # Create covariance matrix
    n_dims = len(x0)
    K_q = alpha*identity(n_dims) 

    # Initialize samples array with guess
    samples = zeros(shape=(n_samples+1, n_dims))
    samples[0] = x0

    # For tracking the acceptance probabilties during MCMC sampling
    acceptance_probabilities = [] 

    # TODO: Track markov chain running mean

    # Perform MCMC sampling w/ metropolis hastings algorithm
    for i in tqdm(
        range(1, n_samples+1), 
        desc="Multivariate Metropolis-Hastings", 
        disable=not verbose):

        prev_sample = samples[i-1]                                  # u_{n-1}

        # Compute proposal
        proposal = mvrnorm(mean=prev_sample, cov=K_q)               # u*

        # Compute pdfs for scaling target densities 
        mvdnorm_kwargs = dict(mean=prev_sample, cov=K_q)
        q_proposal = mvdnorm(proposal, **mvdnorm_kwargs)            # q(w, u)
        q_prev_sample = mvdnorm(prev_sample, **mvdnorm_kwargs)      # q(u, w)
    
        # Compute scaled densities using target distribution pdf
        density_proposal = target_pdf(proposal)*q_proposal          # p(w)*q(w,u)
        density_prev_sample = target_pdf(prev_sample)*q_prev_sample # p(u)*q(u,w)
        
        density_ratio = density_proposal/density_prev_sample        

        # compute acceptance probability and update list
        acceptance_probability = accept(density_ratio)
        acceptance_probabilities.append(acceptance_probability)
    
        # Determine accept/rejection of proposed sample
        if acceptance_probability > runif(0, 1):
            samples[i] = proposal
        else:
            samples[i] = prev_sample
 
    return samples, array(acceptance_probabilities)


def V(x, y):
    return power((x - y - 1), 4) + absolute(x + y - 6)


def hw8_pdf(u: ndarray):
    return exp(-V(u[0], u[1]))


def accept(density_ratio: float):
    return min(1, density_ratio)


if __name__ == "__main__":
    ## CLI
    cli_desc = "Plotting script for UQ homework 8." 
    parser = ArgumentParser(description=cli_desc )
    parser.add_argument(
        "--alpha", 
        help="covariance matrix alpha (REQUIRED).", 
        required=True, 
        type=float)
    parser.add_argument(
        "--n-samples", 
        help="number of MCMC iterations. (default: 500)", 
        type=int, 
        default=500)
    parser.add_argument(
        "--burnin", 
        help="number of initial samples to discard from result of MCMC sampling"
        " (default: 0).", 
        default=0, type=int)

    args = parser.parse_args()
    alpha: float = args.alpha
    n_samples: int = args.n_samples
    burnin: int = args.burnin
    seed(0)

    ## Homework 8
    # init mpl objs
    fig, ax = plt.subplots()

    # Reproduce contour plot
    n_points = 500
    x = linspace(0, 6, n_points)
    y = linspace(0, 5, n_points)
    X, Y = meshgrid(x, y)
    Z = hw8_pdf([X, Y])
    ax.contourf(X, Y, Z, levels=10) 

    # Run MCMC
    x0 = array([0., 0.])
    samples, acceptance_probabilities = multivariate_metropolis_hastings(
        x0, 
        n_samples + burnin,
        target_pdf=hw8_pdf, 
        alpha=alpha)

    ax.plot(
        samples[burnin:, 0], samples[burnin:, 1], 
        marker=".", markersize=2, linewidth=0.75, color="red", alpha=0.5) 

    ax.set_title("Mean Acceptance Probability:" 
        f" {mean(acceptance_probabilities)}")

    plt.show()

from typing import Callable, Dict, Optional, Tuple

from argparse import ArgumentParser, BooleanOptionalAction

import matplotlib.pyplot as plt

from numpy import (
    zeros, ndarray, exp, meshgrid, linspace, absolute, power, mean, array,
    diagonal, log
)

# 'as' naming follows R conventions
from numpy.random import ( 
    uniform as runif,  # sample random uniform
    normal as rnorm,   # sample random normal
    seed
)
from scipy.stats import norm 
dnorm = norm.pdf      # (probability) density of normal

from tqdm import tqdm


def metropolis_hastings(
    x0: ndarray, 
    n_samples: int,
    is_homework_8: bool,
    # proposal distribution args 
    proposal_distribution_issymmetric: bool,
    proposal_distribution_pdf: Optional[Callable] = None,
    proposal_distribution_pdf_kwargs: Optional[Dict] = None,
    proposal_distribution_random_sampler: Callable = rnorm,
    proposal_distribution_random_sampler_kwargs: Dict = dict(loc=0.0, scale=1.0),
    # target distribution args
    target_distribution_pdf: Callable = dnorm,
    target_distribution_pdf_kwargs: Dict = dict(loc=0.0, scale=1.0),
    # misc AND homework arg
    alpha: Optional[float] = None,
    verbose: bool = True) -> Tuple[ndarray, ndarray]:
    """Metropolis-Hastings (MH) for sampling from unknown target distribution.

    In the context of UQ, the target distribution is the posterior distribution
    `p(z|d)` where `p(.)` is a probability density function, `z` is a random 
    realization with prior `p(z)`, and `d` is the data with 
    evidence `p(d)`. In inverse UQ, we seek to sample from the 
    posterior distribution for which we only know how to calculate the 
    likelihood `L(z) = p(d|z)`. It is important to recognize that the forward 
    model `G: \mathbb{R}^{n_z} -> \mathbb{R}^{n_d}` essentially generates the
    the data `d + error` corresponding to `p(d|z)` for `n_z` random variables 
    and an output vector with `n_d` data elements. Since it is also known
    that `p(z|d) \propto f(z)` where `f(z) = p(d|z)*p(z)` from Bayes rule, and
    since the MH algorithm computes the density ratio as 
    `p(z|d = proposal)/p(z|d = prev_sample)`, then this means the density
    ratio is equivalent to `p(d=proposal|z)*p(z)/p(d=prev_sample|z)*p(z)`.
    In other words, the prior distribution `p(z)` cancels, and then we use
    a representation of `p(d|z)` given by a gPC approximation (eq. 8.17 Xiu)
    in order to sample the target (posterior) distribution `p(z|d)`. So with
    some modification, the current function could be used for inverse UQ 
    provided the `target_distribution_pdf` corresponds to a gPC approximation 
    of `p(d|z)`.
 
    Examples:
    ```
    # Using the default arguments of the function, the proposal distribution
    # is defined by the normal distribution, which is symmetric
    proposal_distribution_issymmetric = True
    is_homework_8 = False

    # These parameters reproduce the first row of fig. 1 from ref [2] 
    proposal_distribution_random_sampler_kwargs = dict(loc=0, scale=5)
    target_distribution_pdf_kwargs = dict(loc=100, scale=15)
    n_samples = 500
    x00 = array([150])
    samples0 = metropolis_hastings(
        x00, n_samples, proposal_distribution_issymmetric, is_homework_8,
        proposal_distribution_random_sampler_kwargs=proposal_distribution_random_sampler_kwargs,
        target_distribution_pdf_kwargs=target_distribution_pdf_kwargs,
        verbose = False)

    x01 = array([250])
    samples1 = metropolis_hastings(
        x01, n_samples, proposal_distribution_issymmetric, is_homework_8,
        proposal_distribution_random_sampler_kwargs=proposal_distribution_random_sampler_kwargs,
        target_distribution_pdf_kwargs=target_distribution_pdf_kwargs,
        verbose = False) 

    x02 = array([650])
    samples2 = metropolis_hastings(
        x02, n_samples, proposal_distribution_issymmetric, is_homework_8,
        proposal_distribution_random_sampler_kwargs=proposal_distribution_random_sampler_kwargs,
        target_distribution_pdf_kwargs=target_distribution_pdf_kwargs,
        verbose = False)

    # plot results
    fig, axs = plt.subplots(1, 3, figsize=(8, 6))
    iters = range(len(samples0))
    axs[0].plot(iters, samples0)
    axs[1].plot(iters, samples1)
    axs[2].plot(iters, samples2)
    
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Sampled Values")
    axs[1].set_xlabel("Iteration")
    axs[2].set_xlabel("Iteration")

    plt.show()
    ```

    Args:
        x0: Initial sample array. This is unlikely to come from the target 
            distribution, though over the course of the algorithm, samples
            will converge to being drawn from the target distribution.
        n_samples: Number of samples you wish to draw from the target 
            distribution.
        is_homework_8: True if performing MCMC for homework 8, False otherwise.
            If True, then `proposal_distribution_pdf_kwargs` are updated
            dynamically with the last value of the markov chain.

        proposal_distribution_issymmetric: True if proposal distribution
            is symmetric, false otherwise.
        proposal_distribution_pdf: PDF for posterior distribution. Required
            if `proposal_distribution_issymmetric=False`.
        proposal_distribution_pdf_kwargs: Kwargs for posterior distribution 
            probability density function. Required if 
            `proposal_distribution_issymmetric=False`.
        proposal_distribution_random_sampler: Function that samples from the
            desired proposal distribution.
        proposal_distribution_random_sampler_kwargs: Kwargs for randomly 
            sampling the proposal distribution.

        target_distribution_pdf: Probability density function of target 
            distribution.
        target_distribution_pdf_kwargs: Kwargs for target distribution.

        alpha: Covariance matrix alpha. Required if `if_homework_8=True`.
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
        
    # Force provision of proposal distribution args if nonsymmetric 
    if (not proposal_distribution_issymmetric) and \
        (proposal_distribution_pdf is None or \
        proposal_distribution_pdf_kwargs is None):
        raise ValueError(
            "Proposal distribution is NOT symmetric!"
            " `proposal_distribution_pdf` and `proposal_distribution_pdf_kwargs`"
            " must be provided!")

    # Force provision of alpha if homework8
    if is_homework_8 and alpha is None:
        raise ValueError(
            f"`alpha` must be provided since `is_homework_8={is_homework_8}`")

    # Assert kwargs are dictionaries 
    if not(isinstance(proposal_distribution_random_sampler_kwargs, dict) and \
        isinstance(target_distribution_pdf_kwargs, dict)):
        raise TypeError(
            "Need type `dict` for the following args, but got"
            " `proposal_distribution_random_sampler_kwargs="
            f"{type(proposal_distribution_random_sampler_kwargs)}`"
            " and `target_distribution_pdf_kwargs="
            f" {type(target_distribution_pdf_kwargs)}`.")

    if not proposal_distribution_issymmetric and \
        not isinstance(proposal_distribution_pdf_kwargs, dict):
        raise TypeError(
            "`proposal_distribution_pdf_kwargs` must be of type `dict`" 
            f" but got `{type(proposal_distribution_pdf_kwargs)}`.") 
    
    # Create covariance matrix
    K_q = array([[alpha, 0], [0, alpha]])

    # Initialize samples array with guess
    samples = zeros(shape=(n_samples+1, len(x0)))
    samples[0] = x0

    # For tracking the acceptance probabilties (for hw8) corresponding to 
    # ratio of proposal and previous sample densities
    acceptance_probabilities = [] 

    # Perform MCMC sampling w/ metropolis hastings algorithm
    for i in tqdm(
        range(1, n_samples+1), 
        desc="Metropolis-Hastings MCMC", 
        disable=not verbose):

        prev_sample = samples[i-1]

        # Compute noise from proposal distribution
        if not is_homework_8:
            noise = proposal_distribution_random_sampler(
                **proposal_distribution_random_sampler_kwargs)
        else:
            # returns a matrix for which only the diagonals are nonzero
            noise = rnorm(loc=prev_sample, scale=K_q)
            noise = diagonal(noise)

            ##TODO: create the proposal distribution pdf and its kwargs
            #proposal_distribution_pdf_kwargs = dict(loc=prev_sample, scale=K_q)
            #proposal_distribution_pdf = dnorm
 
        # Compute proposal
        proposal = prev_sample + noise

        # Determine if scaling of densities is needed due to nonsymmetric 
        # proposal pdf (i.e., transition kernel, see slide 31 UQ lecture 9)
        if not proposal_distribution_issymmetric: #or is_homework_8:
            q_proposal = proposal_distribution_pdf(
                proposal, **proposal_distribution_pdf_kwargs)
            q_prev_sample = proposal_distribution_pdf(
                prev_sample, **proposal_distribution_pdf_kwargs)
        else:
            q_proposal = 1
            q_prev_sample = 1
        
        # Compute the ratio of the "height" of the proposal from the 
        # posterior distribution pdf to the "height" of the prev sample
        # on the same pdf
        density_proposal = target_distribution_pdf(
            proposal, **target_distribution_pdf_kwargs)/q_proposal
        density_prev_sample = target_distribution_pdf(
            prev_sample, **target_distribution_pdf_kwargs)/q_prev_sample
        density_ratio = density_proposal/density_prev_sample 

        # compute acceptance probability
        acceptance_probability = accept(density_ratio)

        # Update acceptance probability list
        acceptance_probabilities.append(acceptance_probability)
    
        # Determine accept/rejection of proposed sample (see ref [2])
        if acceptance_probability > runif():
            samples[i] = proposal
        else:
            samples[i] = prev_sample
 
    return samples, array(acceptance_probabilities)


def V(x, y):
    return power((x - y - 1), 4) + absolute(x + y - 6)


def hw8_pdf(u: ndarray, **kwargs):
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
        "--burnout", 
        help="number of initial samples to discard from result of MCMC sampling"
        " (default: 0).", 
        default=0, type=int)

    args = parser.parse_args()
    alpha: float = args.alpha
    n_samples: int = args.n_samples
    burnout: int = args.burnout
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
    ax.contourf(X, Y, Z) 

    # Run MCMC
    x0 = array([0., 0.])
    samples, acceptance_probabilities = metropolis_hastings(
        x0, 
        n_samples, 
        is_homework_8=True, 
        proposal_distribution_issymmetric=True,
        target_distribution_pdf=hw8_pdf,
        alpha=alpha)

    print(samples.shape)
    ax.plot(samples[burnout:, 0], samples[burnout:, 1], "-o") 

    ax.set_title("Mean Acceptance Probability:" 
        f" {mean(acceptance_probabilities)}")

    # ax.set_xlim(1, 6)
    # ax.set_ylim(0, 5)

    plt.show()

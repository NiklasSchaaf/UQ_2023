"""Didactic metropolis-hastings for univariate distributions. Not used in HW."""


from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt

from numpy import zeros, ndarray

# 'as' naming follows R conventions
from numpy.random import ( 
    uniform as runif, # sample random uniform
    normal as rnorm   # sample random normal
)
from scipy.stats import norm 
dnorm = norm.pdf      # (probability) density of normal

from tqdm import tqdm


def metropolis_hastings(
    x0, 
    n_samples: int,
    # proposal distribution args 
    proposal_distribution_issymmetric: bool,
    proposal_distribution_pdf: Optional[Callable] = None,
    proposal_distribution_pdf_kwargs: Optional[Dict] = None,
    proposal_distribution_random_sampler: Callable = rnorm,
    proposal_distribution_random_sampler_kwargs: Dict = dict(loc=0.0, scale=1.0),
    # target distribution args
    target_distribution_pdf: Callable = dnorm,
    target_distribution_pdf_kwargs: Dict = dict(loc=0.0, scale=1.0),
    # misc
    verbose: bool = True):
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

    # These parameters reproduce the first row of fig. 1 from ref [2] 
    proposal_distribution_random_sampler_kwargs = dict(loc=0, scale=5)
    target_distribution_pdf_kwargs = dict(loc=100, scale=15)
    n_samples = 500
    x00 = 150
    samples0 = metropolis_hastings(
        x00, n_samples, proposal_distribution_issymmetric,
        proposal_distribution_random_sampler_kwargs=proposal_distribution_random_sampler_kwargs,
        target_distribution_pdf_kwargs=target_distribution_pdf_kwargs,
        verbose = False)

    x01 = 250
    samples1 = metropolis_hastings(
        x01, n_samples, proposal_distribution_issymmetric,
        proposal_distribution_random_sampler_kwargs=proposal_distribution_random_sampler_kwargs,
        target_distribution_pdf_kwargs=target_distribution_pdf_kwargs,
        verbose = False) 

    x02 = 650
    samples2 = metropolis_hastings(
        x02, n_samples, proposal_distribution_issymmetric,
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
        x0: Initial sample. This is unlikely to come from the target 
            distribution, though over the course of the algorithm, samples
            will (hopefully) converge to being drawn from the target 
            distribution.
        n_samples: Number of samples you wish to draw from the target 
            distribution.

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

        verbose: True to print tqdm progress, false otherwise.

    Returns:
        An array of samples drawn from the target distribution.

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
    # Force provision of proposal distribution args if nonsymmetric 
    if (not proposal_distribution_issymmetric) and \
        (proposal_distribution_pdf is None or \
        proposal_distribution_pdf_kwargs is None):
        raise ValueError(
            "Proposal distribution is NOT symmetric!"
            " `proposal_distribution_pdf` and `proposal_distribution_pdf_kwargs`"
            " must be provided!")

    # Initialize samples array with guess
    samples = zeros(n_samples+1) 
    samples[0] = x0

    # Perform MCMC sampling w/ metropolis hastings algorithm
    for i in tqdm(
        range(1, n_samples+1), 
        desc="Metropolis-Hastings MCMC", 
        disable=not verbose):

        prev_sample = samples[i-1]

        # Propose sample
        noise = proposal_distribution_random_sampler(
            **proposal_distribution_random_sampler_kwargs)
        proposal = prev_sample + noise

        # Determine if scaling of densities is needed due to nonsymmetric 
        # proposal function 
        # NOTE: The notation from ref [1] uses J to denote
        # the transition kernel while other sources denote it as Q or q
        if not proposal_distribution_issymmetric:
            J_proposal = proposal_distribution_pdf(
                proposal, **proposal_distribution_pdf_kwargs)
            J_prev_sample = proposal_distribution_pdf(
                prev_sample, **proposal_distribution_pdf_kwargs)
        else:
            J_proposal = 1
            J_prev_sample = 1
        
        # Compute the ratio of the "height" of the proposal from the 
        # posterior distribution pdf to the "height" of the prev sample
        # on the same pdf
        density_proposal = target_distribution_pdf(
            proposal, **target_distribution_pdf_kwargs)/J_proposal
        density_prev_sample = target_distribution_pdf(
            prev_sample, **target_distribution_pdf_kwargs)/J_prev_sample
        density_ratio = density_proposal/density_prev_sample 
    
        # Determine accept/rejection of proposed sample
        # it seems like this should use min(1, ...) according to ref [5],
        # but i've seen now two other sources that use random uniform
        if density_ratio > runif():
            samples[i] = proposal
        else:
            samples[i] = prev_sample
 
    return samples 


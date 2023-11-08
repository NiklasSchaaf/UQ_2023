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
    proposal_distribution_kwargs: Optional[Dict] = None,
    proposal_distribution_random_sampler: Callable = rnorm,
    proposal_distribution_random_sampler_kwargs: Dict = dict(loc=0.0, scale=1.0),
    # posterior distribution args
    posterior_distribution_pdf: Callable = dnorm,
    posterior_distribution_kwargs: Dict = dict(loc=0.0, scale=1.0),
    # misc
    verbose: bool = True):
    """Metropolis-Hastings algorithm to sample from posterior distribution.

    NOTE: The `posterior_distribution_pdf` in the context of UQ will be a
    gPC approximation.

    One method in a set of methods known as Markov Chain Monte-Carlo for 
    sampling from a distribution even when all that is known about the 
    distribution is how to calculate the (probability) density for different 
    samples.

    The target distribution is the distribution `Z` for which we only know
    how to calculate its likelihood `L(z) = p(d | z)` where `p` is a
    probability density function, `z` is a realization of `Z`, and `d` is some 
    "evidence" (aka, data). According to ref [2], the target distribution
    is the posterior distribution.

    Examples:
    ```
    # Using the default arguments of the function, the proposal distribution
    # is defined by the normal distribution, which is symmetric
    proposal_distribution_issymmetric = True

    # These parameters reproduce the first row of fig. 1 from ref [2] 
    proposal_distribution_random_sampler_kwargs = dict(loc=0, scale=5)
    posterior_distribution_kwargs = dict(loc=100, scale=15)
    n_samples = 500
    x00 = 150
    samples0 = metropolis_hastings(
        x00, n_samples, proposal_distribution_issymmetric,
        proposal_distribution_random_sampler_kwargs=proposal_distribution_random_sampler_kwargs,
        posterior_distribution_kwargs=posterior_distribution_kwargs,
        verbose = False)

    x01 = 250
    samples1 = metropolis_hastings(
        x01, n_samples, proposal_distribution_issymmetric,
        proposal_distribution_random_sampler_kwargs=proposal_distribution_random_sampler_kwargs,
        posterior_distribution_kwargs=posterior_distribution_kwargs,
        verbose = False) 

    x02 = 650
    samples2 = metropolis_hastings(
        x02, n_samples, proposal_distribution_issymmetric,
        proposal_distribution_random_sampler_kwargs=proposal_distribution_random_sampler_kwargs,
        posterior_distribution_kwargs=posterior_distribution_kwargs,
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
        proposal_distribution_kwargs: Kwargs for posterior distribution 
            probability density function. Required if 
            `proposal_distribution_issymmetric=False`.
        proposal_distribution_random_sampler: Function that samples from the
            desired proposal distribution.
        proposal_distribution_random_sampler_kwargs: Kwargs for randomly 
            sampling the proposal distribution.

        posterior_distribution_pdf: Probability density function of posterior 
            distribution
        posterior_distribution_kwargs: Kwargs for posterior distribution.

        verbose: True to print tqdm progress, false otherwise.

    Returns:
        An array of samples (approximately) drawn from the target distribution.

    References:
        [1] : Smith, R.C. (2014). Chapter 8.3.1 and 8.3.3 in "Uncertainty 
            Quantification: Theory, Implementation, and Applications." SIAM
        [2] : Ravenzwaaij, D.v. et al. A simple introduction to Markov 
            Chain Monte–Carlo sampling. Psychon Bull Rev (2018) 25:143-154.
        [3] : Xiu, D. (2010). Chapter 8.2 in "Numerical Methods for Stochastic 
            Computations: A Spectral Method Approach". Princeton University 
            Press.
    """
    # Force provision of proposal distribution args if nonsymmetric 
    if (not proposal_distribution_issymmetric) and \
        (proposal_distribution_pdf is None or \
        proposal_distribution_kwargs is None):
        raise ValueError(
            "Proposal distribution is NOT symmetric!"
            " `proposal_distribution_pdf` and `proposal_distribution_kwargs`"
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
        proposal = prev_sample  + noise

        # Determine if scaling of densities is needed due to nonsymmetric 
        # proposal function
        if not proposal_distribution_issymmetric:
            J_proposal = proposal_distribution_pdf(
                proposal, **proposal_distribution_kwargs)
            J_prev_sample = proposal_distribution_pdf(
                prev_sample, **proposal_distribution_kwargs)
        else:
            J_proposal = 1
            J_prev_sample = 1
        
        # Compute the ratio of the "height" of the proposal from the 
        # posterior distribution pdf to the "height" of the prev sample
        # on the same pdf
        density_proposal = posterior_distribution_pdf(
            proposal, **posterior_distribution_kwargs)/J_proposal
        density_prev_sample = posterior_distribution_pdf(
            prev_sample, **posterior_distribution_kwargs)/J_proposal
        density_ratio = density_proposal/density_prev_sample 
    
        # Determine accept/rejection of proposed sample
        if density_ratio > runif():
            samples[i] = proposal
        else:
            samples[i] = prev_sample
 
    return samples 


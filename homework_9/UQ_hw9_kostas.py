import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm


def func(cand):
    """
     Function to be approximated
    :param cand:
    :return:
    """
    alpha = cand[0]
    beta = cand[1]
    return 1.5 * alpha + np.square(beta - 1) / 4 + np.cos(np.pi + alpha + beta)


# Data observations
y_i = np.array([4.02, 3.97, 4.05, 3.85, 3.94])


def MCMC(param_covariance, observations, init, steps, tqdm_bool=False):
    """
    Function that implements the MCMC Metropolis-Hastings algorithm, in order to calibrate the alpha and beta values.

    The proposal distribution is assumed to be a Gaussian distribution, in order for q(w,u) to be summetric (convenience)
    As a result, q(w, u)/q(u, w) = 1 and we do not have to include it into the calculation of the acceptance probability.
    So we sample each new state from a Gaussian distribution centered at the previous state N(u_n-1, alpha*K_q).
    That "alpha" value, is tuned.
    :param param_covariance: The covariance parameter, this is to be tuned
    :param observations: The data observations
    :param init: Initial state of the Markov chain
    :param steps: steps of the Markov chain
    :param tqdm_bool: visualise the tqdm loading bar
    :return:
    """
    dims = len(init)
    mean = np.array([0, 4])
    # variance of the proposal distribution, to be tuned
    K_q = param_covariance**2 * np.eye(dims)
    acceptance_probs_dict = {}
    states = np.zeros((steps, dims))
    accepted = 0

    if tqdm_bool:
        pbar = tqdm(total=steps)
    for i in range(1, steps):
        # sample new candidate state
        cand = np.random.multivariate_normal(mean=states[i - 1], cov=K_q)

        # "Analytically" compute the r(u,w) ratio. "Analytically" in the sense that I do not use any libraries
        # to sample from the multivariate Gaussian distributions for the likelihood and prior.
        # acceptance_probability = min(1, r(u,w))

        # Construct the p(w)/p(u)
        prior_p_cand_p_prev = np.exp((np.linalg.norm(states[i - 1] - mean) ** 2 -
                                      np.linalg.norm(cand - mean) ** 2) / (2 * 0.25 ** 2))

        # Construct the p(d|u_n)/p(d|u_n-1)
        sub_sum_prev = np.sum([np.sum(np.linalg.norm(obs - func(states[i - 1])) ** 2) for obs in observations])
        sub_sum_cand = np.sum([np.sum(np.linalg.norm(obs - func(cand)) ** 2) for obs in observations])
        likelihood_p_cand_p_prev = np.exp((sub_sum_prev - sub_sum_cand) / (2 * 0.1 ** 2))

        # multiply these two fractions, to obtain r(u_n-1, u_n)
        ratio = prior_p_cand_p_prev * likelihood_p_cand_p_prev

        acceptance_prob = min(1, ratio)
        acceptance_probs_dict[i] = acceptance_prob

        random_threshold = np.random.uniform(0, 1)
        # calculate the acceptance rate
        if acceptance_prob >= random_threshold:
            states[i] = cand
            accepted += 1
        else:
            states[i] = states[i - 1]
            if tqdm_bool:
                pbar.update()
    if tqdm_bool:
        pbar.close()

    acceptance_rate = accepted / steps
    return states, acceptance_probs_dict, acceptance_rate


def visualise_samples(steps, states, yi):
    x = np.arange(0, steps)
    plt.plot(x, states[:, 0])
    plt.xlabel("MCMC steps")
    plt.title(r"MCMC samples of the $\alpha$ value")
    plt.ylabel(r"$\alpha$")
    plt.show()

    plt.plot(x, states[:, 1])
    plt.xlabel("MCMC steps")
    plt.ylabel(r"$\beta$")
    plt.title(r"MCMC samples of the $beta$ value")
    plt.show()

    func_evaluated = [func(state) for state in states]
    print(len(func_evaluated))
    plt.plot(x, func_evaluated)
    plt.title("Model output")
    plt.xlabel("MCMC steps")
    plt.ylabel("model output")
    plt.show()

    plt.plot(states[:, 0], states[:, 1], '-o')
    plt.title(rf"MCMC {len(states)} samples for $\alpha$ & $\beta$")
    plt.xlabel(r"$\alpha$ values")
    plt.ylabel(r"$\beta$ values")
    plt.show()


def manual_tuning_covar(steps):
    """
    Function used for the manual tuning of the aforementioned "alpha" parameter
    :param steps: steps of the MCMC
    :return:
    """
    # Try many different alphas.
    # We run the simulation for each alpha only once, and not multiple times.
    alphas = np.arange(0, 0.2, 0.01)
    alphas_dict = {}
    pbar = tqdm(total=len(alphas))
    for alpha in alphas:
        states, probs, acc = MCMC(alpha, y_i, np.array([0, 0]), steps=steps)
        alphas_dict[alpha] = acc
        pbar.update()
    pbar.close()

    x_acc = list(alphas_dict.keys())
    y_acc = list(alphas_dict.values())

    plt.plot(x_acc, y_acc, '-o')
    plt.axhline(0.3, color='black', linestyle="--", linewidth=0.5, label='rate = 0.3')
    plt.axhline(0.25, color='black', linestyle="--", linewidth=0.5, label='rate = 0.25')
    plt.legend(loc='best')
    plt.title(f"Manual Tuning of the alpha value")
    plt.xlabel("Different 'alpha' values")
    plt.ylabel("Acceptance rate")
    plt.show()

    # Find the value of alpha that is closest to 0.25. We aim for 0.25 acceptance rate
    index_closest = min(range(len(y_acc)), key=lambda i: abs(y_acc[i] - 0.25))
    print(f" The covariance closest to 0.25 is {y_acc[index_closest]}")
    print(f" This corresponds to a value of 'alpha' = {x_acc[index_closest]}")

    alpha_for_tuned_covar = x_acc[index_closest] # the "alpha" value needed,
    # to achieve the covariance that corresponds to acceptance rate 25%

    # the covariance that corresponds to acceptance rate 25%
    tuned_covar = y_acc[index_closest]

    return alpha_for_tuned_covar, tuned_covar


def visualise_marginals(states):
    """
    Function that visualises the marginal posterior distributions of the alpha and beta parameters.
    A kernel density estimation of the distributions is also provided
    Also, the estimated covariance matrix of the posterior distributions of alpha and beta is calculated.
    :param states: the sampled states of the MCMC
    :return:
    """
    for i in range(len(states[0])):
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        kde = sm.nonparametric.KDEUnivariate(states[2000:, i])
        kde.fit()  # Estimate the densities
        # Plot the histogram
        ax.hist(
            states[2000:, i],
            bins=30,
            density=True,
            label="Marginal distributions",
            zorder=5,
            edgecolor="k",
            alpha=0.5,
        )

        # Plot the KDE as fitted using the default arguments
        ax.plot(kde.support, kde.density, lw=3, label="KDE from samples", zorder=10)

        # Plot the samples
        ax.scatter(
            states[2000:, i],
            np.abs(np.random.randn(states[2000:, i].size)) / 40,
            marker="x",
            color="red",
            zorder=20,
            label="MCMC samples",
            alpha=0.5,
        )

        ax.legend(loc="best")
        ax.grid(True, zorder=-5)
        ax.set_ylabel("Frequency")
        if i == 0:
            ax.set_xlabel(r"$\alpha$ value")
            ax.set_title(r"Marginal distribution of posterior for $\alpha$")
        else:
            ax.set_xlabel(r"$\beta$ value")
            ax.set_title(r"Marginal distribution of posterior for $\beta$")
        plt.show()

    estimated_covariance = np.cov(states[2000:].T)
    print(f"estimated covariance is {estimated_covariance}")

    # I also added the code of the other implementation, so that we can compare the marginal distributions
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(states[2000:, 0], bins=30)
    axs[0].set_xlabel(rf'$\alpha$, N={len(states)}')
    axs[0].set_ylabel('frequency')

    axs[1].hist(states[2000:, 1], bins=30)
    axs[1].set_xlabel(rf'$\beta$, N={len(states)}')
    axs[1].set_ylabel('frequency')
    plt.tight_layout()
    plt.show()


steps = 100000
# First, tune the "alpha" value, aiming at an acceptance rate of 0.25
alpha, cov = manual_tuning_covar(steps=steps)
states, prob, acc = MCMC(alpha, y_i, np.array([0, 0]), steps=steps, tqdm_bool=True)
print(f"acceptance rate is {acc}")

visualise_samples(steps=steps, states=states, yi=y_i)
visualise_marginals(states=states)

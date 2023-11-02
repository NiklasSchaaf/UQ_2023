# this script integrates the SEIR model 20 times
# with newly sampled random parameter for each integration

import os 
import sys

import matplotlib.pyplot as plt

from numpy import amax as max
from numpy.random import beta as betarnd

# explicitly add project root dir to path to fix import issue
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from homework_7.SEIRmodel import SEIRmodel

if __name__ == "__main__":
    R0 = 2.2
    T = 9
    infected_state_ix = 2
    betarnd_a = betarnd_b = 2
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    for n in range(1, 20 + 1):
        tau = 1.0 + 13*betarnd(betarnd_a, betarnd_b)

        XOutput = SEIRmodel(R0, T, tau)
        infected = XOutput[:, infected_state_ix]
        Q = max(infected)

        axs[0].plot(infected, "-r")
        axs[1].scatter(tau, Q, s=80, facecolors='none', edgecolors='r')

    axs[0].set_xlabel("timesteps")
    axs[0].set_ylabel("I")
    axs[1].set_xlabel(r"$\tau$")
    axs[1].set_ylabel("Q")

    plt.show()



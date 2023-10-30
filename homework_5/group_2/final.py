# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:00:15 2023

@author: arong
"""
import numpy as np
from scipy.stats import norm
from math import factorial
import matplotlib.pyplot as plt
from util import *
import time

K = 4
num_samples = 100

hermite_array_ijk = np.zeros((K, K, K))

for i in range(K):
    for j in range(K):
        for k in range(K):

            hermite_array_ijk[i, j, k] = hermite_expectation(i, j, k)

# Given parameters and initial conditions
total_time = 2.0
delta_t = 0.1
N = K - 1  # We've computed e_ijk up to N = K - 1
time_steps = int(total_time / delta_t)

# Initial conditions
v_hat = np.zeros((N+1))
v_hat[0] = 1/10

# alpha_hat values
alpha_hat = np.zeros((N+1))
alpha_hat[0] = 1
alpha_hat[1] = 1

# 2D matrix to store all time points
v_hat_all_time = np.zeros((time_steps+1, N+1))

# Store initial conditions in the matrix
v_hat_all_time[0] = v_hat

start_time = time.time()

# Forward Euler time-stepping
for n in range(time_steps):
    v_hat_new = np.zeros((N+1))
    for k in range(N+1):
        sum_ij = 0
        for i in range(N+1):
            for j in range(N+1):
                sum_ij += (-0.5 * v_hat[i] * v_hat[j] + alpha_hat[j] * v_hat[i]) * hermite_array_ijk[i,j, k]
        v_hat_new[k] += v_hat[k] + delta_t * sum_ij / factorial(k)
    v_hat = v_hat_new
    
    # Store the new v_hat values in the matrix
    v_hat_all_time[n+1] = v_hat

# Compute mean and variance using given properties
mean_u_over_time = v_hat_all_time[:, 0]

variance_u_over_time = np.sum(v_hat_all_time**2 * np.array([factorial(k) for k in range(N+1)]), axis=1) - mean_u_over_time**2
std_u_over_time = np.sqrt(variance_u_over_time)

end_time = time.time()
print(f"Runtime of Stochastic Galerkin of order {N}: {(end_time - start_time) * 1000:.4f} milliseconds")

start_time = time.time()
# Parameters

delta_t = 0.1
total_time = 2.0

# Calculate Monte Carlo results
mc_means, mc_variances = monte_carlo_over_time(num_samples, delta_t, total_time)
mc_stds = np.sqrt(mc_variances)

end_time = time.time()
print(f"Runtime of Monte Carlo with {num_samples} samples:  {(end_time - start_time) * 1000:.4f} milliseconds")


# Plotting standards
axis_label_font_size = 24
axis_ticks_font_size = 22
legend_font_size = 22
title_font_size = 26
scale_font_size = 20


# Plotting
plt.figure(figsize=(10, 6))
time_values = np.arange(0, total_time+delta_t, delta_t)

# gPC results
plt.plot(time_values, mean_u_over_time, label='gPC Mean of u(t)', color='blue')
plt.fill_between(time_values, 
                 mean_u_over_time - std_u_over_time, 
                 mean_u_over_time + std_u_over_time, 
                 color='gray', alpha=0.5, label='gPC 1 std deviation')

# Monte Carlo results
plt.plot(time_values, mc_means, label='Monte Carlo Mean of u(t)', color='red', linestyle='--')
plt.fill_between(time_values, 
                 np.array(mc_means) - np.array(mc_stds), 
                 np.array(mc_means) + np.array(mc_stds), 
                 color='pink', alpha=0.5, label='Monte Carlo 1 std deviation')

plt.title('gPC vs MC reconstruction of u(t) over time', fontsize=title_font_size)
plt.xlabel('Time', fontsize=axis_label_font_size)
plt.ylabel('u(t)', fontsize=axis_label_font_size)
plt.legend(fontsize=legend_font_size, loc='upper left')
plt.grid(True)
plt.xticks(fontsize=axis_ticks_font_size)
plt.yticks(fontsize=axis_ticks_font_size)
plt.tight_layout()
plt.savefig(f'u(t)_gpc{N}_mc{num_samples}.pdf', dpi=400, bbox_inches='tight')
plt.show()


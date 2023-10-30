# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:34:18 2023

@author: arong
"""

import numpy as np
from scipy.stats import norm
from math import factorial

def du_dt(u, alpha_Z):
    return -0.5 * u**2 + alpha_Z * u

def euler_method(u0, alpha_Z, dt, T):
    u = u0
    t = 0
    while t < T:
        u += du_dt(u, alpha_Z) * dt
        t += dt
    return u

def monte_carlo_integration(N, dt, T):
    u_values = []
    for _ in range(N):
        Z = np.random.normal(0, 1)
        alpha_Z = 1 + Z
        u_values.append(euler_method(1/10, alpha_Z, dt, T))
    
    mean = np.mean(u_values)
    variance = np.var(u_values, ddof=1)
    
    return mean, variance

# Monte Carlo over time
def monte_carlo_over_time(num_samples, dt, T):
    means = []
    variances = []
    
    total_time_points = int(T/dt) + 1
    for t_idx in range(total_time_points):
        mean, variance = monte_carlo_integration(num_samples, dt, t_idx*dt)
        means.append(mean)
        variances.append(variance)
    
    return means, variances

def monte_carlo_with_CI_collect_data(initial_samples, increment_samples, dt, T, accuracy_threshold=0.01, confidence_level=0.95):
    sample_sizes = []
    means = []
    std_devs = []
    ci_lowers = []
    ci_uppers = []
    ci_width = 1000
    
    N = initial_samples
    z_value = norm.ppf((1 + confidence_level) / 2)  # Z-value for 95% confidence
    
    while ci_width > accuracy_threshold:
        mean, std_dev = monte_carlo_integration(N, dt, T)
        ci_width = z_value * (std_dev / np.sqrt(N))
        ci_lower = mean - ci_width/2
        ci_upper = mean + ci_width/2
        
        # Save data
        sample_sizes.append(N)
        means.append(mean)
        std_devs.append(std_dev)
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
        
        N = int(N*1.3)
    
    return np.array(sample_sizes), np.array(means), np.array(std_devs), np.array(ci_lowers), np.array(ci_uppers)

def hermite_expectation(n, m, l):
    s = (n + m + l) // 2
    if n + m + l == 2 * s and s >= n and s >= m and s >= l:
        return (factorial(n) * factorial(m) * factorial(l)) / (factorial(s - n) * factorial(s - m) * factorial(s - l))
    return 0
            
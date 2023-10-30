import numpy as np
import matplotlib.pyplot as plt
import sys
import os.path 
# explicitly add project root dir to path to fix import issue
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from homework_6 import heateqn


# At first, we calculate the Clenshaw - Curtis nodes for the interpolation
def ClenshawCurtisNodes(M):
    nodes = []
    for i in range(M):
        x = np.cos((i * np.pi) / M)
        nodes.append(x)
    return nodes


# Next, we construct the Lagrange basis functions
def lagrange_basis(x, data_points, j):
    """
    Calculate the Lagrange basis function for the j-th data point.

    Inputs:
    x: The point at which we evaluate the basis function.
    data_points (array-like): The list of data points (x values).
    j: The index of the data point for which we calculate the basis function.

    Output:
    basis: The value of the j-th Lagrange basis function at point x.
    """
    n = len(data_points)
    basis = 1.0
    for i in range(n):
        if j != i:
            basis *= (x - data_points[i]) / (data_points[j] - data_points[i])

    return basis

def to_nonstandard_uniform(val, a, b):
    """Map value on interval [-1, 1] to [a, b] where [a, b] is from U[a, b]."""
    return ((b-a)/2)*val + (a + b)/2


def CollocationApproximation(Z, zetas, pde_eval_dict, x_vals):
    '''
    Calculates the approximation for a given Z

    Arguments:
        Z -- random variable
        zetas -- values of the nodes
        pde_eval_dict -- dictionary with the evaluation of the PDE at the given nodes
        x_vals -- x values to be evaluated

    Returns:
        approximation, x values and the node values
    '''
    lagrange_basis_dict = {}
    # evaluate the PDE for all the zeta values
    for k, val in enumerate(zetas):
        # calculate the Lagrange basis functions
        basis = lagrange_basis(Z, zetas, k)
        lagrange_basis_dict[val] = basis

    u_approx = np.zeros(len(x_vals))
    for val in zetas:
        u_approx += lagrange_basis_dict[val]*pde_eval_dict[val]

    return u_approx, x_vals, zetas

def approx_M(M, zeta_vals):
    '''
    Generates a PDE approximation with M nodes for each value zeta_value

    Arguments:
        M -- number of nodes
        zeta_vals -- array of zeta values

    Returns:
        the approximation and x values
    '''
    nodes = ClenshawCurtisNodes(M)
    # get the transformed set of nodes Z
    a = 2 # distribution parameters from problem statement
    b = 16
    zetas = [to_nonstandard_uniform(val, a, b) for val in nodes]

    # 
    pde_eval_dict = {}
    for k, val in enumerate(zetas):
        x_vals, pde_sol = heateqn.heat_eq(val)
        pde_eval_dict[val] = pde_sol

    realisations = []
    for val in zeta_vals:
        approx, _, _ = CollocationApproximation(val, zetas, pde_eval_dict, x_vals)
        realisations.append(approx)

    return x_vals, realisations

### evaluate the error of approximation against true solution for given Z
Ms = np.arange(2,40)
Z = 2

x, solution = heateqn.heat_eq(Z)

errors = []
for M in Ms:
    _, realization = approx_M(M, np.array([Z]))
    errors.append(np.sqrt(np.mean((realization-solution)**2)))

Z2 = 9
x, solution2 = heateqn.heat_eq(Z2)
errors2 = []
for M in Ms:
    _, realization = approx_M(M, np.array([Z2]))
    errors2.append(np.sqrt(np.mean((realization-solution2)**2)))

# plot
plt.plot(Ms, errors, label = 'Z = '+str(Z))
plt.plot(Ms, errors2, label = 'Z = '+str(Z2))

plt.xlabel('number of nodes')
plt.ylabel('error')
plt.yscale('log')
plt.legend()
plt.show()

### generate realizations of the approximation.
M = 30
zetas_vals = np.random.uniform(2, 16, 200)
_, realisations = approx_M(30, zetas_vals)

for realisation in realisations:
    plt.plot(x, realisation, linewidth = 0.7, color='red')

mean_approx = np.mean(realisations, axis=0)
stdev_approx = np.std(realisations, axis = 0)
percentiles_5 = np.percentile(realisations, 5, axis=0)
percentiles_95 = np.percentile(realisations, 95, axis=0)
plt.xlabel("x")
plt.plot(x, mean_approx, linewidth = 2.5, color='black', label = "mean")
plt.fill_between(x, percentiles_5, percentiles_95, color='gray', alpha=0.4, label='5th-95th Percentiles')
plt.legend()
plt.ylabel("u(x, Z)")
plt.title("Approximation of u(x) with Stochastic Collocation")
plt.show()


# Find the index in the 'x' array that is closest to x = 0.7
x_target = 0.7
index_x_07 = np.abs(x - x_target).argmin()

# Access the values of 'u' at x = 0.7 for all realizations
u_vals_07 = [realisation[index_x_07] for realisation in realisations]

plt.hist(u_vals_07, bins=25)
plt.title('Histogram of the approximated u(x=0.7, Z)')
plt.show()


realisations_exact = []
for i in range(200):
    z = np.random.uniform(2, 16)
    res = heateqn.heat_eq(z)
    realisations_exact.append(res[1])

u_vals_07_exact = [realisation[index_x_07] for realisation in realisations_exact]

plt.hist(u_vals_07_exact, bins=25)
plt.title('Histogram of the exact u(x=0.7, Z)')
plt.show()







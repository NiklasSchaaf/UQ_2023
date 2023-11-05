"""Functions for performing multidimensional stochastic collocation

Usage:
```shell
# To see what the output of stochastic collocation looks like, run the below
python homework_7/multidim_stochastic_collocation 
```
"""


from typing import List, Union, Callable, Tuple

import itertools 

import os
import sys

import matplotlib.pyplot as plt

from numpy import ndarray, cos, pi, zeros, array, zeros_like, allclose
from numpy.random import uniform, beta

from tqdm import tqdm

# explicitly add project root dir to path to fix import issue
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from homework_7.SEIRmodel import SEIRmodel


def get_clenshawcurtis_collocation_nodes_matrix(k: int, d_dims: int) -> ndarray:
    """Matrix where rows are collocation nodes for the i^th random variable.

    Example:

    ```
    # Reproduce figure 7.1 from Xiu
    # NOTE: k = 5 in figure, but to get 1089 points, k = 6 in this example
    # ... i don't know why this is 
    import multidim_stochastic_collocation as msc
    import matplotlib.pyplot as plt 
    k = 6; d = 2;
    collocation_nodes_matrix = msc.get_clenshawcurtis_collocation_nodes_matrix(k, d)
    tensor_grid = msc.get_tensor_grid(collocation_nodes_matrix)
    plt.scatter(tensor_grid[0, :], tensor_grid[1, :], s = 1)
    plt.show()
    ```

    Args:
        k: Level of Clenshaw-Curtis grid.
        d_dims: Number of random variables in problem of interest.

    Returns:
       Clenshaw-Curtis collocation nodes matrix of shape `[d_dims, m_ik]` where 
        `m_ik = 2**(k-1) + 1`. Semantically, each random variable has
        a vector of collocation nodes associated with it. This corresponds
        to `collocation_nodes_matrix[random_variable_ix, :]`
 
    References:
        Xiu ch. 7.2.1 and 7.2.2.
    """
    assert k >= 1, "level of Clenshaw-Curtis grid must be >= 1"
    # The below algorithm is directly from Xiu equation 7.10 and desc. Xiu p. 82
    # and builds the collocation nodes matrix 
    m_ik = 2**(k-1) + 1 if k > 1 else 1 # num collocation nodes per dim
    collocation_nodes_matrix = zeros(shape=(d_dims, m_ik))
    for i in range(1, d_dims+1):
        for j in range(1, m_ik+1):
            if k > 1:
                Z_ij = nested_clenshaw_curtis_node(j, m_ik)
            elif k == 1 and j == 1:
                Z_ij = 0
            
            collocation_nodes_matrix[i-1, j-1] = Z_ij

    return collocation_nodes_matrix


def get_tensor_grid(collocation_nodes_matrix: ndarray) -> ndarray: 
    """Return dense tensor grid by cartesian product of collocation matrix rows.

    NOTE: Only used as a sanity check for the collocation nodes function.

    Args:
        collocation_nodes_matrix: Ndarray in `[d_dims, m_ik]`.

    Returns:
        Tensor grid in `[d_dims, m_ik**d_dims]`.

    References:
        See Xiu figure 7.1.
    """
    d_dims = collocation_nodes_matrix.shape[0]

    tensor_grid = array(
        tuple(
            itertools.product(
                *[collocation_nodes_matrix[i, :] for i in range(d_dims)]))).T

    return tensor_grid

 
def nested_clenshaw_curtis_node(j, m_ik):
    """Extrema of Chebyshev polynomial.
    
    References:
        Xiu eq. 7.10
    """
    return -cos((pi*(j - 1))/(m_ik - 1))


def map_uniform_val_to_new_interval(val, a, b):
    """Map value on interval [-1, 1] to [a, b].

    References:
        Ch. 9.2.1 from "Uncertainty Quantification and Predictive Computational 
        Science" (McClarren 2018).
    """
    return ((b-a)/2)*val + (a + b)/2


def map_beta_val_to_new_interval(val, a, b):
    """Map value on beta interval [0, 1] to [a, b]
    
    References:
        [Difference between standard beta and unstandard beta distributions?](https://stats.stackexchange.com/questions/186465/difference-between-standard-beta-and-unstandard-beta-distributions/186467#186467)
    """
    return val*(b-a) + a


def get_multi_index(collocation_nodes_matrix: ndarray) -> itertools.product:
    """Computes multi-index by taking cartesian product of summation indices.
    
    This is just the d-dimensional case of the below 2D example.

    Example:

    ```
    # Say you have 2 random variables and 2 collocation nodes
    # per random variable, then iterating over these nodes is naively
    # performed as shown below (you have i for first dimension, and j
    # for the second dimension)
    n_nodes = 2
    for i in range(n_nodes):
        for j in range(n_nodes):
            print(f"({i}, {j})")
    # this prints (0, 0), (0, 1), (0, 2), ..., (2, 2)
    # but this set of indices could also be computed by taking a cartesian
    # product, which is exactly what this function does
    from multidim_stochastic_collocation import get_multi_index
    from numpy import zeros
    n_rand_vars = 2
    dummy_matrix = zeros(shape=(n_rand_vars, n_nodes))
    multi_index = get_multi_index(dummy_matrix)
    print()
    for ij in multi_index:
        print(ij)
    ```
    """
    n_collocation_nodes_per_randvar = collocation_nodes_matrix.shape[1]
    n_rand_vars = collocation_nodes_matrix.shape[0] 

    multi_index = itertools.product(
        *[range(n_collocation_nodes_per_randvar) 
        for randvar in range(n_rand_vars)])

    return multi_index


def lagrange_basis(x, data_points, j):
    """Calculate the Lagrange basis function for the j-th data point.

    Args:
        x: The point at which we evaluate the basis function.
        data_points (array-like): The list of data points (x values).
        j: The index of the data point for which we calculate the basis function.

    Returns:
        The value of the j-th Lagrange basis function at point x.
    """
    n = len(data_points)
    basis = 1.0
    for i in range(n):
        if j != i:
            basis *= (x - data_points[i]) / (data_points[j] - data_points[i])

    return basis


def lagrange_basis_product(js, Zs, collocation_nodes_matrix):
    """Product of lagrange basis functions (see slide 9 UQ Lecture 8)."""
    prod = 1
    d_dims = len(js)
    for n in range(d_dims):
        Z_n = Zs[n]
        j_n = js[n]
        theta_M_n = collocation_nodes_matrix[n, :]
        prod *= lagrange_basis(Z_n, theta_M_n, j_n)
    return prod


def stochastic_collocation_summand(
    js: List[int], 
    Zs: ndarray, 
    collocation_nodes_matrix: ndarray, 
    model: Union[Callable, ndarray]):
    """Stochastic collocation on collocation nodes of indices `js`.
  
    Args:
        js: List of indices (i.e., the multi-index).
        Zs: Vector of realizations of the random variables 
            (i.e., Z_1, Z_2, ..., Z_d).
        collocation_nodes_matrix: Matrix of collocation nodes where the i^th
            row is the set of collocation nodes for the i^th random variable.
        model: Function `u` that will be evaluated at collocation nodes 
            corresponding to the indices `js` OR a tensor of precomputed
            model evaluations for the collocation nodes.

    Returns:
        The product of the `model` evaluated at the collocation nodes using the
        multi-index `js` AND the lagrange basis functions product.

    References: 
        On slide 9 of UQ lecture 8, this function corresponds to the summand
        (i.e., the "thing" inside the sums over the multi-index).
    """
    # determine how the model outputs for these nodes are computed
    if isinstance(model, Callable):
        # get the collocation nodes for the current multi-indices
        collocation_nodes_at_j = collocation_nodes_matrix[range(len(js)), js]

        # Determine function u evaluation at collocation nodes from multi-index js 
        u = model(*collocation_nodes_at_j)

    elif isinstance(model, ndarray):
        u = model[js]

    else:
        raise TypeError("`model` must be of type `Callable` or `ndarray`.")

    # get a product of lagrange basis functions
    L = lagrange_basis_product(js, Zs, collocation_nodes_matrix)

    return u*L


def multidim_stochastic_collocation(
    Zs: ndarray, 
    model: Union[Callable, ndarray], 
    collocation_nodes_matrix: ndarray,
    multi_index: Tuple):
    """Multidimensional stochastic collocation using dense tensor product.
 
    Args: 
        Zs: Vector of realizations of the random variables 
            (i.e., Z_1, Z_2, ..., Z_d).
        model: Function `u` that will be evaluated at collocation nodes 
            corresponding to the indices `js` OR a tensor of precomputed
            model evaluations for the collocation nodes.
        collocation_nodes_matrix: Matrix of collocation nodes where the i^th
            row is the set of collocation nodes for the i^th random variable.
        multi_index: Tuple of indices `js` used for the summation over
            a d-dimensional `Zs` vector.

    Returns:
        Multidimensional stochastic collocation approximation of `model`
        using random variables `Zs` and `collocation_nodes_matrix`.

    References:
        Slide 9 from UQ lecture 8.
    """
    # initialize u_approx
    u_approx: ndarray

    if isinstance(model, Callable):
        # uses a dummy evaluation of the model to ascertain its output shape
        u_approx = zeros_like(model(*Zs))

    elif isinstance(model, ndarray):
        precomputed_tensor_shape = model.shape
        model_shape = precomputed_tensor_shape[len(Zs): ]
        u_approx = zeros(shape=model_shape)

    else:
        raise TypeError("`model` must be of type `Callable` or `ndarray`")

    # iterate through multi-indices and sum using stochastic collocation
    for js in tqdm(multi_index, desc="Performing stochastic collocation"):
        u_approx += stochastic_collocation_summand(
            js, Zs, collocation_nodes_matrix, model)

    return u_approx



if __name__ == "__main__":
    ## Example stochastic collocation using uncached model evaluations

    # define random variable intervals 
    R0_interval = [1.5, 3.0] # ~ Uniform distribution   
    T_interval = [5, 10]     # ~ Uniform distribution
    tau_interval = [1, 14]   # ~ Beta distribution  
    beta_a = beta_b = 2      # parameters of beta distribution
 
    # Create the Zs vector by selecting arbitrary values in the appropriate
    # intervals for the random vars
    R0 = 2.2
    T = 9
    tau = 10
    Zs = [R0, T, tau] # this order is important for the model
   
    # Create the collocation nodes 
    # NOTE: The clenshaw curtis level determines how many collocation
    # nodes are used... and since the number of points in the tensor grid
    # is `num_nodes**d`, this grows exponentially fast and is a huge bottleneck
    # e.g., level 5 --> 17 collocation nodes and d=3 --> 17**3 = 4913 
    # evaluations of model function `u`
    clenshaw_curtis_level = 2 
    collocation_nodes_matrix = get_clenshawcurtis_collocation_nodes_matrix(
        k=clenshaw_curtis_level, d_dims=len(Zs)) 

    # Get the number of collocation nodes for each random var
    n_nodes_per_randvar = collocation_nodes_matrix.shape[1]

    # Scale the collocation nodes [-1, 1] -> [a, b] for respective random vars
    collocation_nodes_matrix[0, :] = array( # scale R0 nodes
        [map_uniform_val_to_new_interval(
            collocation_nodes_matrix[0, i], *R0_interval) 
            for i in range(n_nodes_per_randvar)]) 

    collocation_nodes_matrix[1, :] = array( # scale T nodes
        [map_uniform_val_to_new_interval(
            collocation_nodes_matrix[1, i], *T_interval) 
            for i in range(n_nodes_per_randvar)])

    collocation_nodes_matrix[2, :] = array( # scale tau nodes
        [map_uniform_val_to_new_interval(
            collocation_nodes_matrix[2, i], *tau_interval) 
            for i in range(n_nodes_per_randvar)])

    # Compute the multi-index 
    multi_index = tuple(get_multi_index(collocation_nodes_matrix))

    # perform stochastic collocation
    u_approx = multidim_stochastic_collocation(
        Zs, SEIRmodel, collocation_nodes_matrix, multi_index)
    
    # get the approximated infected from the stochastic collocation procedure
    infected_approx = u_approx[:, 2] 

    # get the infected from just running the model as is
    seir_model_solutions = SEIRmodel(*Zs)
    infected = seir_model_solutions[:, 2]
    
    # plot the approximated infected and the regular model infected
    plt.plot(infected_approx, label="Stochastic Collocation")
    plt.plot(infected, label="Regular")

    plt.xlabel("timesteps")
    plt.ylabel("infected")
    plt.title("Multidimensional Stochastic Collocation" 
    f" of {len(multi_index)} Points and Regular Model"
    f"\n[R0, t, tau] -> {Zs}")
    plt.legend()

    plt.show()

    ## Cache the model evaluations at the collocation nodes 
    # Initialize the cache
    n_randvars = len(Zs)

    model_evaluation_cache = zeros(
        shape=(*([n_nodes_per_randvar]*n_randvars), 
                *seir_model_solutions.shape))

    print(model_evaluation_cache.shape) 

    # precompute model at collocation nodes
    for js in multi_index:
        collocation_nodes_at_j = collocation_nodes_matrix[range(len(js)), js]
        model_eval = SEIRmodel(*collocation_nodes_at_j)

        # e.g., update `tensor[0, 0, 1, :, :]`  with the model evaluation
        # having used collocation nodes corresponding to multindices (0, 0, 1)
        eval_cache_ix = (*js, ...)
        model_evaluation_cache[eval_cache_ix]  = model_eval

    # Stochastic collocation using cached model evaluations
    u_approx_cached = multidim_stochastic_collocation(
        Zs, model_evaluation_cache, collocation_nodes_matrix, multi_index)  

    ## Check the cached and on the fly computed u_approx's are the same
    assert_msg =  "cached and on-the-fly u_approx are same"
    assert allclose(u_approx, u_approx_cached), assert_msg

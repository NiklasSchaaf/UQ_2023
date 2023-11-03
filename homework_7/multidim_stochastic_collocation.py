import itertools 
from numpy import ndarray, cos, pi, zeros, array


def get_collocation_nodes_matrix(k: int, d_dims: int) -> ndarray:
    """Return collocation nodes matrix using Curtis-Clenshaw nodes of level `k`.

    Example:
   
    >>> # Reproduce figure 7.1 from Xiu
    >>> # NOTE: k = 5 in figure, but to get 1089 points, k = 6 in this example
    >>> # ... i don't know why this is 
    >>> from multidim_stochastic_collocation import get_tensor_grid
    >>> from multidim_stochastic_collocation import  get_collocation_nodes_matrix
    >>> import matplotlib.pyplot as plt 
    >>> k = 6; d = 2;
    >>> collocation_nodes_matrix = get_collocation_nodes_matrix(k, d)
    >>> tensor_grid = get_tensor_grid(collocation_nodes_matrix)
    >>> plt.scatter(tensor_grid[0, :], tensor_grid[1, :], s = 1)
    >>> plt.show()

    Args:
        k: Level of Clenshaw-Curtis grid.
        d_dims: Number of random variables in problem of interest.

    Returns:
       Collocation nodes matrix of shape `[d_dims, m_ik]` where 
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


import itertools 
from numpy import ndarray, cos, pi, zeros, array

def dense_tensor_grid(k: int, d_dims: int, m_i: int) -> ndarray:
    """Return collocation nodes matrix and dense tensor grid at level `k`.

    Uses extrema of Chebyshev polynomials for grid nodes.

    Example:
   
    >>> # Reproduce figure 7.1 from Xiu
    >>> # NOTE: k = 5 in figure, but to get 1089 points, k = 6 in this example
    >>> # ... i don't know why this is 
    >>> from multidim_stochastic_collocation import dense_tensor_grid
    >>> import matplotlib.pyplot as plt 
    >>> k = 6; d = 2; m_i = 30 # m_i is pretty much arbitrary
    >>> collocation_nodes_matrix, tensor_grid = dense_tensor_grid(k, d, m_i)
    >>> plt.scatter(tensor_grid[:, 0], tensor_grid[:, 1], s = 1)
    >>> plt.show()

    Args:
        k: Level of Clenshaw-Curtis grid.
        d_dims: Number of random variables in problem of interest.
        m_i: Desired number of collocation points in each dimension. Note
            that the actual number of points will be reduced by the level
            `k`.

    Returns:
       Collocation nodes matrix  of shape `[d_dims, m_ik]` where 
        `m_ik = 2**(k-1) + 1` and a tensor grid representing the cartesian
        product of row `i` in the collocation matrix with row `j` s.t. `i != j`.

    References:
        Xiu ch. 7.2.1 and 7.2.2.
    """
    assert k >= 1, "level of Clenshaw-Curtis grid must be >= 1"
    # The below algorithm is directly from Xiu equation 7.10 and desc. Xiu p. 82
    # and builds the collocation nodes matrix (i.e., collocation nodes
    # per dimension)
    m_ik = 2**(k-1) + 1 if k > 1 else 1
    collocation_nodes_matrix = zeros(shape=(d_dims, m_ik))
    for i in range(1, d_dims+1):
        for j in range(1, m_ik+1):
            if k > 1:
                Z_ij = nested_clenshaw_curtis_node(j, m_ik)
            elif k == 1 and j == 1:
                Z_ij = 0
            
            collocation_nodes_matrix[i-1, j-1] = Z_ij

    # Create the tensor grid by taking a cartesian product of all rows in 
    # the collocation matrix 
    tensor_grid = array(
        tuple(
            itertools.product(
                *[collocation_nodes_matrix[i, :] for i in range(d_dims)])))

    return collocation_nodes_matrix, tensor_grid
   
 
def nested_clenshaw_curtis_node(j, m_ik):
    """Extrema of Chebyshev polynomial.
    
    References:
        Xiu eq. 7.10
    """
    return -cos((pi*(j - 1))/(m_ik - 1))


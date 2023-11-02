import numpy as np
import matplotlib.pyplot as plt


def heat_eq(z):
    """
    Solve the steady-state 1-d heat equation with finite differencing
    i.e., compute u(x) such that d_x( a(x) d_x u(x)) = -f(x),
    with d_x denoting derivative wrt x.
    
    Here, assume a(x)=a, constant in space -> equation becomes a u'' = -f    

    Parameters
    ----------
    z : float
        The input parameter z is the wavelength in the source term,
        f=sin(z*x-0.2);.

    Returns
    -------
    x : array, shape(n,)
        The spatial grid.
    u : array, shape (n,)
        The solution of the heat equation at source term sin(z*x-0.2)

    """

    # left and right boundaries of the domain
    xl = -1;
    xr = 1

    # grid spacing:
    h = 0.01;

    # number of points in x
    n = int((xr - xl) / h)

    # spatial grid x
    x = np.linspace(xl, xr, n)

    # boundary conditions
    ul = 0;
    ur = 0

    # source, with random wavelength z
    f = np.sin(z * x - 0.2)

    # diffusivity
    a = 1

    # create matrix and augmented source vector to solve system
    M = -2 * np.diag(np.ones(n)) + np.diag(np.ones(n - 1), k=1) + np.diag(np.ones(n - 1), k=-1);

    # apply boundary conditions    
    M[0, 0] = 1;
    M[0, 1] = 0
    M[n - 1, n - 1] = 1;
    M[n - 1, n - 2] = 0

    # construct right hand side
    b = -f * (h ** 2) / a
    b[0] = ul;
    b[n - 1] = ur

    # solve Mu=b:
    u = np.linalg.solve(M, b)

    return x, u

if __name__ == "__main__":
    x, u = heat_eq(np.pi)

    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel='x', ylabel='u')
    ax.plot(x, u)
    plt.tight_layout()
    plt.show()

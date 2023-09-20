import numpy as np
from scipy.integrate import quad
from scipy.special import legendre
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 1000)


def function(x):
    return np.exp(2 * x + np.sin(4 * x))


def InterpolationError(x):
    """
    Function is used in the calculation of the interpolation error.
    The coefficients are calculated by hand and correspond to n=3 order of interpolation.
    :param x:
    :return:
    """
    return np.square(np.exp(2 * x + np.sin(4 * x)) - 0.877 * np.square(x) - 1.589 * x - 1)


I = quad(InterpolationError, -1, 1)
error = np.sqrt(I[0])  # this is the approximation error

print(
    f"The estimation for the approximation error (interpolation) is {error} with an upper bound of the error being {I[1]}")


# Approximation of function by projection on Legendre polynomials

def integral(x, u, v, w):
    if type(w) == int or type(w) == float:
        return u(x) * v(x) * w
    else:
        return u(x) * v(x) * w(x)


def inner_product(u, v, w):
    return quad(integral, -1, 1, args=(u, v, w))[0]


def coefficients(order):
    numerator = inner_product(u=function, v=legendre(order), w=1)
    denominator = np.square(np.sqrt(inner_product(u=legendre(order), v=legendre(order), w=1)))
    result = numerator / denominator

    return result


def projection(order):
    projection_poly = np.poly1d(np.zeros(order + 1))
    for j in range(order + 1):
        projection_poly += coefficients(j) * legendre(j)

    return projection_poly


def projection_error(func, proj):
    return np.sqrt(quad(integral_error, -1, 1, args=(func, proj))[0])


def integral_error(x, func, proj):
    return np.square(np.abs(func(x) - proj(x)))


ord = 2
proj = projection(order=ord)
poly_proj_error = projection_error(function, proj)

print(f"The projection error of order {ord} is {poly_proj_error}")

error_projection = []
n = 50
for i in range(n + 1):
    proj = projection(order=i)
    error_projection.append(projection_error(function, proj))

plt.scatter(np.linspace(0, n, n + 1), error_projection)
# plt.xscale('log')
plt.yscale('log')
plt.xlabel('degree')
plt.ylabel('approximation error')
plt.title("Projection error for approximating the given function")
plt.show()


# Compute the normalisation coefficients of exercise 1
def Z_pdf(x):
    return (1 - x) ** 6 * (1 + x) ** 2


def Jacobi_1(x):
    return 1


def Jacobi_2(x):
    return 5 * x + 2


def Jacobi_3(x):
    return np.square(x)*(33/2) + 11*x + 0.5


gamma_1 = inner_product(u=Jacobi_1, v=Jacobi_1, w=Z_pdf)
gamma_2 = inner_product(u=Jacobi_2, v=Jacobi_2, w=Z_pdf)
gamma_3 = inner_product(u=Jacobi_3, v=Jacobi_3, w=Z_pdf)

print(f"The gamma normalisation constants are:\n Gamma1 = {gamma_1}\n Gamma2 = {gamma_2}\n Gamma3 = {gamma_3}")
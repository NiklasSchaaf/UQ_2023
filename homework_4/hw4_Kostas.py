import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.special import legendre, hermite, eval_hermitenorm
from scipy.integrate import nquad, quad, dblquad
from tqdm import tqdm
from numba import njit


@njit
def f(z1, z2):
    return -z1 * z2 ** 3 + np.exp(-0.5 * np.square(z1) - 0.1 * np.square(z2))


X, Y = np.meshgrid(sorted(np.random.uniform(-1, 1, 1000)), sorted(np.random.normal(0, 1, 1000)))
# X, Y = np.meshgrid(np.linspace(-1,1,1000), np.linspace(-3,3,1000))
plt.contourf(X, Y, f(X, Y))
plt.title("Contour plot of the deterministic f(x,y)")
plt.colorbar()
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f(X, Y))
ax.set_title('3D plot of the deterministic f(x, y)', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_zlabel('z', fontsize=12)
plt.show()


def get_pairs(max_order, dims=2):
    pairs = []
    for pair in product(np.arange(0, max_order+1), repeat=dims):
        if sum(pair) <= max_order:
            pairs.append(pair)
    return pairs


def inner_product(z, w, order, poly):
    if poly == "Hermite":
        return eval_hermitenorm(order, z) * eval_hermitenorm(order, z) * w(z)
    elif poly == "Legendre":
        return legendre(order)(z) * legendre(order)(z) * w


@njit
def pdf_normal(z, mean=0, sigma=1):
    return np.exp(-0.5 * np.square((z - mean) / sigma)) / (sigma * np.sqrt(2 * np.pi))


def numerator_func(z1, z2, i_vector):
    return f(z1, z2) * legendre(i_vector[0])(z1) * eval_hermitenorm(i_vector[1], z2) * 0.5 * pdf_normal(z2, mean=0, sigma=1)


def coefficients(i_vector):
    numerator = dblquad(numerator_func, -np.inf, np.inf, -1, 1, args=[i_vector])[0]
    gamma_i1 = quad(inner_product, -1, 1, args=(0.5, i_vector[0], "Legendre"))[0]
    gamma_i2 = quad(inner_product, -np.inf, np.inf, args=(pdf_normal, i_vector[1], "Hermite"))[0]
    denominator = gamma_i1 * gamma_i2

    result = numerator / denominator

    return result


def gPC(N, z1, z2):
    vectors = get_pairs(N)
    proj = np.zeros((len(z1), len(z1)))
    pbar = tqdm(total=len(vectors))
    coeffs = []
    for vec in vectors:
        proj += coefficients(vec) * legendre(vec[0])(z1) * eval_hermitenorm(vec[1], z2)
        coeffs.append(coefficients(vec))
        pbar.update()
    pbar.close()

    return proj


def visualise(X, Y, gpc, title):
    plt.contourf(X, Y, gpc)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, gpc)
    ax.set_title('3D plot of the ' + title, fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    plt.show()


def projection_error(X, Y, N):
    result_proj = gPC(N, X, Y)
    difference = f(X, Y) - result_proj
    visualise(X, Y, result_proj, title=f"gPC expansion of f(Z1, Z2) order {N}")
    visualise(X, Y, difference, title=fr"$f- P_{N} f$ Projection error of order {N}")
    pdf_uniform = 0.5 * np.ones(f(X, Y).shape)
    pdf_gaussian = pdf_normal(Y)
    weighted_diff = np.multiply(np.square(difference), np.multiply(pdf_uniform, pdf_gaussian))
    visualise(X, Y, np.multiply(pdf_uniform, pdf_gaussian), title="PDFs multiplied")
    error = np.mean(np.sqrt(np.sum(weighted_diff, axis=1)))
    visualise(X, Y, weighted_diff, title=f"Weighted projection error order {N}")
    print(f"projection error is {error}")

    return error


n_vals = np.arange(2, 14, 2)
errors = []
for val in n_vals:
    errors.append(projection_error(X, Y, val))
plt.plot(n_vals, errors, '-o')
plt.title("error for different orders")
plt.xlabel("N")
plt.ylabel("Error")
plt.show()

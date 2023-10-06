# Jared Frazier
# Implementation notes for homework 4
# Slide <##> denotes the homework 4 slide at number <##>
# Xiu <###> denotes the page number of the course textbook 
# Xiu "Numerical Methods for Stochastic Computation"
import itertools
from typing import List, Callable

import numpy as np
from scipy.integrate import dblquad, nquad
from scipy.special import legendre, hermite


def orthogonal_projection(
   f: Callable, 
   polynomials: List[Callable], 
   Zs: List[float], 
   order_n: int) -> Callable:
    """Orthogonal projection of `f(z1, z2, ..., z_d)` (slide 12)

    NOTE: Minimal implementation should only support 2D
    Not sure what Zs are supposed to be here... callables or actual
    realizations (e.g., uniform is a callable, but uniform() is a realization)

    Args:
        f: Function to approximate.
        polynomials: Scipy callables of form `f(order_n) -> poly1d`.
        Zs: The k^th element is the realization of the k^th proba. density func
        order_n:
    """
    projection = 0
    polynomial_order_combos = polynomial_order_combinations(order_n)
    polynomial_orders: List[int] = None

    # iterate through polynomial order combinations 
    for polynomial_orders in polynomial_order_combos:
        polynomial_product = compute_polynomial_product(
            polynomials=polynomials,
            polynomial_orders=polynomial_orders,
            Zs=Zs)
        
        coefficient = compute_coefficient(
            polynomials=polynomials,
            polynomial_orders=polynomial_orders)
        
        projection += coefficient*polynomial_product
        
    return projection


def compute_coefficient(
    polynomials: List[Callable],
    polynomial_orders: List[int]):
    """Orthogonal projection coefficients `\hat{f}_{\mathbf{i}}`

    NOTE: Call quadrature here.
    """
    pass 


def compute_polynomial_product(
    polynomials: List[Callable], # [legendre, hermite]
    polynomial_orders: List[int], # [2, 0]
    Zs: List[float]):
    """
    Args:
        polynomials: Special scipy orthogonal polynomials of form `f(n)`.

    Return:
        A scalar representing the product of orthogonal polynomials...
    """
    prod = 1
    d = len(polynomials)
    for k in range(d):
        i_k = polynomial_orders[k]
        polynomial_k = polynomials[k]
        Z_k = Zs[k]
        prod *= polynomial_k(i_k)(Z_k)
    return prod


def polynomial_order_combinations(N, dims:int = 2):
    result = []
    for combo_vector in itertools.product(range(N+1), repeat=dims):
        if sum(combo_vector) <= N:
            result.append(combo_vector)
    return result

 

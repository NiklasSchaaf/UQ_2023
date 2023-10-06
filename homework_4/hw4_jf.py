# Jared Frazier
# Implementation notes for homework 4
# Slide <##> denotes the homework 4 slide at number <##>
# Xiu <###> denotes the page number of the course textbook 
# Xiu "Numerical Methods for Stochastic Computation"
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

    NOTE: Since this needs to return a callable, the input to such
    a callable should be a vector of realization of the `k^th` distribution
    Z_k... so 
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
    
    Still thinking about this one... product implementation is also not optimal
    """
    prod = 1
    d = len(polynomials)
    for k in range(d):
        i_k = polynomial_orders[k]
        polynomial_k = polynomials[k]
        Z_k = Zs[k]
        prod *= polynomial_k(i_k)(Z_k)
    return prod


def polynomial_order_combinations(order_n: int) -> List[List[int]]:
    """Return combinations of polynomial order <= order_n (slide 12,13)"""
    pass

 

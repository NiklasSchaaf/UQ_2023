# Jared Frazier
# Implementation notes for homework 4
# Slide <##> denotes the homework 4 slide at number <##>
# Xiu <###> denotes the page number of the course textbook 
# Xiu "Numerical Methods for Stochastic Computation"
from typing import List, Callable
from scipy.integrate import dblquad, nquad
import numpy as np

def orthogonal_projection(
   f: Callable, polynomials: List[Callable], Zs: List[Callable], order_n: int):
    """Orthogonal projection of `f(z1, z2, ..., z_d)` (slide 12)

    Minimal implementation should only support 2D
    """
    projection = 0
    polynomial_order_combos = polynomial_order_combinations(order_n)
    polynomial_orders: List[int] = None
    for polynomial_orders in polynomial_order_combos 
        i = polynomial_orders
         

def coefficients(
    polynomial_orders: List[int], polynomials: List[Callable]):
    """Orthogonal projection coefficients `\hat{f}_{\mathbf{i}}`

    NOTE: Call quadrature here.
    """
    pass 


def polynomial_functions_prod(
    polynomials: List[Callable], 
    polynomial_orders: List[int], 
    Zs: List[Callable]):
    pass 


def polynomial_order_combinations(order_n: int) -> List[int]:
    """Return combinations of polynomial order <= order_n (slide 12,13)"""
    pass

 

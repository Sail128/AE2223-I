# import numpy as np
from math import sin, cos, pi

def phi_exact(x: float, y: float):
    """calculates the exact potential

    Arguments:
        x {[float]} --
        y {[float]} --

    Returns:
        float -- potential
    """
    phi_exact = sin(2*pi*x) * sin(2*pi*y)
    return phi_exact


def u_exact_calc(x: float, y: float):
    """calculates the exact velocity

    Arguments:
        x {float} -- [description]
        y {float} -- [description]

    Returns:
        float -- ux, uy
    """
    u_exact_i = 2*pi * cos(2*pi*x) * sin(2*pi*y)
    u_exact_j = 2*pi * sin(2*pi*x) * cos(2*pi*y)
    return u_exact_i, u_exact_j


def u_exact_x(x: float, y: float):
    return 2*pi * cos(2*pi*x) * sin(2*pi*y)


def dux_dx(x: float, y: float):
    return -4*pi*pi*sin(2*pi*x) * sin(2*pi*y)


def u_exact_y(x: float, y: float):
    return 2*pi * sin(2*pi*x) * cos(2*pi*y)


def duy_dy(x: float, y: float):
    return -4 * pi * pi * sin(2*pi*x) * sin(2*pi*y)


def divu_exact(x: float, y: float):
    return dux_dx(x, y)+duy_dy(x, y)


def f_exact(x: float, y: float):
    return -8*pi * pi * sin(2*pi*x) * sin(2*pi*y)

# import numpy as np
from libc.math cimport sin, cos, pi
#from math import sin, cos, pi

cpdef double phi_exact(double x, double y):
    """calculates the exact potential

    Arguments:
        x {[float]} --
        y {[float]} --

    Returns:
        float -- potential
    """
    return sin(2*pi*x) * sin(2*pi*y)


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


cpdef double u_exact_x(double x, double y):
    return 2*pi * cos(2*pi*x) * sin(2*pi*y)


def dux_dx(x: float, y: float):
    return -4*pi*pi*sin(2*pi*x) * sin(2*pi*y)


cpdef double u_exact_y(double x, double y):
    return 2*pi * sin(2*pi*x) * cos(2*pi*y)


def duy_dy(x: float, y: float):
    return -4 * pi * pi * sin(2*pi*x) * sin(2*pi*y)


def divu_exact(x: float, y: float):
    return dux_dx(x, y)+duy_dy(x, y)


cpdef double f_exact(double x, double y):
    return -8*pi * pi * sin(2*pi*x) * sin(2*pi*y)

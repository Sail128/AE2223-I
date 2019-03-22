import numpy as np


def phi_exact(x: float, y: float):
    """calculates the exact potential

    Arguments:
        x {[float]} --
        y {[float]} --

    Returns:
        float -- potential
    """
    phi_exact = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
    return phi_exact


def u_exact_calc(x: float, y: float):
    """calculates the exact velocity

    Arguments:
        x {float} -- [description]
        y {float} -- [description]

    Returns:
        float -- ux, uy
    """
    u_exact_i = 2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
    u_exact_j = 2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
    return u_exact_i, u_exact_j


def u_exact_x(x: float, y: float):
    return 2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)


def dux_dx(x: float, y: float):
    return -4*np.pi*np.pi*np.sin(2*np.pi*x) * np.sin(2*np.pi*y)


def u_exact_y(x: float, y: float):
    return 2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)


def duy_dy(x: float, y: float):
    return -4 * np.pi * np.pi * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)


def divu_exact(x: float, y: float):
    return dux_dx(x, y)+duy_dy(x, y)


def f_exact(x: float, y: float):
    return -8*np.pi * np.pi * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

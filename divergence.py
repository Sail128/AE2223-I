import numpy as np

def divergence(f,di=None):
    """Calculates the divergence of a vector field.
    
    Arguments:
        f {List} -- List of N, Nd arrays [Fx_1, Fx_2, ..., Fx_n] where F is nd array. assumed indexing 'ij'
        di {List} -- List of the associated dx sizes
    Returns:
        ndarray -- ndarray of the divergence 
    """
    num_dims = len(f)
    if di == None:
        di = [1]*num_dims
    return np.ufunc.reduce(np.add, [np.gradient(f[i], di[i], axis=i) for i in range(num_dims)])


def test():
    from tabulate import tabulate
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    from e_functions import phi_exact,u_exact_x,u_exact_y,f_exact,divu_exact
    N = 50
    xrange = np.linspace(-1., 1., N)
    yrange = np.linspace(-1., 1., N)
    xs, ys = np.meshgrid(xrange, yrange, indexing='ij', sparse=False)
    dx = (np.abs(xs[0, 0]-xs[-1, -1]))/(xs.shape[0]-1)
    dy = (np.abs(ys[0, 0]-ys[-1, -1]))/(ys.shape[1]-1)
    print(dx, dy)
    phi = np.vectorize(phi_exact)(xs, ys)
    ux = np.vectorize(u_exact_x)(xs, ys)
    uy = np.vectorize(u_exact_y)(xs, ys)
    dux = np.gradient(ux, dx, axis=0)
    duy = np.gradient(uy, dy, axis=1)
    U = [ux, uy]
    divu = divergence(U,di=[dx,dy])  # dux+duy
    divu_a = dux+duy
    divu_e = np.vectorize(f_exact)(xs, ys)
    print(np.sum(np.vectorize(divu_exact)(xs, ys)-divu_e))
    print(np.sum(divu-divu_e))
    # print(tabulate(xs))
    # print(tabulate(ys))
    # print(tabulate(phi))
    # print(tabulate(np.gradient(phi)))
    # print(np.gradient(phi))
    plt.contourf(xs, ys, divu, 50, cmap=cm.plasma)
    plt.colorbar()
    plt.contour(xs, ys, divu, 10, colors='k',
                linewidths=1, linestyles='solid')
    plt.quiver(xs, ys, ux, uy)
    plt.show()

    NY = 50
    ymin = -2.
    ymax = 2.
    dy = (ymax -ymin )/(NY-1.)

    NX = NY
    xmin = ymin
    xmax = ymax
    dx = (xmax -xmin)/(NX-1.)

    y = np.linspace(ymin,ymax,NY)#np.array([ ymin + float(i)*dy for i in range(NY)])
    x = np.linspace(xmin,xmax,NX)#np.array([ xmin + float(i)*dx for i in range(NX)])

    x, y = np.meshgrid( x, y, indexing = 'ij', sparse = False)

    Fx  = np.cos(x + 2*y)
    Fy  = np.sin(x - 2*y)

    F = [Fx, Fy]
    g = divergence(F)

    plt.pcolormesh(x, y, g)
    plt.colorbar()
    plt.quiver(x,y,Fx,Fy)
    plt.show()


if __name__ == "__main__":
    test()

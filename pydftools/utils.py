"""
Some handy utility functions used by the other modules.
"""

from copy import deepcopy
import numpy as np

def centres_to_edges(centres):
    "Assuming centres is regularly spaced, return bin edges"
    dx = centres[1] - centres[0]
    return np.linspace(centres[0] - dx/2, centres[-1]+dx/2, len(centres)+1)

def numerical_jac(func, args, dx=1e-5 ):
    if np.isscalar(dx):
        dx = np.repeat(dx,len(args))

    y0 = func(args)
    out = [0]*len(args)
    args = list(args)
    for i in range(len(args)):
        args[i] += dx[i]

        yy = func(args)
        out[i] = (yy - y0)/dx[i]
        args[i] -= dx[i]
    return np.array(out)


def numerical_hess(func, args, dx=1e-5):
    if np.isscalar(dx):
        dx = np.repeat(dx, len(args))

    j0 = numerical_jac(func, args, dx)
    out = [0]*len(args)
    args = list(args)
    for i in range(len(args)):
        args[i] += dx[i]
        out[i] = (numerical_jac(func, args, dx) - j0)/dx[i]
        args[i] -= dx[i]
    return np.array(out)


def sample_ellipsoid(cov, size=1, add_boundaries=False, mean=None):
    """
    Creates a random sample of points within an n-dimensional ellipsoid defined by a covariance matrix.

    Parameters
    ----------
    cov : array-like
        A covariance matrix defining the ellipsoid. The ellipsoid boundaries are defined by the 1-sigma
        standard deviations.

    size : int, optional
        The number of points to sample.

    add_boundaries : bool, optional
        Optionally return (non-random) points at the extremeties of the each dimension of the ellipsoid.

    Returns
    -------
    v : array-like
        A (p, size)-array, where p is the number of dimensions. The random points in the ellipse

    Notes
    -----
    The returned points are *not* uniformly random in the ellipse.
    """
    eigval, eigvec = np.linalg.eig(cov)
    npar = len(cov)

    if mean is None:
        mean = np.zeros(npar)
    # sample surface of covariance ellipsoid
    nsteps = size
    v_new = np.zeros((npar, nsteps ))

    # Do random points
    if size>0:
        e = np.random.normal(size=(npar, size))
        e /= np.sqrt(np.sum(e**2, axis=0))
    else:
        e = np.zeros((npar, 0))

    if add_boundaries:
        e = np.concatenate((e.T, np.diag(np.ones(3))))
        e = np.concatenate((e, -np.diag(np.ones(3)))).T

    # for i in range(nsteps + 2 * npar):
    #     if i <= nsteps - 1:
    #         e = np.random.normal(size=npar)
    #         e /= np.sqrt(np.sum(e ** 2))
    #     else:
    #         e = np.zeros(npar)
    #         if i > nsteps + npar - 1:
    #             e[i - nsteps - npar] = 1
    #         else:
    #             e[i - nsteps] = -1


    v = np.array([np.matmul(eigvec, np.sqrt(eigval) * ei) for ei in e.T])
    return v + mean


# def deepcopy_with_sharing(obj, shared_attribute_names, memo=None):
#     '''
#     Deepcopy an object, except for a given list of attributes, which should
#     be shared between the original object and its copy.
#
#     obj is some object
#     shared_attribute_names: A list of strings identifying the attributes that
#         should be shared between the original and its copy.
#     memo is the dictionary passed into __deepcopy__.  Ignore this argument if
#         not calling from within __deepcopy__.
#
#     Taken from this StackOverflow answer: https://stackoverflow.com/a/24621200/1467820
#     '''
#     assert isinstance(shared_attribute_names, (list, tuple))
#     shared_attributes = {k: getattr(obj, k) for k in shared_attribute_names}
#
#     if hasattr(obj, '__deepcopy__'):
#         # Do hack to prevent infinite recursion in call to deepcopy
#         deepcopy_method = obj.__deepcopy__
#         obj.__deepcopy__ = None
#
#     for attr in shared_attribute_names:
#         del obj.__dict__[attr]
#
#     clone = deepcopy(obj)
#
#     for attr, val in shared_attributes.items():
#         setattr(obj, attr, val)
#         setattr(clone, attr, val)
#
#     if hasattr(obj, '__deepcopy__'):
#         # Undo hack
#         obj.__deepcopy__ = deepcopy_method
#         del clone.__deepcopy__
#
#     return clone
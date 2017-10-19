"""
A module for defining generative distribution function models.

All models *must* be subclassed from :class:`~Model`, which provides the abstract base methods required to implement.
"""

import numpy as np
from .utils import numerical_jac, numerical_hess


class Model(object):
    """
    Base class defining a generative distribution function model

    All models *must* be subclassed from this, which provides the abstract base methods required to implement.
    The primary method is :meth:`~gdf`, which defines the generative distribution, though the class also provides
    information about the parameters and other useful things.

    Parameters
    ----------
    p0 : sequence
        A vector of parameters to use as the default for any methods that require them.

    Examples
    --------

    Evaluate and plot a Schechter function
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(7,11,100)
    >>> mass = 10**x
    >>> parameters = (-2,10,-1.5)
    >>> model = Schechter(parameters)
    >>> plt.plot(mass, model.gdf(x, parameters))
    >>> plt.xscale('log')
    >>> plt.yscale('log')

    Any model can be inspected before instantiation. Its default parameters are (using Schechter as an example):
    >>> Schechter._p0_default

    Its equation is
    >>> Schechter.gdf_equation

    And the names of its parameters are
    >>> Schechter.names_text
    """

    "Latex-Equation for gdf"
    gdf_equation = None

    "Text-friendly parameter names"
    names_text = None

    "Latex-friendly parameters names"
    names = None

    _p0_default = None

    def __init__(self, p0=None):
        if p0 is None:
            self.p0 = self._p0_default
        else:
            self.p0 = p0

        if not hasattr(self, "n_param"):
            if self.names is not None:
                self.n_param = len(self.names)
            elif self.names_text is not None:
                self.n_param = len(self.names)
            else:
                raise ValueError("Model has not specified the number of parameters")

    def gdf(self,x,p):
        """
        The generative distribution function.

        Parameters
        ----------
        x : array-like
            The n-dimensional variate.

        p : tuple
            The parameters of the distribution.

        Returns
        -------
        phi : array-like
            Array of same size as `x`, with value at each point.
        """
        pass

    def gdf_jacobian(self, x, p):
        """
        The jacobian of the GDF as a function of x at point p.
        """
        fnc = lambda p : self.gdf(x, p)
        jac = numerical_jac(fnc, p)
        return jac

    def gdf_hessian(self, x, p):
        """
        The jacobian of the GDF as a function of x at point p.
        """
        fnc = lambda p : self.gdf(x, p)
        return numerical_hess(fnc, p)


class Schechter(Model):
    """
    A Schechter function model.
    """
    _p0_default = (-2.,11.,-1.3)
    names_text = [
        'log_10 (phi_star)',
        'log_10 (M_star)',
        'alpha'
    ]

    names = [r'$\log_{10} \phi_\star$',
             r'$\log_{10} M_\star$',
             r'$\alpha$']

    gdf_equation = r"$\frac{dN}{dVdx} = \log(10) \phi_\star \mu^{\alpha+1} \exp(-\mu)$, where $\mu = 10^{x - \log_{10} M_\star}$"

    def gdf(self, x, p):
        mu = 10 ** (x - p[1])
        return np.log(10) * 10 ** p[0] * mu ** (p[2] + 1) * np.exp(-mu)

    def gdf_jacobian(self,x,p):
        g = self.gdf(x,p)
        return np.log(10)*g* np.array([np.ones_like(x), (-p[2]-1) + 10**(x-p[1]), (x-p[1])])

    def gdf_hessian(self, x, p):
        g = self.gdf(x,p)
        jac = self.gdf_jacobian(x,p)

        p00 = jac[0]
        p01 = jac[1]
        p02 = jac[2]
        p22 = jac[2] * (x-p[1])
        p11 = jac[1]*(-p[2]-1) - np.log(10)*10**(x-p[1])*g + 10**(x-p[1])*jac[1]
        p12 = jac[1]*x - g - p[1]*jac[1]

        return np.log(10) * np.array([[p00, p01, p02],
                                      [p01, p11, p12],
                                      [p02, p12, p22]])

class MRP(Model):
    """
    An MRP model (see Murray, Robotham, Power, 2017)
    """
    _p0_default = (-2., 11., -1., 1)
    names_text = [
        'log_10 (phi_star)',
        'log_10 (M_star)',
        'alpha',
        'beta'
    ]

    names = [r'$\log_{10} \phi_\star$',
             r'$\log_{10} M_\star$',
             r'$\alpha$',
             r'$\beta$']

    gdf_equation = r"$\frac{dN}{dVdx} = \log(10) \beta \phi_\star \mu^{\alpha+1} \exp(-\mu^\beta)$, where $\mu = 10^{x - \log_{10} M_\star}$"

    def gdf(self, x, p):
        mu = 10 ** (x - p[1])
        return np.log(10) * p[3] * 10 ** p[0] * mu ** (p[2] + 1) * np.exp(-mu**abs(p[3]))

class PL(Model):
    """
    A power-law model.
    """
    _p0_default = (2.,-1.)
    names_text = ( "log_10(A)", "alpha")
    names = (r'$\log_{10}A$', r'$\alpha$')

    gdf_equation = r"$\frac{dN}{dVdx} = A 10^{\alpha x}$"

    def gdf(self, x, p):
        return 10** p[0] * (10**(p[1]*x))
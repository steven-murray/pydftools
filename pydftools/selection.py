"""
Provides several classes that define the selection function in different ways.

All selection function classes are derived from the :class:`~Selection` abstract base class.
This should not be instantiated directly, but provides the structure for its subclasses. Namely,
each subclass should provide a single method, ``Veff(x)``, which defines the effective volume as a
function of the object properties (eg. log mass).

Using the three subclasses defined in this module, there are 5 ways of defining a selection function:

1. Using :class:`~SelectionVeff`, and supplying `veff` as a single positive number. This number will
   be interpreted as a constant volume, ``Veff(x)=selection``, in which all objects are fully
   observable. ``V(xval)=0`` is assumed outside the "observed domain" (defined by xmin,xmax). This
   definition can be used for volume-complete surveys or for simulated galaxies in a box.
2. Using :class:`~SelectionVeff`, and supplying `veff` as a function of a ``D``-dimensional vector.
   This is directly used as the effective volume, and values outside the observed domain are
   rendered as zero.
3. Using :class:`~SelectionVeffPoints`, and supplying `veff` as a length ``N`` vector, and ``xval``
   as an ``NxD`` array. The elements will be interpreted as the volumes of each sample point.
   ``Veff(xval)`` is interpolated (linearly in ``1/V``) for other values of x. ``V(xval)=0``
   is assumed outside the observed domain, which is defined as the min, max of the passed `xval`.
4. The same as (3), except that an additional function, `veff_extrap` is passed, which defines the
   effective volume outside of the observed domain.
5. Using :class:`~SelectionRdep`, and supplying `f`, `dvdr`, `rmin` and `rmax`, where ``f(xval,r)``
   is the isotropic selection function and ``dvdr(r)`` is the derivative of the total survey volume
   as a function of comoving distance `r`. The scalars ``rmin`` and ``rmax`` (can be ``0`` and
   ``Inf``) are the minimum and maximum comoving distance limits of the survey. Outside these limits
   ``Veff(x)=0`` will be assumed.

"""
import attr
import scipy.special as sp
import numpy as np
from cached_property import cached_property
from scipy.integrate import quad
from scipy.optimize import minimize, brentq
from scipy.interpolate import (
    InterpolatedUnivariateSpline as spline,
    RectBivariateSpline,
)
from abc import ABCMeta, abstractmethod


@attr.s
class Selection(object):
    """
    Abstract base class representing the selection function of the data used when fitting the generative DF.

    Parameters
    ----------
    vol_renorm : float
        A single number which re-normalises the total volume of the sample. Useful for creating mock observations
        tuned to a given output number of samples.
    """

    __metaclass__ = ABCMeta

    vol_renorm = attr.ib(default=1.0)
    xmax = attr.ib(default=20.0, converter=lambda x: np.atleast_1d(np.array(x)))
    xmin = attr.ib(default=0.0, converter=lambda x: np.atleast_1d(np.array(x)))

    def __attrs_post_init__(self):
        x = np.linspace(self.xmin, self.xmax, 1000)
        veff = self.Veff(x)
        if np.any(veff == 0) or np.any(np.isinf(veff)):
            indx = np.where(np.logical_and(veff > 0, np.logical_not(np.isinf(veff))))[0]

            print(
                "Warning: xmin returns Veff(xmin)=0, setting xmin, xmax to %s, %s"
                % (x[indx].min(), x[indx].max())
            )

            self.xmin = x[indx].min()
            self.xmax = x[indx].max()

    @xmin.validator
    def _xmin_validator(self, att, val):
        if np.any(val > self.xmax):
            raise ValueError("xmin cannot be greater than xmax.")

        if val.size != self.xmax.size:
            raise ValueError("xmax and xmin must be of the same length")

    @abstractmethod
    def _veff_fnc(self, x):
        raise NotImplementedError(
            "The Selection abstract base class should not be instantiated directly"
        )

    @abstractmethod
    def _veff_extrap(self, x):
        return np.zeros_like(x)

    def Veff(self, x):
        """
        The effective volume of the observation for a set of properties x.

        Parameters
        ----------
        x : array-like
            Either a 1D vector of an observed property, or a 2D vector, where the 2nd dimension corresponds to the different properties observed.

        Returns
        -------
        V : array
            A 1D vector, of the same length as x, giving the effective volume of the observation at that point in observation space.
        """
        x = np.atleast_1d(x)
        # Return vol-renormed function of veff_extrap outside observed region, and veff_fnc inside it.
        return self.vol_renorm * np.where(
            np.logical_or(x < self.xmin, x > self.xmax),
            self._veff_extrap(x),
            self._veff_fnc(x),
        )


def _veff_converter(val):
    if callable(val):
        return val
    elif np.isscalar(val):
        return lambda x: val * np.ones_like(x)


@attr.s
class SelectionVeff(Selection):
    """
    Base class for simple Selection functions, where only the effective volume function is given.

    Parameters
    ----------
    Veff : callable, optional
        A function of a D-dimensional vector `x`, specifying the effective volume associated with an object of properties `x`.
        Default is 10 ** (2x).
    """

    veff = attr.ib(lambda x: 10 ** (2 * x), convert=_veff_converter)

    @veff.validator
    def _veff_validator(self, att, val):
        assert callable(val)

    def _veff_fnc(self, x):
        return self.veff(x)

    def _veff_extrap(self, x):
        return super(SelectionVeff, self)._veff_extrap(x)


def _callable_validator(inst, att, val):
    assert callable(val)


@attr.s
class SelectionVeffPoints(Selection):
    """
    Simple Selection function where only effective volume is given, for a set of discrete points

    In this case, we set xmin, xmax equal to the min/max of the passed xval.

    Parameters
    ----------
    veff : array-like
        Array of effective volumes
    xval : array-like
        Array of x-values to which veff correspond
    veff_extrap: callable, optional
        A function of one variable, x, which defines the effective volume outside the observed limits.
    """

    veff = attr.ib(default=None)
    xval = attr.ib(default=None, convert=lambda x: np.atleast_2d(x).T)
    veff_extrap = attr.ib(
        default=None, validator=attr.validators.optional(_callable_validator)
    )

    @veff.validator
    def _veff_validator(self, att, val):
        assert hasattr(val, "__len__")
        assert len(val.shape) == 1
        if val.min() < 0:
            raise ValueError("All values of selection (=Veff) must be positive.")

    @xval.validator
    def _xval_validator(self, att, val):
        assert len(val) == len(self.veff)

    @cached_property
    def xmin(self):
        return np.array([x.min() for x in self.xval.T])

    @cached_property
    def xmax(self):
        return np.array([x.max() for x in self.xval.T])

    @cached_property
    def _veff_fnc(self):

        n_dim = self.xval.shape[1]

        if n_dim == 1:
            # Sort the inputs so as to get a good spline
            sort_ind = np.argsort(self.xval[:, 0])
            veff = self.veff[sort_ind]
            xval = self.xval[:, 0][sort_ind]

            spl = spline(
                xval, 1 / veff, k=1, ext=3
            )  # Setup to imitate dftools R version
            return lambda x: np.where(
                x < xval.min(), self._veff_extrap(x), (1 / spl(x))
            )
        elif n_dim == 2:

            def vapprox(xval):
                spl = RectBivariateSpline(
                    self.xval[:, 0], self.xval[:, 1], 1 / self.veff, kx=1, ky=1
                )
                z = 1 / spl.ev(xval[:, 0], xval[:, 1])
                #                    z = 1 / (akima::interp(x[, 1], x[, 2], 1 / Veff.values, xval[1], xval[2], duplicate = 'mean'))$z
                if np.isnan(z):
                    return 0
                else:
                    return z

            return np.vectorize(vapprox)
        else:
            raise ValueError(
                "Linear interpolation of Veff not implemented for DF with more than 2 dimensions. Use a different selection type."
            )

    def _veff_extrap(self, x):
        if self.veff_extrap is not None:
            return self.veff_extrap(x)
        else:
            return super(SelectionVeffPoints, self)._veff_extrap(x)


@attr.s
class SelectionRdep(Selection):
    """
    Base class for selection functions given as r-dependent functions

    Parameters
    ----------
    f : callable, optional
        The selection function ``f(x,r)``, giving the ratio between the expected number of detected galaxies and true
        galaxies of log-mass ``x`` and comoving distance ``r``. Normally this function is bound between 0 and 1.
        It takes the value 1 at distances, where objects of mass ``x`` are easily detected, and 0 at distances where
        such objects are impossible to detect. A rapid, continuous drop from 1 to 0 normally occurs at the limiting
        distance ``rmax``, at which a galaxy of log-mass ``x`` can be picked up. ``f(x,r)`` can never by smaller than 0,
        but values larger than 1 are conceivable, if there is a large number of false positive detections in the survey.
        The default is ``f(x,r) = erf((1-1e3*r/sqrt(10**x))*20)*0.5+0.5}``, which mimics a sensitivity-limited survey
        with a fuzzy limit.
    dvdr : callable, optional
        The function ``dVdr(r)``, specifying the derivative of the survey volume ``V(r)`` as a function of comoving
        distance ``r``. This survey volume is simply the total observed volume, irrespective of the detection probability,
        which is already specified by the function ``f``. Normally, the survey volume is given by ``V(r)=Omega*r**3/3``,
        where ``Omega`` is the solid angle of the survey. Hence, the derivative is ``dVdr(r)=Omega*r**2``.
        The default is ``Omega=2.13966`` [sterradians], chosen such that the expected number of galaxies is exactly 1000
        when combined with the default selection function ``f(x,r)``.
    g : callable, optional
        Function of distance ``r`` describing the number-density variation of galaxies due to cosmic large-scale
        structure (LSS). Explicitly, ``g(r)>0`` is the number-density at ``r``, relative to the number-density without
        LSS. Values between 0 and 1 are underdense regions, values larger than 1 are overdense regions. In the absence
        of LSS, ``g(r)=1``. Note that g is automatically rescaled, such that its average value in the survey volume is 1.
    rmin,rmax : float, optional
        Minimum and maximum distance of the survey. Outside these limits the function ``f(x,r)`` will automatically be
        assumed to be 0.
    """

    f = attr.ib(
        default=lambda x, r: sp.erf((1 - 1e3 * r / np.sqrt(10 ** x)) * 20) * 0.5 + 0.5,
        validator=_callable_validator,
    )
    dvdr = attr.ib(default=lambda r: 2.13966 * r ** 2, validator=_callable_validator)
    g = attr.ib(default=None, validator=attr.validators.optional(_callable_validator))
    rmin = attr.ib(default=0, convert=np.float)
    rmax = attr.ib(default=20, convert=np.float)

    @rmax.validator
    def _rmax_validator(self, att, val):
        assert val > self.rmin

    def dVdr(self, r):
        """
        The function dvdr, re-normalised by :attr:`vol_renorm`
        """
        return self.vol_renorm * self.dvdr(r)

    @cached_property
    def _veff_no_lss_fnc(self):
        def fnc(xval):
            # Use the un-normalised dvdr because it will be normalised.
            return quad(lambda r: self.f(xval, r) * self.dvdr(r), self.rmin, self.rmax)[
                0
            ]

        return np.vectorize(fnc)

    def _veff_no_lss(self, x):
        """
        The effective volume without LSS
        """
        return self._veff_no_lss_fnc(x)

    @cached_property
    def _gnorm(self):
        """
        g(r) properly normalised, such that the average value of g in the survey volume is 1

        Returns
        -------
        g : callable
            Scaled g(r).
        """
        if self.g is None:
            return None
        else:
            gnorm = (
                quad(lambda r: self.dVdr(r) * self.g(r), self.rmin, self.rmax)[0]
                / quad(self.dVdr, self.rmin, self.rmax)[0]
            )
            return lambda r: self.g(r) / gnorm

    @cached_property
    def _veff_fnc(self):
        """
        The effective volume (including LSS, if any provided).

        Parameters
        ----------
        x

        Returns
        -------

        """
        if self.g is None and hasattr(self, "_veff_lss"):
            return self._veff_lss
        elif self.g is not None:
            # evaluate effective volume and source count density with LSS
            def veff_lss_elemental(x):
                fct = (
                    lambda r: self.f(x, r) * self._gnorm(r) * self.dvdr(r)
                )  # Use the un-normalised dvdr because it will be normalised.
                return quad(fct, self.rmin, self.rmax)[0]

            return np.vectorize(veff_lss_elemental)

        else:
            return self._veff_no_lss

    def _veff_extrap(self, x):
        return super(SelectionRdep, self)._veff_extrap(x)

    def _get_veff_lss(self, r, grid, p, model, weight=lambda x: np.ones_like(x)):
        """
        Generate the best-fit Veff in the presence of unknown LSS.

        Parameters
        ----------
        p : tuple
            Parameters of the current model.
        """
        if self.g is not None:
            raise RuntimeError("You do not need to correct for LSS bias if g is known.")

        use_simpson = len(grid.xmin) == 1

        # evaluate integrals
        def integrand_lss(x, r):
            return self.f(x, r) * model.gdf(x, p)

        integral = np.empty(len(r))
        if use_simpson:
            for i in range(len(r)):
                integral[i] = quad(integrand_lss, grid.xmin, grid.xmax, args=(r[i],))[0]

        else:
            for i in range(len(r)):
                integral[i] = np.sum(integrand_lss(grid.x, r[i])) * grid.dvolume

        # make Veff.lss function
        def veff_lss_function_elemental(xval):
            f = self.f(xval, r)
            lst = f > 0
            return np.sum(f[lst] / integral[lst])

        veff_lss_scale = np.vectorize(
            veff_lss_function_elemental
        )  # Vectorize(Veff.lss.function.elemental)

        def int_ref(x):
            return self._veff_no_lss(x) * model.gdf(x, p) * weight(x)

        def int_exp(x):
            return veff_lss_scale(x) * model.gdf(x, p) * weight(x)

        if use_simpson:
            reference = quad(int_ref, grid.xmin, grid.xmax)[0]
            expectation = quad(int_exp, grid.xmin, grid.xmax)[0]
        else:
            reference = np.sum(int_ref(grid.x)) * grid.dvolume
            expectation = np.sum(int_exp(grid.x)) * grid.dvolume

        self._veff_lss = lambda x: veff_lss_scale(x) * reference / expectation

        # We must do this otherwise we just get the cached version of _veff_fnc
        del self._veff_fnc

        return self._veff_lss

    def mock_r(self, x, verbose=True):
        """
        Create a random sample of distances given a sample of x.

        Returns
        -------
        r : array-like
            Array of the same length as x given distances to each object.
        """
        # ======================================
        #  find maximum of fg(x,r) = f(x,r)*g(r)
        # ======================================
        def fg(x, r):
            if self.g is not None:
                return self.f(x, r) * self._gnorm(r)
            else:
                return self.f(x, r)

        xseq = np.linspace(self.xmin, self.xmax, 100)
        rseq = np.linspace(self.rmin, self.rmax, 100)
        X, R = np.meshgrid(xseq, rseq)

        def fct(p):
            return -fg(p[0], p[1])

        q = fct((X.flatten(), R.flatten()))  # apply(xrgrid, 1, fct)
        if np.max(q) > 0:
            raise ValueError("f*g can never by smaller than 0.")

        xbegin = X.flatten()[np.argmin(q)]
        rbegin = R.flatten()[np.argmin(q)]

        opt = minimize(
            fct,
            x0=(xbegin, rbegin),
            method="L-BFGS-B",
            bounds=((self.xmin, self.xmax), (self.rmin, self.rmax)),
        )

        fgmax = -opt.fun

        if fgmax > 5 and verbose:
            print(
                "The maximum of f(r)*<g(r)> (=%f) is significantly larger than 1. Check if this is intended."
                % fgmax
            )

        # ============================================
        #  sample distances (r) using cumsum algorithm
        # ============================================
        n = len(x)
        r = np.empty(n)
        dr = min(0.005, (self.rmax - self.rmin) / 1000)
        rgrid = np.arange(self.rmin, self.rmax, dr)
        cdf = np.cumsum(self.dVdr(rgrid))  # cumulative volume out to r
        qnf = spline(cdf, rgrid)  # quantile function of source count density
        lst = np.arange(n)
        m = n
        count = 0
        while m > 0 and count < 100:
            count += 1
            r[lst] = qnf(np.random.uniform(cdf[0], cdf[-1], m))
            rejected = fg(x[lst], r[lst]) < np.random.uniform(size=m) * fgmax
            lst = lst[rejected]
            m = len(lst)

        # sample distances (r) using deterministic uniroot algorithm to avoid iterating forever
        if m > 0:

            def get_random_r(x):
                H = np.vectorize(
                    lambda r: quad(lambda r: fg(x, r) * self.dVdr(r), self.rmin, r)[0]
                )

                def H_inv(y):
                    return brentq(lambda x: H(x) - y, a=self.rmin, b=self.rmax)

                return H_inv(np.random.uniform() * H(self.rmax))

            for i in lst:
                r[i] = get_random_r(x[i])

        return r

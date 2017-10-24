"""
A module primarily defining the :class:`~DFFit` class, which brings together :class:`~Data`, :class:`~._Grid`,
:class:`~.selection.Selection` and :class:`~.model.Model` to perform a parameter fit, using the MML method of Obreschkow et al., (2017).
"""

import attr
import numpy as np
from cached_property import cached_property
from scipy.optimize import minimize
from .utils import numerical_jac, numerical_hess, centres_to_edges, sample_ellipsoid
from .model import Model, Schechter
from scipy.stats import poisson
from .selection import Selection



@attr.s
class Data(object):
    """
    Class representing survey data. This acts as an input to the :class:`~DFFit` class.

    Parameters
    ----------
    x : (N,D)-array
        A DxN matrix (or N-element vector if D=1) containing the observed quantities of N objects (e.g. galaxies).

    x_err :  (N,D,[D])-array, optional
        Specifies the observational errors of ``x``. If ``x_err`` is a ``NxD`` matrix, the elements ``x_err[i,]`` are
        interpreted as the standard deviations of Gaussian uncertainties on ``x[i,]``. In the other case, the ``D-by-D``
        matrices ``x_err[i,,]`` are interpreted as the covariance matrices of the ``D`` observed values ``x[i,]``.

    r : (N,)-array, optional
        Specifies the comoving distances of the N objects (e.g. galaxies). This vector is only needed if
        ``correct_lss_bias = True``.
    """

    def _x_err_converter(val):
        if val is not None:
            if np.all(val == 0):
                return None

            return np.atleast_2d(val).T
        else:
            return None

    x          = attr.ib(convert = lambda x : np.atleast_2d(x).T)
    x_err      = attr.ib(convert = _x_err_converter, default=None,)
    r          = attr.ib(default=None)


    @x.validator
    def _x_validator(self, att, val):

        if len(val) < 1:
            raise ValueError('Give at least one data point.')

        if len(val.shape) > 2:
            raise ValueError('x cannot have more than two dimensions.')

    @x_err.validator
    def _x_err_validator(self, att, val):

        if val is not None:
            if len(val.shape) == 2 and val.shape != self.x.shape:
                raise ValueError('Size of x_err not compatible with size of x.')
            elif len(val.shape) == 3:
                if self.n_dim == 1:
                    raise ValueError('For one-dimensional distribution function x_err cannot have 3 dimensions.')
                if not (val.shape[0] == self.n_data and val.shape[1] == self.n_dim and val.shape[2] == self.n_dim):
                    raise ValueError('Size of x_err not compatible with size of x.')

            elif len(val.shape) > 3 or len(val.shape)==1:
                raise ValueError('x_err cannot have more than three dimensions, has shape: ',val.shape)

            if np.min(val) <= 0:
                raise ValueError('All values of x_err must be positive.')


    @staticmethod
    def _r_converter(val):
        if val is not None:
            return np.array(val)

    @r.validator
    def _r_validator(self,att, val):
        # Handle distance
        if val is not None:
            if len(val) != self.n_data:
                raise ValueError('The number of r values (%s) must be equal to the number of data points (%s).'%(len(val), self.n_data))

            if np.min(val) <= 0:
                raise ValueError('All distance values must be positive.')


    @property
    def n_data(self):
        return self.x.shape[0]

    @property
    def n_dim(self):
        return self.x.shape[1]

    @cached_property
    def invC(self):
        # Make inverse covariances
        invC = np.empty((self.n_data, self.n_dim, self.n_dim))
        if len(self.x_err.shape) == 2:
            if self.n_dim == 1:
                for i in range(self.n_data):
                    invC[i, :, :] = 1 / self.x_err[i, :] ** 2

            else:
                for i in range(self.n_data):
                    invC[i, :, :] = np.diag(1 / self.data.x_err[i, :] ** 2)

        elif len(self.x_err.shape) == 3:
            for i in range(self.n_data):
                invC[i, :, :] = np.linalg.solve(self.x_err[i, :, :])

        return invC


@attr.s
class _Grid(object):
    """
    Class of arrays with numerical evaluations of different functions on a grid in the D-dimensional observable space.
    This grid is used for numerical integrations and graphical representations.

    Note that all attributes listed below are added by the :class:`~DFFit` class, and cannot be accessed otherwise.

    Parameters
    ----------
    xmin, xmax, dx : (P)-vector, optional
        ``P``-element vectors (i.e. scalars for 1-dimensional DF) specifying the points used for some numerical integrations.

    Attributes
    ----------
    gdf : array-like with ``n_points``
        Values of the best-fitting generative DF at each grid point.
    gdf_error_neg, gdf_error_pos : array-like with ``n_points``
        The 68\%-confidence range in the Hessian approximation of the parameter covariances.
    gdf_quantile : (4,n_points)-array
        Quantiles of the generative DF at each grid point.
    Veff : (n_points)-array
        Effective volumes at each grid point.
    scd : (n-points)-array
       Predicted source counts according to the best-fitting model
    scd_posterior : (n_points)-array
        Oserved source counts derived from the posterior PDFs of each object.
    effective_counts : (n_points)-array
        Fractional source counts derived from the posterior PDFs of each object
    """
    dx = attr.ib(convert=lambda x : np.atleast_1d(np.array(x)))
    xmin = attr.ib(convert = lambda x : np.atleast_1d(np.array(x)))
    xmax = attr.ib(convert = lambda x : np.atleast_1d(np.array(x)))

    @xmax.validator
    def _xmax_validator(self, att, val):
        if np.any(val < self.xmin + self.dx):
            raise ValueError('xmax cannot be smaller than xmin+dx.')

        if val.size != self.xmin.size or val.size != self.dx.size or self.xmin.size != self.dx.size:
            raise ValueError("xmax, xmin and dx must be of the same length")

    @cached_property
    def _xgrid(self):
        "Grid of x in each dimension, shape (N,D)"
        return [np.arange(xmin, xmax, dx) for (xmin, xmax, dx) in zip(self.xmin, self.xmax, self.dx)]

    @cached_property
    def _nx(self):
        "The length of each grid dimension"
        return [len(x) for x in self._xgrid]

    @cached_property
    def n_points(self):
        "Number of grid points total."
        return np.product(self._nx)

    @cached_property
    def x(self):
        "Full mesh of x in each dimension, derived via meshgrid. Note that for performance, we use copy=False and sparse=True."
        return np.meshgrid(*self._xgrid, sparse=True, copy=False)

    @cached_property
    def dvolume(self):
        "The volume of a grid cell"
        return np.product(self.dx)


@attr.s
class Posteriors(object):
    """
    Specifies the posterior PDFs of the observed data, given the best-fitting model.

    Parameters
    ----------
    x_mean : (N,D)-array
        Gives the D-dimensional means of the posterior PDFs of the N objects.
    x_mode : (N,D)-array
        Gives the D-dimensional modes of the posterior PDFs of the N objects.
    x_stdev : (N,D)-array
        Gives the D-dimensional standard deviations of the posterior PDFs of the N objects.
    x_random : (N,D)-array
        Gives one random D-dimensional value drawn from the posterior PDFs of each of the N objects.
    """
    x_mean = attr.ib()
    x_mode = attr.ib()
    x_stdev = attr.ib()
    x_random = attr.ib()


@attr.s
class Fit(object):
    """
    Describes the fitted generative distribution function. Its most important entries are:

    Parameters
    ----------
    p_best : tuple
        A P-tuple giving the most likely model parameters according to the MML method.
    p_covariance : (P,P)-array
        The covariance matrix of the best-fitting parameters in the Gaussian approximation from the Hessian matrix of the modified likelihood function.
    gdf : callable
        A function of a D-dimensional vector, which is the generative DF, evaluated at the parameters ``p_best``
    scd : callable
        A function of a D-dimensional vector, which gives the predicted source counts of the most likely model,
        i.e. ``scd(x)=gdf(x)*Veff(x)``
    """
    p_best = attr.ib()
    p_covariance = attr.ib()
    lnL = attr.ib()
    status = attr.ib()
    ln_evidence = attr.ib()
    gdf_ = attr.ib()
    veff_ = attr.ib()
    opt = attr.ib()

    @property
    def p_sigma(self):
        "The standard deviation of the best-fitting parameters in the Gaussian approximation from the Hessian matrix of the modified likelihood function"
        return np.sqrt(np.diag(self.p_covariance))

    def gdf(self, x):
        return self._gdf(x, self.p_best)

    def scd(self, x):
        return self._gdf(x, self.p_best) * self._veff(x)


@attr.s
class DFFit(object):
    """
    A distribution-function fit object.

    This object contains all the attributes and methods necessary to perform an MML fit for the parameters of a given
    model to given data, as well as several methods for resampling and bias correction. The object can be used as
    input to plotting methods found in :module:`~.plotting`.

    Parameters
    ----------
    data : :class:`~Data` instance
        Specifies the data to be fit.
    selection : :class:`~.selection.Selection` instance
        An object defining a selection function appropriate for the data. See :module:`pydftools.selection` for more info.
    grid_dx : float, optional
        Specifies grid resolution for all grid integrations.
    model : :class:`~Model` instance
        Specifies the model used in the fit.
    n_iterations : int, optional
        Maximum number of iterations in the repeated fit-and-debias algorithm to evaluate the maximum likelihood.
    keep_eddington_bias : bool, optional
        If ``True``, the data is not corrected for Eddington bias. In this case no fit-and-debias iterations are
        performed and the argument ``n_iterations`` will be ignored.
    correct_lss_bias : bool, optional
        If ``True`` the ``distance`` values are used to correct for the observational bias due to galaxy clustering
        (large-scale structure). The overall normalization of the effective volume is chosen such that the expected mass
        contained in the survey volume is the same as for the uncorrected effective volume.
    ignore_uncertainties : bool, optional
        If ``True``, treat data as if it had no ``x_err``. This is subtly different from ``keep_eddington_bias=True``,
        as it enforces a uniform prior on the masses.
    lss_weight : callable, optional
        If ``correct_lss_bias=True``, this optional function of a ``P``-vector is the weight-function used for the mass
        normalization of the effective volume. For instance, to preserve the number of galaxies, choose
        ``lss_weight = lambda x : 1``, or to preserve the total mass, choose ``lss_weight = lambda x : 10**x``
        (if the data ``x`` are log10-masses).

    Notes
    -----
    For a detailed description of the method, please refer to the peer-reviewed publication by Obreschkow et al. 2017
    (in prep.).


    Examples
    --------
    First, generate a mock sample of 1000 galaxies with 0.5dex mass errors, drawn from
    a Schechter function with parameters (-2,11,-1.3):

    >>> import pydftools as df
    >>> data, selection, model, other = dfmockdata(n=1000, sigma=0.5)

    Fit a Schechter function to the mock sample without accounting for errors

    >>> survey1 = df.DFFit(data=data, selection=selection, model=model, ignore_uncertainties=True)

    Plot fit and add a black dashed line showing the input MF

    >>> df.mfplot(survey1, xlim=c(1e6,2e12), ylim=c(2e-4,2), p = model.p0)

    Now, do the same again, while accountting for measurement errors in the fit
    This time, the posterior data, corrected for Eddington bias, is shown as black points

    >>> survey2 = df.DFFit(data=data, selection=selection, model=model)
    >>> mfplot(survey2,xlim=c(1e6,2e12), ylim=c(2e-4,2), p = model.p0)

    Now create a smaller survey of only 30 galaxies with 0.5dex mass errors

    >>> data, selection, model, other = dfmockdata(n=30, sigma=0.5)

    Fit a Schechter function and determine uncertainties by resampling the data

    >>> survey = df.DFFit(data=data, selection=selection, model=model)
    >>> survey.resample(n_bootstrap=30)

    Show best fit with 68% Gaussian uncertainties from Hessian and posterior data as black points

    >>> df.mfplot(survey, uncertainty_type = 1, p = model.p0)

    Show best fit with 68% and 95% resampling uncertainties and posterior data as black points

    >>> df.mfplot(survey, uncertainty_type = 3, p = model.p0)
    """

    data = attr.ib()
    selection = attr.ib()
    grid_dx = attr.ib(default = 0.05, convert = np.atleast_1d)
    model = attr.ib(default= Schechter())

    n_iterations        = attr.ib(default = 100,  convert = int)
    keep_eddington_bias = attr.ib(default = False, convert = bool)
    correct_lss_bias    = attr.ib(default = False, convert = bool)
    ignore_uncertainties= attr.ib(default = False, convert=bool)
    lss_weight = attr.ib(default=lambda x : 10**x, validator=[attr.validators.optional(lambda s, a, v: callable(v))])

    def __attrs_post_init__(self):
        if self.data.x_err is None:
            self.ignore_uncertainties = True

    @n_iterations.validator
    def _n_iterations_validator(self, att, val):
        if val < 1:
            raise ValueError('n_iterations must be a positive integer.')


    @data.validator
    def _data_validator(self, att, val):
        if not (hasattr(val, "x") and hasattr(val, "x_err") and hasattr(val, "r")):
            raise ValueError("data must be a Data instance")

        if val.x_err is not None:
            if len(val.x_err) == 2:
                if np.any(np.min(val.x - val.x_err, axis=1) < self.grid.xmin):
                    raise ValueError('xmin cannot be larger than smallest observed value x - x_err')
                if np.any(np.max(val.x + val.x_err, axis=1) > self.grid.xmax):
                    raise ValueError('xmax cannot be smaller than largest observed value x + x_err')
            elif len(val.x_err) == 3:
                xerr = np.einsum("ijj -> ij", val.x_err)

                if np.any(np.min(val.x - xerr, axis=1) < self.grid.xmin):
                    raise ValueError('xmin cannot be larger than smallest observed value x - x_err')
                if np.any(np.max(val.x + xerr, axis=1) > self.grid.xmax):
                    raise ValueError('xmax cannot be smaller than largest observed value x + x_err')

    @grid_dx.validator
    def _grid_dx_validator(self, att, val):
        if np.any(val < 0):
            raise ValueError("grid_dx must be greater than 0")
        if val.size != self.data.n_dim:
            raise ValueError('dx must be a D-element vector, where D is the number of columns of x.')

    @model.validator
    def _model_validator(self, att, val):
        if not isinstance(val, Model):
            raise ValueError("model must be a Grid instance")


    @selection.validator
    def _selection_validator(self, att, val):
        if not isinstance(val, Selection):
            raise ValueError("The selection function must be a subclass of the Selection class.")

    @cached_property
    def grid(self):
        "Contains the gridded evaluations of various functions used for integration throughout."
        return _Grid(dx = self.grid_dx, xmin = self.selection.xmin, xmax = self.selection.xmax)

    @cached_property
    def rho_observed(self):
        d = np.array([np.add.outer(-xgrid, xval) for xval, xgrid in
                      zip(self.data.x.T, self.grid.x)]).T  # has shape (N, Ngrid, Ndim)
        rho_observed = np.exp(-np.einsum("ijk, ikl, ijl -> ij", d, self.data.invC, d) / 2)
        # TODO: is it correct to leave off the normalisation?

        # if the xrange is very far from xobs, set probability=1 at closest point
        bad_indx = np.sum(rho_observed, axis=1) < 0.01
        if np.any(bad_indx):
            indices = np.argmin(np.sum(d[bad_indx] ** 2, axis=2), axis=1)
            rho_observed[bad_indx][indices] = 1.
        return rho_observed

    def rho_corrected(self, p):
        if self.keep_eddington_bias:
            prior = np.ones(self.grid.n_points)
        else:
            # predicted source counts (up to a factor x.mesh.dv)
            prior = self.model.gdf(*self.grid.x, p=p) * self.grid.veff
            prior[np.isinf(prior)] = 0
            np.clip(prior, 0, np.inf, out=prior)

        rho_corrected = (prior * self.rho_observed).T  # shape npoints, ndata
        rho_corrected /= np.sum(rho_corrected, axis=0) * self.grid.dvolume

        return rho_corrected.T

    def rho_unbiased(self, p):
        if self.ignore_uncertainties:
            # Assign each obs to its nearest grid point
            rho_unbiased = np.histogramdd(self.data.x, bins=[centres_to_edges(x) for x in self.grid.x])[0]
            rho_unbiased /= self.grid.dvolume
        else:
            rho_corrected = self.rho_corrected(p)
            rho_unbiased = np.sum(rho_corrected, axis=0)

        return rho_unbiased

    def scd(self, x, p):
        "The expected source-count distribution for property x with parameters p."
        return self.selection.Veff(x) * self.model.gdf(x, p)

    # def jacobian(self, p0, p):
    #     """
    #     The jacobian of the MML likelihood for a debias iteration with "guess" p0, and at parameters p.
    #
    #     Parameters
    #     ----------
    #     p0
    #     p
    #
    #     Returns
    #     -------
    #     """
    #     scd_p = self.scd(*self.grid.x, p)
    #     scd_p0 = self.scd(*self.grid.x, p0)
    #     scd_jac = self.selection.Veff(*self.grid.x) * self.model.gdf_jacobian(*self.grid.x, p)
    #     first_term = -np.sum(scd_jac,axis=1)*self.grid.dvolume
    #     numerator = np.einsum("ij, kj, j,j -> ki", self.rho_observed, scd_jac ,scd_p0,1./scd_p) * self.grid.dvolume
    #     denom = np.sum(self.rho_observed * scd_p0, axis=1) * self.grid.dvolume
    #
    #     second_term = np.sum(numerator/denom, axis=1)
    #     return first_term + second_term

    def hessian(self,p):
        "Determines the Hessian of the MML likelihood, for p0 = p."
        if not self.correct_lss_bias:
            self.grid.veff = self.selection.Veff(*self.grid.x)
        elif self.correct_lss_bias and self.selection.g is None:
            self.selection._get_veff_lss(self.data.r, self.grid, p, self.model,
                                         weight=self.lss_weight if self.lss_weight is not None else lambda
                                             x: np.ones_like(x))
            self.grid.veff = self.selection.Veff(*self.grid.x)

        # make unbiased source density function
        rho_unbiased = self.rho_unbiased(p)

        # make -ln(L)
        def neglogL(p):
            phi = self.model.gdf(*self.grid.x, p=p)
            # safety operations (adding about 50% computation time)
            phi[np.isinf(phi)] = 0
            mask = phi > 0

            # end safety operations
            # print(p, phi)
            lnl = np.sum(
                phi[mask] * self.grid.veff[mask] - np.log(phi[mask]) * rho_unbiased[mask]) * self.grid.dvolume
            return lnl

        return -numerical_hess(neglogL, p)

    def covariance(self, p):
        "Gaussian covariance of the MML likelihood at p0 = p."
        return -np.linalg.inv(self.hessian(p))

    @cached_property
    def fit(self):
        """
        Perform the actual MML fit.

        This function finds the most likely P-dimensional model parameters of a D-dimensional distribution function (DF)
        generating an observed set of N objects with D-dimensional observables x, accounting for measurement uncertainties
        and a user-defined selection function. For instance, if the objects are galaxies, \code{dffit} can fit a mass
        function (D=1), a mass-size distribution (D=2) or the mass-spin-morphology distribution (D=3). A full description of
        the algorithm can be found in Obreschkow et al. (2017).

        Parameters
        ----------
        self

        Returns
        -------

        """
        # Input handling
        if not self.correct_lss_bias:
            self.grid.veff = self.selection.Veff(*self.grid.x)

            if self.ignore_uncertainties:
                self.n_iterations = 1

        # Iterative algorithm
        running = True
        offset = 0
        chain = np.empty((self.n_iterations, len(self.model.p0) + 1))
        k = 0
        p0 = self.model.p0
        while running:

            # determine Veff LSS
            if self.correct_lss_bias and self.selection.g is None:
                self.selection._get_veff_lss(self.data.r, self.grid, p0, self.model,
                                             weight = self.lss_weight if self.lss_weight is not None else lambda x : np.ones_like(x))

                self.grid.veff = self.selection.Veff(*self.grid.x)

            # make unbiased source density function
            rho_unbiased = self.rho_unbiased(p0)

            # make -ln(L)
            def neglogL(p):
                phi = self.model.gdf(*self.grid.x, p=p)
                # safety operations (adding about 50% computation time)
                phi[np.isinf(phi)] = 0
                mask = phi >0

                # end safety operations
                lnl = np.sum(phi[mask] * self.grid.veff[mask] - np.log(phi[mask]) * rho_unbiased[mask]) * self.grid.dvolume - offset
                return lnl


            # test
            if np.all(np.isinf(self.model.gdf(*self.grid.x, p=p0))):
                raise RuntimeError('cannot evaluate GDF at initial parameters provided')
            try:
                test = neglogL(p0)
                if np.isinf(test):
                    raise RuntimeError('cannot evaluate likelihood at initial parameters provided')
            except Exception as e:
                raise e

            # maximize ln(L) (NOTE: this is set up to match the R version)
            opt_options = {"maxiter":100000}
            if self.ignore_uncertainties:
                # Coarse binning when x_err is None requires different solver that doesn't use jacobian.
                method = "Nelder-Mead"
            else:
                method = "BFGS"

            opt = minimize(neglogL, p0, method = method, options=opt_options)
            offset += opt.fun
            chain[k] = np.concatenate((opt.x, [opt.fun]))

            # assess convergence
            if ( self.ignore_uncertainties and not self.correct_lss_bias) or self.keep_eddington_bias:
                converged = opt.success
                running = False
            else:
                # asses convergence
                if k == 0:
                    converged = False
                    d_old = np.inf
                else:
                    d = np.abs(opt.fun-value_old)
                    converged = d >= d_old
                    d_old = d *1

                value_old = 1.*opt.fun

                if converged:
                    running = False
                elif k == self.n_iterations-1:
                    converged = False
                    running = False
                    print('WARNING: Maximum number of iteration reached. Consider increasing n.iterations and/or providing better initial parameters.')

                # prepare initial values for next iteration
                p0 = opt.x
                k += 1

        # make output
        cov = self.covariance(opt.x)
        if np.linalg.det(cov) > 1e12:
            converged = False
            cov = None
            print('WARNING: Fit ill-conditioned. Consider providing better initial parameters or selection arguments.')

            ln_evidence = False
        else:
            n_para = len(opt.x)
            ln_evidence = -offset + 0.5 * n_para * np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(cov))

        if self.correct_lss_bias:
            self.selection._get_veff_lss(self.data.r, self.grid, opt.x, self.model,
                                         weight=self.lss_weight if self.lss_weight is not None else lambda
                                             x: np.ones_like(x))

        fit = Fit(p_best = opt.x, p_covariance = cov, lnL = lambda p : -neglogL(p),
                  opt = opt,
                  status = dict(n_iterations = k,
                                 converged = converged,
                                 chain = chain[:k]),
                   ln_evidence = ln_evidence,
                   gdf_ = self.model.gdf,
                   veff_ = self.selection.Veff
                   )

        # UPDATE GRID
        self.grid.gdf = self.model.gdf(*self.grid.x, p=opt.x)
        self.grid.veff = self.selection.Veff(*self.grid.x)
        self.grid.scd = self.grid.gdf * self.grid.veff

        # finalize output
        return fit


    @cached_property
    def _gaussian_errors(self):
        cov = self.fit.p_covariance
        eigvals, eigvectors = np.linalg.eig(cov)
        npar = self.model.n_param
        nx = self.grid.n_points

        # sample surface of covariance ellipsoid
        nsteps = 500
        p_new = sample_ellipsoid(cov, nsteps, add_boundaries=True, mean =self.fit.p_best)
        y_new = np.zeros((self.grid.n_points, len(p_new)))
        for i,p in enumerate(p_new):
            y_new[:,i] = self.model.gdf(*self.grid.x, p)

        # y_new = np.empty((nx, nsteps + 2 * npar))
        # for i in range(nsteps + 2 * npar):
        #     if i <= nsteps-1:
        #         e = np.random.normal(size=npar)
        #         e /= np.sqrt(np.sum(e ** 2))
        #     else:
        #         e = np.zeros(npar)
        #         if i >= nsteps+npar:
        #             e[i+1-nsteps-np] = 1
        #         else:
        #             e[i+1-nsteps] = -1
        #
        #
        #     v = np.matmul(eigvectors, np.sqrt(eigvals) * e)
        #     p_new = self.fit.p_best+v
        #     y_new[:, i] = self.model.gdf(self.grid.x, p_new)

        self.grid.gdf_gaussian_min = np.nanmin(y_new, axis=1) #np.nanmin([y_new, np.inf])
        self.grid.gdf_gaussian_max = np.nanmax(y_new, axis=1) #- self.grid.gdf

        return self.grid.gdf_gaussian_min, self.grid.gdf_gaussian_max

    @cached_property
    def gdf_gaussian_min(self):
        return self._gaussian_errors[0]

    @cached_property
    def gdf_gaussian_max(self):
        return self._gaussian_errors[1]

    def _refit_to_new_sample(self,n, do_jackknife=False, lss_errors=True):
        if not self.fit.status['converged']:
            print("The fit did not converge, and therefore resampling cannot be performed.")

        # input handling
        n_data = self.data.n_data
        npar = self.model.n_param

        if do_jackknife:
            n = min(n, n_data)

        if n_data < 3:
            raise ValueError('Resampling/Jackknifing requires at least three objects.')

        if n < 2:
            raise ValueError("Resampling/Jackknifing requires at least 2 iterations")

        # set up resample survey
        # Copy current object into new one, sharing most things.

        # randomly resample and refit the DF
        p_new = np.empty((n, npar))

        if do_jackknife:
            reject = np.random.choice(n_data, size= n, replace=False)

        for iteration in range(n):

            #print('Resampling: %4.2f'%(float(iteration) / n))

            if not do_jackknife:
                n_data = max(2, np.random.poisson(self.data.n_data))
                s = np.random.randint(0,self.data.n_data-1, size=n_data)
            else:
                s = np.arange(n_data) != reject[iteration]

            x = self.data.x[s]

            if self.data.x_err is not None:
                x_err = self.data.x_err[s]
            else:
                x_err = None

            if self.data.r is not None:
                r = self.data.r[s]
            else:
                r = None

            b = DFFit(data = Data(x=x.flatten(), x_err=np.squeeze(x_err), r=r),
                      selection = self.selection,
                      grid_dx = self.grid_dx,
                      model = self.model,
                      n_iterations = self.n_iterations,
                      keep_eddington_bias = self.keep_eddington_bias,
                      correct_lss_bias    = self.correct_lss_bias and lss_errors,
                      lss_weight = self.lss_weight)

            b.model.p0 = self.fit.p_best

            p_new[iteration] = b.fit.p_best
        return p_new

    def resample(self, n_bootstrap=30, lss_errors=True):
        """
        Performs a bootstrapping of the sample to provide a better covariance estimate.

        The data is resampled ``n_bootstrap`` times using a non-parametric
        bootstrapping method to produce more accurate covariances.

        Parameters
        ----------
        n_bootstrap: int,optional
            Number of bootstrapping iterations.

        lss_errors : bool, optional
            A logical flag specifying whether uncertainties computed via resampling should include errors due to the
            uncertainty of large-scale structure (LSS). If ``True` the parameter uncertainties are estimated by refitting
            the LSS correction at each resampling iteration. This argument is only considered if ``correct_lss_bias=True``
            and ``n_bootstrap>0``.

        Notes
        -----
        This routine does not return anything, but rather adds properties to the object. Importantly, it adds
        :attr:`~.fit.p_covariance_resample` along with :attr:`~.fit.p_quantile` and :attr:`~.grid.gdf_quantile`.
        """
        p_new = self._refit_to_new_sample(n_bootstrap, lss_errors=lss_errors)

        # compute covariance
        self.fit.p_covariance_resample = np.cov(p_new.T)

        # make parameter quantiles
        q = [2., 16, 84., 98.]
        self.fit.p_quantile = np.percentile(p_new, q, axis=0)

        # make DF quantiles
        s = np.empty((n_bootstrap, self.grid.n_points))
        for i in range(n_bootstrap):
            s[i] = self.model.gdf(*self.grid.x, p_new[i])

        y_quant = np.empty((4, self.grid.n_points))
        for i in range(self.grid.n_points):
            lst = np.logical_and(np.logical_not(np.logical_and(np.isnan(s[:,i]), np.isfinite(s[:,i]))), s[:,i]>0)
            y_quant[:, i] = np.percentile(s[lst, i], q)

        self.grid.gdf_quantile = y_quant

    def jackknife(self, n_jackknife=30, lss_errors=True):
        """
        Perform a jack-knife resampling to account for bias in the estimator.

        The data is jackknife-resampled ``n_jackknife`` times,
        removing exactly one data point from the observed set at each iteration. This resampling adds model parameters,
        maximum likelihood estimator (MLE) bias corrected parameter estimates (corrected to order 1/N).

        Parameters
        ----------
        n_jackknife :  int
            The number of re-samplings to perform. If ``n_jackknife`` is larger than the number of data points N,
            it is automatically reduced to N.

        lss_errors : bool, optional
            A logical flag specifying whether uncertainties computed via resampling should include errors due to the
            uncertainty of large-scale structure (LSS). If ``True` the parameter uncertainties are estimated by refitting
            the LSS correction at each resampling iteration. This argument is only considered if ``correct_lss_bias=True``
            and ``n_bootstrap>0``.

        Notes
        -----
        This routine does not return anything, but rather adds properties to the object. Importantly, it adds
        :attr:`~.fit.p_covariance_jackknife`.
        """
        p_new = self._refit_to_new_sample(n_jackknife, do_jackknife = True, lss_errors=lss_errors)

        # estimate covariance
        n_data = self.data.n_data
        npar = self.model.n_param
        ok = np.sum(p_new,axis=1) != np.nan
        cov_jn = np.cov(p_new[ok]) * (n_data - 1)

        # compute poisson covariance
        jn = DFFit(data=self.data,
                  selection=self.selection,
                  grid_dx=self.grid_dx,
                  model=self.model,
                  n_iterations=self.n_iterations,
                  keep_eddington_bias=self.keep_eddington_bias,
                  correct_lss_bias=self.correct_lss_bias and lss_errors,
                  lss_weight=self.lss_weight)

        jn.model.p0 = self.fit.p_best
        jn.options.n_iterations = 1

        q = [.16, 0.5, 0.84]
        p_pois = np.empty((3,npar))
        for i in range(3):
            n_new = poisson.ppf(q[i], n_data)
            jn.grid.veff = self.grid.veff * n_new / n_data
            p_pois[i] = jn.fit.p_best

        cov_pois = np.cov(p_pois)

        # estimate combined covariance
        if np.isnan(cov_pois[0,0]):
            self.fit.p_covariance_jackknife = cov_jn
        else:
            self.fit.p_covariance_jackknife = cov_jn+cov_pois
        

        # correct estimator bias
        p_reduced = np.nanmean(p_new, axis=1)
        self.fit.p_best_mle_bias_corrected = n_data * self.fit.p_best - (n_data - 1) * p_reduced
        self.fit.gdf_mle_bias_corrected = lambda x : self.model.gdf(x, self.fit.p_best_mle_bias_corrected)
        self.fit.scd_mle_bias_corrected = lambda x : self.fit.gdf_mle_bias_corrected(x) * self.selection.Veff(x)
        self.grid.gdf_mle_bias_corrected = self.fit.gdf_mle_bias_corrected(self.grid.x)
        self.grid.scd_mle_bias_corrected = self.fit.scd_mle_bias_corrected(self.grid.x)


    @cached_property
    def posterior(self):
        
        if self.ignore_uncertainties:
            return None

        # Input handling
        x = self.data.x
        x_mesh = self.grid.x
        x_mesh_dv = self.grid.dvolume
        n_data = x.shape[0]
        n_dim = x.shape[1]

        # produce posteriors
        m0 = np.empty((n_data, n_dim))
        m1 = np.empty((n_data, n_dim))

        # make posterior PDF for data point i
        rho_corrected = self.rho_corrected(self.fit.p_best)
        s = np.sum(rho_corrected, axis=1) #shape ndata
        rho_unbiased = self.rho_unbiased(self.fit.p_best)
        rho_unbiased_sqr = np.sum((rho_corrected.T / (s * x_mesh_dv)) ** 2, axis=1)

        # mean, standard deviation and mode
        for j in range(n_dim):
            m0[:, j] = np.sum(x_mesh[j] * rho_corrected, axis=1) / s
            m1[:, j] = np.sqrt(np.sum(np.add.outer(-m0[:, j], x_mesh[j]) ** 2 * rho_corrected, axis=1) / s)

        a = np.argmax(rho_corrected, axis=1)
        md = np.array([xj[a] for xj in x_mesh]).T

        posterior = Posteriors(x_mean = m0, x_stdev = m1, x_mode = md, x_random = m0 + m1 * np.random.normal(size = (n_data,n_dim)))
        self.grid.scd_posterior = rho_unbiased
        self.grid.effective_counts = rho_unbiased ** 2 / rho_unbiased_sqr  # this equation gives the effective number of sources per bin
        self.grid.effective_counts[np.isinf(self.grid.effective_counts)] = 0

        return posterior
    
    def fit_summary(self, format_for_notebook=False):

        p = self.fit.p_best

        string = ""
        br = "<br>" if format_for_notebook else "\n"
        if self.model.gdf_equation is not None:

            string += '%s%s'% (self.model.gdf_equation,br*2)


#        if format_for_notebook:
#            string += "\n```\n"

        if not self.fit.status['converged']:
            for i in range(len(p)):
                string += '%s = %7.2f (not converged)%s'%(self.model.names[i], p[i],br)
        else:
            if hasattr(self.fit, "p_quantile"):
                sigma_84 = self.fit.p_quantile[2]-self.fit.p_best
                sigma_16 = self.fit.p_best-self.fit.p_quantile[1]
                for i in range(len(p)):
                    string += '%s = %8.3f (+%4.3f -%4.3f)%s'%(self.model.names[i], p[i], sigma_84[i], sigma_16[i],br)

            else:
                sigma = self.fit.p_sigma
                for i in range(len(p)):
                    string += '%s = %8.3f (+-%4.3f)%s'%(self.model.names[i], p[i], sigma[i],br)

#        if format_for_notebook:
#            string += "\n```"

        return string
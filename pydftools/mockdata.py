"""
A module defining a function :func:`~mockdata` which is able to generate 1D mock data given a model, selection function
and several other parameters.
"""

from .model import Schechter
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from .dffit import Data
from .selection import SelectionRdep


def mockdata(n=None, seed=None,  model = Schechter(), selection = None,
             p=None,  sigma = 0, shot_noise = False, verbose = False):
    """
    Generate 1D mock data.

    This function produces a mock survey with observed log-masses with Gaussian uncertainties
    and distances, using a custom mass function and selection function.

    Parameters
    ----------
    n : int, optional
        Number of objects (galaxies) to be generated. If None, the number is determined from the mass function model
        and the selection criteria (specified by ``f`` and ``dVdr``). Otherwise, the survey volume
        (specified by the derivative ``dVdr``) is automatically multiplied by the scaling factor required to obtain the
        requested number of objects ``n``.
    seed : int, optional
        Used as seed for the random number generator. If you wish to generate different realizations, with the same
        survey specifications, it suffices to vary this number.
    model : :class:`~model.Model` intance
        Defines the 'generative distribution function', i.e. the underlying mass function, from which the galaxies are drawn.
    selection : :class:`~.selection.Selection` instance
        Defines the selection function. Any sub-class of :class:`~.selection.Selection` may be used. See docstrings for more info.
    p : tuple, optional
        Model parameters for the `model`.
    sigma : scalar or array-like
        Gaussian observing errors in log-mass ``x``, which are automatically added to the survey. If array-like, sigma
        must be equal to or longer than the number of samples, ``n``.
    shot_noise : bool, optional
        Whether the number of galaxies in the survey can differ from the expected number, following a Poisson distribution.
    verbose : bool, optional
        Whether information will be displayed in the console while generating the mock survey.

    Returns
    -------
    data : :class:`~.dffit.Data` instance
        An instance of :class:`~.dffit.Data` ready to be passed to the fitting routine.
    selection : :class:`~.selection.Selection` instance
        An instance of :class:`~.selection.Selection` containing selection function quantities. Note, this is not to be passed to
        :class:`~dfft.DFFit`, as in real situations, these quantities are unknown.
    model : :class:`~.model.Model` instance
        A :class:`~.model.Model` instance defining the generative distribution used in this function (which can be directly passed
        to :class:`~.dffit.DFFit` to fit to the mock data).
    other : dict
        A dictionary containing the following entries:

        * scd: function returning the expected source count density as a function of log-mass ``x``.
        * rescaling_factor: value of rescaling factor applied to the cosmic volume to match the requested number of galaxies ``n``.
        * n: number of galaxies in sample
        * n_expected: expected number of galaxies in volume.
        * p: parameters of the model used.
        * dx: grid spacing used
        * x_true: the true log-masses (before scattering by uncertainty).

    Examples
    --------
    Draw 1000 galaxies with mass errors of 0.3 dex from a Schechter function
    with parameters (-2,11,-1.3) and a preset selection function

    >>> import pydftools as df
    >>> data, selection, model, other = df.mockdata(n = 1000, sigma = 0.3)

    Plot the distance-log(mass) relation of observed data, true data, and approximate survey limit

    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(data.r,data.x,color='blue')
    >>> plt.scatter(data.r,other['x_true'],color='green')
    >>> x = np.arange(5,11,0.01)
    >>> plt.plot(1e-2*sqrt(10**x),x,color='red')

    These data can then be used to fit a MF in several ways. For instance,
    assuming that the effective volume function Veff(x) is known:

    >>> selection_veff = df.selection.SelectionVeff(veff=selection.Veff)
    >>> survey = df.DFFit(data=data, selection=selection, model=model)

    Or assuming that Veff is known only on a galaxy-by-galaxy basis

    >>> selection_pts = df.selection.SelectionVeffPoints(xval = data.x, veff = selection.Veff(data.x))
    >>> survey = df.DFFit(data=data, selection=selection_pts,model=model)

    Or assuming that Veff is known on a galaxy-by-balaxy basis, but approximate analytically
    outside the range of observed galaxy masses

    >>> selection_pts_fnc = df.selection.SelectionVeffPoints(xval = data.x, veff = selection.Veff(data.x), veff_extrap=selection.Veff)
    >>> survey = df.DFFit(data=data, selection=selection_pts_fnc,model=model)

    Or assuming that the full selection function f(x,r) and the observing volume
    derivative dVdr(r) are known

    >>> survey = df.DFFit(data=data, selection=selection,model=model)
    """
    # Set default p
    if p is None:
        p = model.p0
    if selection is None:
        selection = SelectionRdep(xmin=4.0, xmax=13.0, rmin=0, rmax=20)

    # Check whether the model can be evaluated at p
    try:
        test = model.gdf(selection.xmin, p)
    except Exception as e:
        raise e

    if np.isinf(test):
        raise ValueError('model cannot be evaluated for parameter-vector p.')

    if seed:
        np.random.seed(seed)

    # Generate source count function (including LSS if present)
    scd = lambda x : selection.Veff(x) * model.gdf(x, p)


    # compute expected number of galaxies (accounting for lss if present)
    n_expected = quad(scd, selection.xmin, selection.xmax)[0]
    n_expected_large = quad(scd, 2 * selection.xmin - selection.xmax, 2 * selection.xmax - selection.xmin)[0]
    if n_expected_large > 1.001 * n_expected:
        raise ValueError('A non-negligible number of galaxies lies outside the range xmin-xmax. Please change this range.')

    # rescale effective volume to match the (optional) requested number of galaxies
    if n is None:
        if n_expected < 2:
            raise ValueError('Input arguments imply less than two sources in the survey.')
        rescaling_factor = 1
    else:
        if n < 2:
            raise ValueError('Number of sources must be at least 2.')

        rescaling_factor = n / n_expected
        selection.vol_renorm *= rescaling_factor
        n_expected = n


    # make actual number of sources
    if shot_noise:
        n = int(max(1, np.random.poisson(n_expected)))
    else:
        n = int(round(n_expected))

    if verbose:
        print('Number of sources in the mock survey (expected): %.3f'% n_expected)
        print('Number of sources in the mock survey (selected): %d'% n)

    # sample masses (x)
    dx = min(0.005, (selection.xmax-selection.xmin) / 1000.)
    xgrid = np.arange(selection.xmin, selection.xmax, dx)
    cdf = np.cumsum(scd(xgrid))  # cumulative distribution function of source count density

    if cdf[-2]==cdf[-1]:
        indxu = np.where(cdf == cdf[-1])[0][0] # only interpolate up to where cdf stops rising, otherwise errors occur
    else:
        indxu = len(cdf)-1

    if cdf[1]==cdf[0]:
        indxl = np.where(cdf == cdf[0])[0][-1] # only interpolate up to where cdf stops rising, otherwise errors occur
    else:
        indxl = 0

    qnf = spline(cdf[indxl:indxu], xgrid[indxl:indxu])  # quantile function of source count density
    x = qnf(np.random.uniform(cdf[0], cdf[-1], size=n))

    # add mass observing errors (x.err)
    if sigma is not None:
        if hasattr(sigma,"__len__"):
            if len(sigma) < n:
                raise ValueError('If sigma is a vector its too short.')
            else:
                x_err = sigma[:n]
        else:
            x_err = np.repeat(sigma, n)

        x_obs = x + np.random.normal(size=n) * x_err
    else:
        x_obs = x
        x_err = None

    # make effective volumes for each observation, that an observer would assign, not knowning the observational error in x
    # veff_values = selection.Veff(x_obs)

    if hasattr(selection, "mock_r"):
        r = selection.mock_r(x, verbose=verbose)
    else:
        r = None

    # # If LSS is supplied, we need to remove that for the output selection function, becau
    # if isinstance(selection, SelectionRdep) and selection.g is not None:
    #     selection = SelectionRdep(xmin = selection.xmin, xmax = selection.xmax,
    #                               rmin = selection.rmin, rmax = selection.rmax,
    #                               f = selection.f, dvdr = selection.dvdr,
    #                               vol_renorm = selection.vol_renorm)

    return (
        Data(
            x = x_obs,
            x_err = x_err,
            r = r
        ),
        selection,
        model,
        # Grid(
        #     xmin = xmin,
        #     xmax = xmax,
        #     dx = dx
        # ),
        dict(
            x_true = x,
            dx = dx,
            scd = scd,
            p = p,
            rescaling_factor = rescaling_factor,
            n = n,
            n_expected = n_expected
        )
    )




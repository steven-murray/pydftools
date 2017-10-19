"""
A module defining various plotting functions that generally act on a :class:`~dffit.DFFit` instance.

The primary function for general use is :func:`~mfplot`, which is a wrapper around :func:`~dfplot`, and shows a fitted
mass function optionally with uncertainty region, and binned data, along with a histogram of data counts.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson
from chainconsumer import ChainConsumer
import warnings

def dfplot(dffit, xlab=r'Observable $x$', ylab=r'Generative distribution function, $\phi$',
           ylab_histogram = 'Counts',
           fit_label=None,
           xlim = None, ylim = None, p_true = None, xpower10 = False,
           show_input_data = True, show_posterior_data = True,  show_data_histogram = True,
           uncertainty_type = 1, show_bias_correction = True, nbins = None,
           bin_xmin = None, bin_xmax = None,
           col_fit = 'blue', lw_fit = 2, ls_fit = '-',
           col_data = 'purple', size_data = 20, lw_data = 1,
           col_posterior = 'black', size_posterior = 20, lw_posterior = 1,
           col_hist = 'grey',
           col_ref = 'black', lw_ref = 1, ls_ref = ':',
           col_veff = 'black', lw_veff = 1.5, ls_veff = "--",
           legend=True,
           fig=None, ax0 = None, ax1=None):
    """
    Display fitted generative distribution function
    
    This function creates a one-dimensional generative distribution function fitted using :class:`pydftools.dffit.DFFit`.
    
    Parameters
    ----------
    dffit : :class:`~dffit.DFFit` instance
        Provides the data to be fit
    xlab: str, optional
        An x-axis label
    ylab: str, optional
        A y-axis label for the mass-function axis of the figure
    ylab_histogram: str, optional
        A y-axis label for the histogram axis of the figure
    fit_label: str, optional
        A label for the fitted curve, to appear in a legend (useful when overplotting several fits).
    xlim: 2-tuple, optional
        x-axis range
    ylim: 2-tuple, optional
        y-axis range (for mass function axis)
    p_true: sequence,optional
        Parameters of a reference distribution function to be over-plotted on the fitted function. Using `None` will
        omit the reference function.
    xpower10: bool, optional
        If `True`, the model argument x is elevated to the power of 10 in the plots.
    show_input_data: bool, optional
        Whether the input data is shown in bins. Each bin value is simply the sum 1/Veff(x) of the observed x-values
        in this bin.
    show_posterior_data: bool, optional
        Whether the posterior data, constructed from all the individual posterior PDFs of the observed data,
        are shown in bins. Note that posterior data only exists of the fitted data is uncertain (i.e. `x_err` is not
        None in the Data object).
    show_data_histogram: bool, optional
        Whether a histogram of source counts, based on the input data, is displayed in a bottom panel.
    uncertainty_type: int, optional
        How to plot uncertainty regions around the fit. 0: don't plot any. 1: plot Gaussian 1-sigma uncertanties
        propagated from the Hessian matrix of the likelihood. 2: plot 68 percentile region (from 16 to 84 percent).
        3: plot 68 (16 to 84) and 95 (2 to 98) percentile regions.
    show_bias_correction: bool, optional
        Whether the bias corrected MLE is shown instead of the native ML parameters. Note, the ``jackknife()`` method
        must have been called on the fit object for this to work.
    nbins: int, optional
        Number of bins to be plotted in data scatter; must be larger than 0. Choose `None` (default) to determine the
        number of bins automatically.
    bin_xmin, bin_xmax: float, optional
        Left, right edge of first, last bin (for data scatter)
    col_<x>: str, optional
        The color of the line showing object <x>, where x is "fit", "data", "posterior", "hist", "ref" or "veff"
    lw_<x>: float, optional
        The line-width of the line showing object <x>, where x is "fit", "data", "posterior", "ref", or "veff"
    ls_<x>: str, optional
        The linestyle of the line showing object <x>, where x is "fit", "ref" or "veff"
    size_data, size_posterior: float, optional
        The size of the markers in the binned data/posteriors
    legend: bool, optional
        Whether to draw a legend.
    fig, ax0, ax1: optional
        Figure and Axis objects (from matplotlib) defining the canvas on which to draw the plots. These are useful
        for overplotting new fits on the same axis, since they are returned from this function.
    
    Returns
    -------
    fig : matplotlib figure object
        The figure on which the plot is drawn
    ax : list of axes
        The list of axes (upper and lower, if it exists) that have been plotted, which can be passed in to the same
        function for overplotting
    """
    if dffit.data.n_dim != 1:
        raise ValueError("dfplot only handles 1D distribution functions. Use dfplot2 for 2D functions.")

    #Make figure, ax
    # open plot
    if fig is None:
        if show_data_histogram:
            subplot_kw = {}
            if xpower10:
                subplot_kw.update({"xscale":'log'})
            if xlim is not None:
                subplot_kw.update({"xlim":xlim})
            fig, ax = plt.subplots(2, 1, sharex=True,
                                   subplot_kw=subplot_kw,
                                   gridspec_kw={"height_ratios": (3, 1), "hspace": 0})
            ax0 = ax[0]
            ax1 = ax[1]
        else:
            fig, ax0 = plt.subplots(1, 1, subplot_kw={'xlim': xlim, "ylim": ylim, "xscale":"log" if xpower10 else None},
                                    gridspec_kw={})

    # Plot DF
    fig, ax = plot_dffit(
        dffit=dffit,
        ylab=ylab, xlab=None if show_data_histogram else xlab,
        fit_label=fit_label,
        xpower10=xpower10,
        uncertainty_type=uncertainty_type, show_bias_correction=show_bias_correction,
        p_true=p_true,
        col_fit=col_fit, lw_fit=lw_fit, ls_fit = ls_fit,
        col_ref=col_ref, lw_ref=lw_ref, ls_ref=ls_ref,
        ylim=ylim,
        legend=legend,
        fig=fig, ax=ax0
    )

    ax0.set_yscale('log')

    # Plot Histogram
    if show_data_histogram:
        fig, ax = plot_hist(
            dffit=dffit, xlab = xlab if show_data_histogram else None,
            ylab = ylab_histogram,
            nbins=nbins, xpower10=xpower10, col_hist=col_hist,
            col_veff=col_veff, ls_veff=ls_veff, lw_veff=lw_veff,
            fig=fig, ax=ax1
        )

    if show_input_data or show_posterior_data:
        fig, ax = plot_dfdata(
            dffit=dffit, nbins=nbins, bin_xmin=bin_xmin, bin_xmax=bin_xmax,
            show_input_data=show_input_data, show_posterior_data=show_posterior_data,
            xpower10=xpower10,
            col_data=col_data, size_data=size_data, lw_data=lw_data,
            col_posterior=col_posterior, size_posterior=size_posterior, lw_posterior=lw_posterior,
            fig=fig,ax=ax0
        )

    try:
        ax = [ax0, ax1]
    except NameError:
        ax = ax0

    if fit_label is not None:
        ax0.legend(loc=0)

    return fig, ax


def plot_dffit(dffit, ylab= r"$\phi [{\rm Mpc}^{-3}{\rm dex}^{-1}]$",
               fit_label=None,
               xlab = None,
               show_uncertainties=True, xpower10=False, uncertainty_type=1, show_bias_correction=True,
               p_true=None,
               col_fit='blue', lw_fit=2, ls_fit = '-',
               col_ref='black', lw_ref=1, ls_ref=':',
               ylim=None,xlim=None, legend=True,
               fig=None, ax=None, figsize=None):
    # Make sure it's a 1D plot
    if dffit.data.n_dim > 1:
        raise RuntimeError("This plotting routine only deals with 1D distributions.")

    # If not passed a figure/axis, create one.
    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize, subplot_kw={"xscale":'log' if xpower10 else None})
        ax.set_yscale('log')
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel(xlab)

    # PLOT UNCERTAINTY REGIONS
    poly_x = 10 ** dffit.grid.x[0] if xpower10 else dffit.grid.x[0]
    if show_uncertainties and dffit.fit.status['converged']:


        if uncertainty_type > 1 and not hasattr(dffit.grid, "gdf_quantile"):
            raise ValueError('Quantiles not available. Use resampling in dffit.')

        if uncertainty_type == 3:
            ax.fill_between(poly_x, dffit.grid.gdf_quantile[0], dffit.grid.gdf_quantile[-1],
                            color=col_fit, alpha = 0.15)
            # poly_y
            # .95 = pmax(ylim[1], c(dffit.grid.gdf_quantile
            # .02, rev(dffit.grid.gdf_quantile
            # .98)))
            # list = is_finite(poly_x) & is_finite(poly_y
            # .95)
            # polygon(poly_x[list], poly_y
            # .95[list], col = rgb(r, g, b, 0.15), border = np.nan)

        if uncertainty_type >= 2:
            ax.fill_between(poly_x, dffit.grid.gdf_quantile[1], dffit.grid.gdf_quantile[-2],
                            color=col_fit, alpha = 0.25)

        if uncertainty_type == 1:
            ax.fill_between(poly_x, dffit.gdf_gaussian_min, dffit.gdf_gaussian_max,
                            color=col_fit, alpha=0.25)

            # poly_y
            # .68 = pmax(ylim[1], c(dffit.grid.gdf - dffit.grid.gdf_error_neg,
            #                       rev(dffit.grid.gdf + dffit.grid.gdf_error_pos)))
            # list = is_finite(poly_x) & is_finite(poly_y
            # .68)
            # polygon(poly_x[list], poly_y
            # .68[list], col = rgb(r, g, b, 0.25), border = np.nan)  # plot central fit


    # PLOT ACTUAL FIT
    if show_bias_correction and dffit.fit.status['converged'] and hasattr(dffit.grid, "gdf_mle_bias_corrected"):
        fit = dffit.grid.gdf_mle_bias_corrected
    else:
        fit = dffit.grid.gdf

    ax.plot(poly_x, fit, color=col_fit, lw=lw_fit, ls=ls_fit, label=fit_label)


    # PLOT REFERENCE
    if p_true is not None:
        ax.plot(poly_x, dffit.model.gdf(np.log10(poly_x) if xpower10 else poly_x, p_true),
                color=col_ref, lw=lw_ref, linestyle=ls_ref, label="Input" if legend else None)


    # Pretty Up Plot
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylab is not None:
        ax.set_ylabel(ylab)

    return fig, ax


def plot_hist(dffit, xlab = r"$M [M_\odot]$",
              ylab = "Counts",
              nbins=None, xpower10=False, col_hist="grey",
              col_veff="black", ls_veff='--', lw_veff=1.5,
              fig=None, ax=None, figsize=None, xlim=None):
    # Make sure it's a 1D plot
    if dffit.data.n_dim > 1:
        raise RuntimeError("This plotting routine only deals with 1D distributions.")

    # If not passed a figure/axis, create one.
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize,subplot_kw={"xscale":'log' if xpower10 else None, "xlim":xlim})



    # Plot histogram of input data
    # determine number of bins
    if nbins is None:
        nbins = min(100, round(np.sqrt(dffit.data.n_data)))
    else:
        if nbins <= 0:
            raise ValueError('Choose more than 0 bins.')

    if xpower10:
        bins = np.logspace(dffit.data.x.min(), dffit.data.x.max(), nbins)
    else:
        bins = np.linspace(dffit.data.x.min(), dffit.data.x.max(), nbins)

    hval, bin_edges, patches = ax.hist(10**dffit.data.x if xpower10 else dffit.data.x,
                                       bins=bins, color=col_hist)
    ax.get_yaxis().set_ticks([])


    # Plot selection function
    selfnc = dffit.grid.veff * hval.max() * 1.2 / dffit.grid.veff.max()
    ax.plot(10**dffit.grid.x[0] if xpower10 else dffit.grid.x[0],
            selfnc,
            color = col_veff, ls = ls_veff, lw=lw_veff
            )
    ax.set_ylim((0, selfnc.max()*1.2))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    return fig, ax


def plot_dfdata(dffit, nbins=None, bin_xmin=None, bin_xmax=None, show_input_data=True, show_posterior_data=True,
                xpower10=False,
                col_data='grey', size_data=20, lw_data=1,
                col_posterior='blue', size_posterior=20, lw_posterior=1,
                xlab=None,ylab=None,
                fig=None,ax=None, figsize=None, xlim=None):

    # Make sure it's a 1D plot
    if dffit.data.n_dim > 1:
        raise RuntimeError("This plotting routine only deals with 1D distributions.")

    # If not passed a figure/axis, create one.
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize,subplot_kw={"xscale":'log' if xpower10 else None, "xlim":xlim})
        ax.set_yscale('log')
        ax.set_xlabel(xlab)

    # bin data
    bin = bin_data(dffit, nbins, bin_xmin, bin_xmax)

    # plot binned input data points
    # bin = list()
    for mode in range(2):
        if mode == 0:
            show = show_input_data
            if show:
                bin_count = bin['histogram']
                bin_gdf = bin['gdf_input']
                bin_xmean = bin['xmean_input']

                col = col_data
                size = size_data
                lw = lw_data

        else:
            show = show_posterior_data and not dffit.ignore_uncertainties
            # First make sure effective_counts has been created:
            dffit.posterior

            if show:
                bin_count = bin['count_posterior']
                bin_gdf = bin['gdf_posterior']
                bin_xmean = bin['xmean_posterior']

                col = col_posterior
                size = size_posterior
                lw = lw_posterior

        if show:
            lst = bin_gdf > 0
            bin_count = bin_count[lst]
            bin_gdf = bin_gdf[lst]
            bin_xmean = bin_xmean[lst]

            pm = 0.05
            f_16 = poisson.ppf(0.16, bin_count) / bin_count
            f_84 = poisson.ppf(0.84, bin_count) / bin_count
            upper = f_16 < pm
            f_16 = np.clip(f_16, pm, np.inf)

            def xpow(x):
                return 10 ** x if xpower10 else x

            ax.errorbar(xpow(bin_xmean), bin_gdf,
                         yerr=[bin_gdf * (1-f_16), bin_gdf * (f_84-1)],
                         xerr=[xpow(bin_xmean) - xpow(bin['xedges'][:-1][lst]),
                              xpow(bin['xedges'][1:][lst]) - xpow(bin_xmean)],
                         color=col, lw=lw, uplims=upper, ls='none', ms=size)

    if ylab is not None:
        ax.set_ylabel(ylab)

    return fig, ax


def bin_data(dffit, nbins=None, bin_xmin=None, bin_xmax=None):

    # initialize
    x = dffit.data.x
    bin = dict()
    n_data = len(x)

    # determine number of bins
    if nbins is None:
        nbins = min(100, int(round(np.sqrt(n_data))))
    else:
        if nbins <= 0:
            raise ValueError('Choose more than 0 bins.')
    bin['n'] = int(nbins)

    # make bin intervals
    if bin_xmin is None:
        bin['xmin'] = x.min() - (x.max() - x.min()) / bin['n'] * 0.25
    else:
        bin['xmin'] = bin_xmin

    if bin_xmax is None:
        bin['xmax'] = x.max() + (x.max() - x.min()) / bin['n'] * 0.25
    else:
        bin['xmax'] = bin_xmax

    wx = bin['xmax'] - bin['xmin']
    bin['dx'] = wx / bin['n']
    bin['xedges'] = np.linspace(bin['xmin'], bin['xmax'], bin['n'] + 1)
    bin['xcenter'] = (bin['xedges'][1:] + bin['xedges'][:-1])/2

    # Mask out entries outside bin range
    x = x[np.logical_and(x >= bin['xmin'], x < bin['xmax'])]

    # fill input data into bins
    v = dffit.selection.Veff(x)

    # Generate the index of each sample in the bin space
    x_bins = np.digitize(x, bin['xedges']) - 1

    bin['gdf_input'] = np.bincount(x_bins, weights=1 / bin['dx'] / v, minlength=bin['n'])
    bin['histogram'] = np.bincount(x_bins, minlength=bin['n'])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bin['xmean_input'] = np.bincount(x_bins, weights=x, minlength=bin['n']) / np.clip(bin['histogram'], 0, np.inf)

    # fill posterior data into bins
    if not dffit.ignore_uncertainties:
        # Ensure that effective counts has been initialised
        dffit.posterior

        # bin['gdf_posterior'] = bin.count_posterior = bin.xmean_posterior = array(0, bin.n)
        xg = dffit.grid.x[0]
        mask = np.logical_and(xg >= bin['xmin'], xg < bin['xmax'])
        xg = xg[mask]

        xg_bins = np.digitize(xg, bin['xedges']) - 1
        scd = np.bincount(xg_bins, weights=dffit.grid.scd_posterior[mask], minlength=bin['n'])
        cnts = np.bincount(xg_bins, minlength=bin['n'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            bin['xmean_posterior'] = np.bincount(xg_bins, weights=dffit.grid.scd_posterior[mask]*xg, minlength=bin['n'])/scd
            bin['count_posterior'] = np.bincount(xg_bins, weights=dffit.grid.effective_counts[mask], minlength=bin['n'])/cnts
            bin['gdf_posterior'] = np.bincount(xg_bins, weights=dffit.grid.scd_posterior[mask] / dffit.grid.veff[
                mask], minlength=bin['n']) / cnts

    return bin

    #
    #
    #
    # par(omd=c(0, 1, 0, 1))
    # xmarg = sum(par().mai[c(2, 4)])
    # xplot = par().pin[1]
    # ymarg = sum(par().mai[c(1, 3)])
    # yplot = par().pin[2]
    # par(new=T, omd=c(xleft * xplot / (xplot + xmarg), (xright * xplot + xmarg) / (xplot + xmarg),
    #                  ybottom * yplot / (yplot + ymarg), (ytop * yplot + ymarg) / (yplot + ymarg)))
    #
    # ymax = np.max(bin['histogram']) * 1.2
    # if (length(grep('x', log)) == 1) {lg='x'} else {lg=''}
    # plot(1, 1, type='n', log=lg, xaxs='i', yaxs='i', xaxt='n', yaxt='n',
    #      xlim=xlim, ylim=c(0, ymax), xlab='', ylab='', bty='n')
    # xbin = rep(dffit.bin.xmin + seq(0, dffit.bin.n) * dffit.bin.dx, each=2)
    # if (xpower10) xbin = 10 ** xbin
    # xhist = c(xlim[1], xbin, xlim[2])
    # yhist = c(0, 0, rep(dffit.bin.histogram, each=2), 0, 0)
    # polygon(xhist, yhist, col=col_hist, border=np.nan)
    # if (! is_null(veff)) {
    # if (xpower10) {
    # x = seq(log10(xlim[1]), log10(xlim[2]), length=200)
    # } else {
    # x = seq(xlim[1], xlim[2], length=200)
    # }
    # y = veff(x)
    # if (xpower10) {x = 10 ** x}
    # lines(x, y / max(y) * ymax * 0.85, col=col_veff, lwd=lwd_veff, lty=lty_veff)
    # }
    # par(xpd=True)
    # lines(xlim, rep(ymax, 2))
    # par(xpd=False)
    # magicaxis::
    # magaxis(side=2, ylab=ylab_histogram, lwd=np.nan, labels=False, lwd_ticks=np.nan)
    # magicaxis::magaxis(side=4, labels=False, lwd=np.nan, lwd_ticks=np.nan)
    #
    #
    #
    # par(oma=c(0, 0, 0, 0))
    # par(omi=c(0, 0, 0, 0))
    # par(omd=c(0, 1, 0, 1))
    #
    #
    # # ' @export
    # .hist = function(x, breaks)
    # {
    #     b = c(-1e99, breaks, 1e99)
    # counts = hist(x, breaks=b, plot=F).counts[2:(length(b) - 2)]
    # center = (breaks[1:(length(breaks) - 1)] + breaks[2:length(breaks)]) / 2
    # x = array(rbind(array(breaks), array(breaks)))
    # y = c(0, array(rbind(array(counts), array(counts))), 0)
    # return (list(counts=counts, center=center, x=x, y=y))
    # }
    #
    # # ' @export
    # .dfsun < - function(x, y, cex=1)
    # {
    # par(xpd=True)
    # points(rep(x, 2), rep(y, 2), pch=c(1, 20), cex=c(1.2, 0.45) * cex)
    # par(xpd=False)
    # }
    #


def mfplot(dffit,xlab = r"$M [M_\odot]$",
           ylab = r"$\phi [{\rm Mpc}^{-3}{\rm dex}^{-1}]$",
           xpower10=True,
           show_data_histogram=True, **kwargs):
    """
    A convenience wrapper around :func:`~dfplot` which provides some nice defaults for plotting mass functions.
    """
    return dfplot(dffit, xlab=xlab, ylab=ylab, xpower10=xpower10, show_data_histogram=show_data_histogram, **kwargs)


def plotcov(fits, names=None, p_true=None, figsize="grow"):
    """
    Plot covariance ellipses for each of the fitted parameters.

    Parameters
    ----------
    fits : list
        A list of :class:`~dffit.DFFit` objects for which to show the ellipses.

    names: list
        A list with the same length as `fits`, defining names for each fit to appear in a legend.

    p_true : vector
        A vector defining a reference set of parameters

    figsize : str or tuple
        Either a tuple defining the figure size in inches, or a string defining a sizing scheme (see ChainConsumer
        documentation for details).

    Returns
    -------
    fig : matplotlib figure
    """
    if names is None:
        names = [None for i in range(len(fits))]

    # This is a bit slow and hacky, but easy to write up
    c = ChainConsumer()
    for fit,name in zip(fits,names):
        chain = np.random.multivariate_normal(mean=fit.fit.p_best, cov=fit.fit.p_covariance, size=10000)
        c.add_chain(chain, parameters=fit.model.names, name=name)


    fig = c.plotter.plot(figsize=figsize,truth=list(p_true) if p_true is not None else None,
                         legend=names[0] is not None)
    return fig


# def plot_veff(dffit,
#               xlim=None,
#               ylim=None,
#               xlab='Mass M',
#               ylab='Effective volume',
#               legend=True,
#               ylog = True,
#               xpower10=True):
#     """"""
#     n_dim = dffit.data.n_dim
#     if n_dim != 1:
#
#         if n_dim == 2:
#             raise RuntimeError('Use plot_veff2 for two-dimensional Veff functions.')
#         else:
#             raise ValueError('plot_veff only handles one-dimensional Veff functions.')
#
#     if xpower10:
#         x = 10**dffit.grid.x[0]
#         plt.xscale('log')
#     else:
#         x = dffit.grid.x[0]
#
#     plt.plot(x, dffit.grid.veff)
#
#     if xlim is not None:
#         plt.xlim(xlim)
#     if ylim is not None:
#         plt.ylim(ylim)
#
#     if xlab is not None:
#         plt.xlabel(xlab)
#     if ylab is not None:
#         plt.ylabel(ylab)
#
#     if ylog:
#         plt.yscale('log')

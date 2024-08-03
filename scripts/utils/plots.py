import logging

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pixell import curvedsky, enmap

from .utils import trim_alms

logger = logging.getLogger(__name__)


def pol_str(pol, double=False):
    if double:
        return ["TT", "EE", "TE"][pol]
    return ["T", "E", "B"][pol]


def plot_patches(patches, n_plots=8, title="Patches", save_file=None, show=False):
    """
    Plot a grid of image patches.

    Args:
        patches (ndarray): Array of image patches.
        n_plots (int): Number of patches to plot.
        title (str, optional): Title of the plot. Defaults to "Patches".
        save_file (str, optional): File path to save the plot. If None, the plot will be displayed. Defaults to None.
    """

    n_plots = min(n_plots, patches.shape[0])
    nrows = int(np.ceil(n_plots / 4))
    ncols = min(n_plots, 4)
    idxs = range(min(patches.shape[0], n_plots, nrows * ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for idx, patch in enumerate(patches[idxs]):
        ax = axes[idx]
        ax.imshow(patch)
        ax.axis("off")

    for idx in range(n_plots, nrows * ncols):
        fig.delaxes(axes[idx])

    plt.suptitle(title)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)

    if show:
        plt.show()
    else:
        plt.close()


def plot_cl(
    cls,
    lmax,
    title="Angular power spectrum from cl",
    labels=None,
    xlabel=r"$\ell$",
    ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}$",
    legend=True,
    scale=True,
    grid=True,
    save_file=None,
    plot_func=plt.semilogy,
    plot_camb=False,
    camb_cls=None,
    plot_noise=False,
    camb_noise=None,
    camb_beam=None,
    plot_full_camb=False,
    show=False,
    close=True,
):
    r"""
    Plots the angular power spectrum from given Cl values.

    Parameters:
    cls (array-like): The Cl values to plot. Can be a single array or a list of arrays.
    lmax (int): The maximum multipole moment to plot.
    title (str, optional): The title of the plot. Default is "Angular power spectrum from cl".
    labels (list of str, optional): The labels for each Cl array. Default is None.
    xlabel (str, optional): The label for the x-axis. Default is r"$\ell$".
    ylabel (str, optional): The label for the y-axis. Default is r"$\ell(\ell+1)/2\pi\;C_{\ell}$".
    legend (bool, optional): Whether to display the legend. Default is True.
    scale (bool, optional): Whether to scale the Cl values by $\ell(\ell+1)/2\pi$. Default is True.
    grid (bool, optional): Whether to display the grid. Default is True.
    save_file (str, optional): The file path to save the plot. If None, the plot is shown. Default is None.
    plot_func (function, optional): The plotting function to use (e.g., plt.plot, plt.semilogy). Default is plt.semilogy.
    plot_camb (bool, optional): Whether to plot CAMB Cl values. Default is False.
    camb_cls (array-like, optional): The CAMB Cl values to plot. Default is None.
    plot_noise (bool, optional): Whether to plot noise Cl values. Default is False.
    camb_noise (array-like, optional): The noise Cl values to plot. Default is None.
    plot_full_camb (bool, optional): Whether to plot the full CAMB Cl values. Default is False.
    camb_beam (array-like, optional): The beam for CAMB Cl values. Default is None.

    Returns:
    None
    """
    # check args and ensure shape is correct
    if plot_camb:
        if camb_cls is None:
            raise ValueError("Need to provide camb_cls if plt_camb is True.")
        camb_cls = np.atleast_2d(camb_cls)

        if plot_noise:
            if camb_noise is None:
                raise ValueError("Need to provide noise if plotting noise.")
            camb_noise = np.atleast_2d(camb_noise)

        if plot_full_camb:
            if camb_noise is None or camb_beam is None:
                raise ValueError(
                    "Need to provide camb_noise and camb_beam if plotting full camb."
                )

            camb_noise = np.atleast_2d(camb_noise)
            camb_beam = np.atleast_2d(camb_beam)
            camb_cl_full = camb_cls * camb_beam**2 + camb_noise

    ells = np.arange(2, lmax + 1)
    scale = (ells * (ells + 1) / 2 / np.pi) if scale else 1
    cls = np.atleast_2d(cls)
    npols = cls.shape[0]

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]

        if len(labels) != npols:
            raise ValueError("Number of labels must match number of Cl arrays.")
    else:
        labels = ["data"] if npols == 1 else [f"data {i}" for i in range(npols)]

    for pol in range(npols):
        plot_func(ells, scale * cls[pol, ells], label=labels[pol], linestyle=":")

        if plot_camb:
            pstr = "" if npols == 1 else f", {pol_str(pol)}"
            plot_func(
                ells,
                scale * camb_cls[pol, ells],
                label="camb" + pstr,
            )

            if plot_noise:
                plot_func(
                    ells,
                    scale * camb_noise[pol, ells],
                    label=r"noise" + pstr,
                    linestyle="--",
                )

            if plot_full_camb:
                plot_func(
                    ells,
                    scale * camb_cl_full[pol, ells],
                    label=r"camb * beam$^2$ + noise" + pstr,
                    linestyle="--",
                )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend:
        plt.legend()
    if grid:
        plt.grid()

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)

    if show:
        plt.show()
    if close:
        plt.close()


def plot_cl_alm(alm, lmax=None, title="Angular power spectrum from alm", **kwargs):
    """
    Plot the angular power spectrum from alm.

    Parameters:
    alm (array-like): The alm coefficients.
    lmax (int, optional): The maximum multipole moment. If not provided, it will be determined from the length of alm.
    title (str, optional): The title of the plot.
    **kwargs: Additional keyword arguments to be passed to the plot_cl function.

    Returns:
    None
    """
    lmax = lmax if lmax else hp.Alm.getlmax(np.shape(alm)[-1])
    if len(np.shape(alm)) > 1:
        cls = []
        for single_alm in alm:
            cls.append(curvedsky.alm2cl(single_alm))
    else:
        cls = [curvedsky.alm2cl(alm)]
    plot_cl(cls, lmax=lmax, title=title, **kwargs)


def plot_cl_map(map, wcs, lmax, title="Angular power spectrum from map", **kwargs):
    """
    Plot the angular power spectrum from a map.

    Parameters:
    map (ndarray): The input map.
    wcs (WCS): The world coordinate system of the map.
    lmax (int): The maximum multipole moment.
    title (str, optional): The title of the plot. Default is "Angular power spectrum from map".
    **kwargs: Additional keyword arguments to be passed to the plot_cl function.

    Returns:
    None
    """
    if len(np.shape(map)) > 1:
        cls = []
        for single_map in map:
            tmap = enmap.ndmap(single_map, wcs)
            alm = curvedsky.map2alm(tmap, lmax=lmax, copy=True)
            cls.append(curvedsky.alm2cl(alm))
    else:
        tmap = enmap.ndmap(map, wcs)
        alm = curvedsky.map2alm(tmap, lmax=lmax, copy=True)
        cls = [curvedsky.alm2cl(alm)]
    plot_cl(cls, lmax=lmax, title=title, **kwargs)


def plot_predictions(
    truth,
    preds,
    title="Predictions",
    fisher=None,
    save_file=None,
    show=False,
    close=True,
):
    """
    Plots the true labels against the predicted labels. If provided will plot the expected deviations from the provided fisher
    and scaled_variance.

    Args:
        truth (array-like): The true labels.
        preds (array-like): The predicted labels.
        title (str, optional): The title of the plot. Defaults to "Predictions".
        fisher (float, optional): The Fisher value. Defaults to None.
        save_file (str, optional): The file path to save the plot. Defaults to None.
    """
    df = pd.DataFrame(
        {
            "True Fnl": np.array(truth).flatten(),
            "Predicted Fnl": np.array(preds).flatten(),
        }
    )

    # Create a scatter plot with seaborn
    plt.figure(figsize=(16, 12))
    sns.scatterplot(data=df, x="True Fnl", y="Predicted Fnl")

    # Truth line
    line = [min(truth), max(truth)]
    plt.plot(line, line, color="red", linestyle="--", label="truth")

    if fisher is not None:
        std_dev = np.sqrt(1 / fisher)

        plt.plot(line, line + std_dev, color="blue", linestyle="--", label="Fisher")
        plt.plot(line, line - std_dev, color="blue", linestyle="--")

    plt.title(title)
    # plt.legend()
    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_histogram(truth, preds, save_file=None, show=False, close=True):
    """Plot and save a histogram of predictions with mean and std dev as title"""
    # Calculate mean and standard deviation
    truth = truth.flatten()
    preds = preds.flatten()

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, figsize=(16, 12))

    # Plot the predictions on the first subplot
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    sns.histplot(preds, ax=axs[0], legend=False)
    axs[0].set_title(
        f"Predictions - Mean: {mean_pred:.2f}, Standard Deviation: {std_pred:.2f}"
    )

    # Plot the differences on the second subplot
    diff = preds - truth
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    sns.histplot(diff, ax=axs[1], legend=False)
    axs[1].set_title(
        f"Differences - Mean: {mean_diff:.2f}, Standard Deviation: {std_diff:.2f}"
    )

    plt.tight_layout()
    # Save the plot
    if save_file is not None:
        plt.savefig(save_file)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_mollview(maps, title, save_file=None, show=False, close=True):
    """
    Plot a Mollweide projection of the given maps.

    Args:
        maps (ndarray): Array of maps to plot.
        title (str): Title of the plot.
        save_file (str, optional): File path to save the plot. If None, the plot will be displayed. Defaults to None.
    """
    maps = np.atleast_2d(maps)
    pols = maps.shape[0]

    _, axes = plt.subplots(
        1, pols, figsize=(16, 12), subplot_kw={"projection": "mollweide"}
    )

    for pol in range(pols):
        # Create the mollview plot in the corresponding subplot
        plt.axes(axes)
        hp.mollview(
            maps[pol],
            title=f"Polarization {pol}",
            hold=True,
        )

    plt.title(title)

    if save_file is not None:
        plt.savefig(save_file)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_heidel_comp(
    alm_l,
    alm_nl,
    hei_idx=1,
    TCMB=2.7255,
    title="Heidelberg comparison",
    save_file=None,
    show=False,
    close=True,
    **kwargs,
):
    idx = str(hei_idx).zfill(4)
    alm_heidelberg_l = np.array(
        hp.read_alm(f"data/heidelberg/alm_l_{idx}_v3.fits", hdu=(1, 2, 3))
    )
    alm_heidelberg_nl = np.array(
        hp.read_alm(f"data/heidelberg/alm_nl_{idx}_v3.fits", hdu=(1, 2, 3))
    )

    t_scale = TCMB * 1e6
    alm_heidelberg_l *= t_scale
    alm_heidelberg_nl *= t_scale

    # we need to reshape either the alms or the heidelberg to match the same lmax to plot
    if alm_l.shape[-1] > alm_heidelberg_l.shape[-1]:
        lmax = hp.Alm.getlmax(alm_heidelberg_l.shape[-1])
        alm_l = trim_alms(alm_l, lmax)
        alm_nl = trim_alms(alm_nl, lmax)
    else:
        lmax = hp.Alm.getlmax(alm_l.shape[-1])
        alm_heidelberg_l = trim_alms(alm_heidelberg_l, lmax)
        alm_heidelberg_nl = trim_alms(alm_heidelberg_nl, lmax)

    npols = alm_l.shape[0]
    _, axes = plt.subplots(3, npols, figsize=(16, 12))
    for pol in range(npols):
        ylabel = r"$\ell(\ell+1)/2\pi\;C_{\ell}" + f"^{pol_str(pol)}$"
        plt.sca(axes[0, pol])
        plot_cl_alm(
            alm_heidelberg_l[pol],
            lmax,
            labels="heidelberg.",
            title=f"linear, pol: {pol_str(pol)}",
            ylabel=ylabel,
            close=False,
            **kwargs,
        )
        plot_cl_alm(
            alm_l[pol],
            lmax,
            labels="sim",
            title=f"linear, pol: {pol_str(pol)}",
            ylabel=ylabel,
            close=False,
            **kwargs,
        )

        plt.sca(axes[1, pol])
        plot_cl_alm(
            [alm_heidelberg_nl[pol]],
            lmax,
            labels=["heidelberg"],
            title=f"non-linear alms, pol: {pol_str(pol)}",
            ylabel=ylabel,
            close=False,
            **kwargs,
        )
        plot_cl_alm(
            [alm_nl[pol]],
            lmax,
            labels=["sim"],
            title=f"non-linear alms, pol: {pol_str(pol)}",
            ylabel=ylabel,
            close=False,
            **kwargs,
        )

        plt.sca(axes[2, pol])
        plot_cl_alm(
            [alm_heidelberg_l[pol] + alm_heidelberg_nl[pol], alm_l[pol] + alm_nl[pol]],
            lmax,
            labels=["heidelberg", "sim"],
            title=f"full alms, fnl: 1, pol: {pol_str(pol)}",
            ylabel=ylabel,
            close=False,
            **kwargs,
        )

    plt.suptitle(title)
    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file)
    if show:
        plt.show()
    if close:
        plt.close()

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
    """
    Converts the polarization number, {0, 1, 2} to a string {T, E, B} or {TT, EE, TE} if double is True
    """
    if double:
        return ["TT", "EE", "TE"][pol]
    return ["T", "E", "B"][pol]


def finalize_plot(
    title=None,
    tight_layout=True,
    legend=True,
    grid=True,
    save_file=None,
    show=False,
    close=True,
):
    """
    Finalize the plot by adding a legend, grid, tight layout, saving the file, showing the plot and closing it.
    """
    if title:
        plt.title(title)
    if legend:
        plt.legend()
    if grid:
        plt.grid()
    if tight_layout:
        plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_patches(patches, n_plots=8, title="Patches", **kwargs):
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

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)

    kwargs["legend"] = False
    finalize_plot(**kwargs)


def plot_cl(
    cls,
    lmax,
    title="Angular power spectrum from cl",
    labels=None,
    xlabel=r"$\ell$",
    ylabel=r"$\ell(\ell+1)/2\pi\;C_{\ell}$",
    scale=True,
    plot_func=plt.semilogy,
    plot_camb=False,
    camb_cls=None,
    plot_noise=False,
    camb_noise=None,
    camb_beam=None,
    plot_full_camb=False,
    **kwargs,
):
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
    finalize_plot(title, **kwargs)


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


def plot_predictions(truth, preds, title="Predictions", fisher=None, **kwargs):
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
    sns.scatterplot(data=df, x="True Fnl", y="Predicted Fnl", label="Estimates")

    # Truth line
    line = [min(truth), max(truth)]
    plt.plot(line, line, color="red", linestyle="--", label="Truth")

    if fisher is not None:
        std_dev = np.sqrt(1 / fisher)

        plt.plot(line, line + std_dev, color="blue", linestyle="--", label="Fisher")
        plt.plot(line, line - std_dev, color="blue", linestyle="--")

        # lets also print the number of points within 1 sigma
        diff = np.array(preds).flatten() - np.array(truth).flatten()
        std_dev = np.sqrt(1 / fisher)
        within = np.sum(np.abs(diff) < std_dev) / len(diff) * 100
        bbox = dict(boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5)
        plt.text(
            0.95,
            0.05,
            f"{within:.2f}% within 1 sigma",
            bbox=bbox,
            ha="right",
            va="bottom",
            transform=plt.gca().transAxes,
        )

    finalize_plot(title, **kwargs)


def plot_histogram(truth, preds, **kwargs):
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
    finalize_plot(legend=False, **kwargs)


def plot_mollview(maps, title, **kwargs):
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
        pols, 1, figsize=(16, 12), subplot_kw={"projection": "mollweide"}
    )
    axes = np.atleast_1d(axes)

    for pol in range(pols):
        # Create the mollview plot in the corresponding subplot
        plt.sca(axes[pol])
        hp.mollview(
            maps[pol],
            title=f"Polarization {pol}",
            hold=True,
        )

    plt.suptitle(title)
    finalize_plot(tight_layout=False, legend=False, **kwargs)


def plot_elsner_comp(
    alm_l,
    alm_nl,
    elsner_idx=1,
    TCMB=2.7255,
    title="Elsner comparison",
    **kwargs,
):
    idx = str(elsner_idx).zfill(4)
    alm_elsner_l = np.array(
        hp.read_alm(f"data/elsner/alm_l_{idx}_v3.fits", hdu=(1, 2, 3))
    )
    alm_elsner_nl = np.array(
        hp.read_alm(f"data/elsner/alm_nl_{idx}_v3.fits", hdu=(1, 2, 3))
    )

    t_scale = TCMB * 1e6
    alm_elsner_l *= t_scale
    alm_elsner_nl *= t_scale

    # we need to reshape either the alms or the elsner to match the same lmax to plot
    if alm_l.shape[-1] > alm_elsner_l.shape[-1]:
        lmax = hp.Alm.getlmax(alm_elsner_l.shape[-1])
        alm_l = trim_alms(alm_l, lmax)
        alm_nl = trim_alms(alm_nl, lmax)
    else:
        lmax = hp.Alm.getlmax(alm_l.shape[-1])
        alm_elsner_l = trim_alms(alm_elsner_l, lmax)
        alm_elsner_nl = trim_alms(alm_elsner_nl, lmax)

    # need to set up some intter args, but dont want to remove all options
    inner_kwargs = kwargs.copy()
    inner_kwargs["close"] = False
    inner_kwargs["show"] = False

    npols = alm_l.shape[0]
    _, axes = plt.subplots(3, npols, figsize=(16, 12))
    if len(axes.shape) == 1:
        # fix for single pol
        axes = axes[:, np.newaxis]
    for pol in range(npols):
        ylabel = r"$\ell(\ell+1)/2\pi\;C_{\ell}" + f"^{pol_str(pol)}$"
        plt.sca(axes[0, pol])
        plot_cl_alm(
            alm_elsner_l[pol],
            lmax,
            labels="elsner",
            title=f"linear, pol: {pol_str(pol)}",
            ylabel=ylabel,
            **inner_kwargs,
        )
        plot_cl_alm(
            alm_l[pol],
            lmax,
            labels="sim",
            title=f"linear, pol: {pol_str(pol)}",
            ylabel=ylabel,
            **inner_kwargs,
        )

        plt.sca(axes[1, pol])
        plot_cl_alm(
            [alm_elsner_nl[pol]],
            lmax,
            labels=["elsner"],
            title=f"non-linear alms, pol: {pol_str(pol)}",
            ylabel=ylabel,
            **inner_kwargs,
        )
        plot_cl_alm(
            [alm_nl[pol]],
            lmax,
            labels=["sim"],
            title=f"non-linear alms, pol: {pol_str(pol)}",
            ylabel=ylabel,
            **inner_kwargs,
        )

        plt.sca(axes[2, pol])
        plot_cl_alm(
            [alm_elsner_l[pol] + alm_elsner_nl[pol], alm_l[pol] + alm_nl[pol]],
            lmax,
            labels=["elsner", "sim"],
            title=f"full alms, fnl: 1, pol: {pol_str(pol)}",
            ylabel=ylabel,
            **inner_kwargs,
        )

    plt.suptitle(title)
    kwargs.pop("plot_func", None)
    finalize_plot(**kwargs)

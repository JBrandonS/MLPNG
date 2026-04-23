import logging

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pixell import curvedsky, enmap
from tqdm.auto import trange

from .utils import trim_alms

logger = logging.getLogger(__name__)

plt.rc("font", family="serif")
plt.rc("mathtext", **{"default": "regular"})
plt.style.use("seaborn-v0_8-paper")
plt.rc("xtick", **{"direction": "in"})
plt.rc("ytick", **{"direction": "in"})
plt.rc("figure", **{"figsize": (12, 9)})


def finalize_plot(
    finalize: bool = True,
    title: str | None = None,
    tight_layout: bool = True,
    legend: bool = True,
    grid: bool = False,
    save: bool = True,
    save_file: str | None = None,
    show: bool = False,
    close: bool = True,
):
    """
    Finalize the plot by adding a legend, grid, tight layout, saving the file, showing the plot and closing it.

    This is used to keep from having to copy the code, also allows for standarization across calls via kwargs.
    """
    if finalize:
        if title:
            plt.title(title)
        if legend:
            plt.legend()
        if grid:
            plt.grid()
        if tight_layout:
            plt.tight_layout()
        if save and save_file is not None:
            logger.debug("Saving plot to '%s'", save_file)
            plt.savefig(save_file)
        if show:
            plt.show()
        if close:
            plt.close()


def plot_map(maps, zoom=False, **kwargs):
    if zoom:
        hp.mollzoom(maps)
    else:
        hp.mollview(maps)
    finalize_plot(**kwargs)


def plot_map_alm(
    core,
    alm: np.ndarray | list[np.ndarray],
    lmax: int | None = None,
    title: str = "Map from alm",
    **kwargs,
):
    """
    Plot a map from alm coefficients.

    Parameters:
        alm (array-like): The alm coefficients.
        lmax (int, optional): The maximum multipole moment. If not provided, it will be determined from the length of alm.
        title (str, optional): The title of the plot.
        **kwargs: Additional keyword arguments to be passed to the plot function.

    Returns:
        None
    """
    lmax = lmax if lmax else hp.Alm.getlmax(np.shape(alm)[-1])
    if len(np.shape(alm)) > 1:
        maps = []
        for single_alm in alm:
            maps.append(hp.alm2map(single_alm, core.nside, lmax=lmax))
    else:
        maps = hp.alm2map(alm, core.nside)

    plot_map(maps, title=title, **kwargs)


def plot_cl(
    core,
    c_ells,
    lmin=None,
    lmax=None,
    title="Angular power spectrum from cl",
    labels=None,
    xlabel=r"$\ell$",
    ylabel=None,
    scale=True,
    plot_func=plt.semilogy,
    plot_camb=False,
    plot_noise=False,
    plot_full_camb=False,
    **kwargs,
):
    if lmax is None:
        lmax = core.lmax
    if lmin is None:
        lmin = core.lmin
    if ylabel is None:
        ylabel = r"$C_{\ell}$"
    if scale:
        ylabel = f"$\\ell(\\ell+1)/2\\pi\\;${ylabel}"

    ells = np.arange(lmin, lmax + 1)
    scale = (ells * (ells + 1) / 2 / np.pi) if scale else 1
    c_ells = np.atleast_2d(c_ells)[:, lmin : lmax + 1]

    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]

        if len(labels) != core.npols:
            raise ValueError(
                f"Number of labels must match number of Cl arrays. Got {len(labels)} labels and {core.npols} Cl arrays of shape {np.shape(c_ells)}."
            )
    else:
        labels = [f"{core.pols[i]}" for i in range(core.npols)]

    for pol in range(core.npols):
        plot_func(ells, scale * c_ells[pol, :], label=labels[pol], linestyle=":")

        # we need to get the correct pol for the camb and noise
        cpol = core.pol_idxs()[pol]
        if plot_camb:
            c_ell = core.cosmo.c_ell["unlensed_scalar"]["c_ell"][: core.nell].T
            plot_func(
                ells,
                scale * c_ell[cpol, lmin : lmax + 1],
                label=f"camb {core.pols[pol]}",
            )

        if plot_noise:
            plot_func(
                ells,
                core.n_ell[cpol, lmin : lmax + 1],
                label=f"noise {core.pols[pol]}",
                linestyle="--",
            )

        if plot_full_camb:
            c_ell = core.cosmo.c_ell["unlensed_scalar"]["c_ell"][: core.nell].T
            s_ell = core.b_ell**2 * c_ell + core.n_ell
            plot_func(
                ells,
                scale * s_ell[cpol, lmin : lmax + 1],
                label=f"camb + noise, {core.pols[pol]}",
                linestyle="--",
            )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    finalize_plot(title=title, **kwargs)


def plot_cl_alm(
    core,
    alm: np.ndarray | list[np.ndarray],
    lmax: int | None = None,
    title: str = "Angular power spectrum from alm",
    **kwargs,
):
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
            cls.append(hp.alm2cl(single_alm))
    else:
        cls = np.array([hp.alm2cl(alm)])
    plot_cl(core, cls, lmax=lmax, title=title, **kwargs)


def plot_cl_map(
    core,
    maps: np.ndarray,
    wcs,
    lmax: int,
    title: str = "Angular power spectrum from map",
    **kwargs,
):
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
    if len(np.shape(maps)) > 1:
        cls = []
        for single_map in maps:
            tmap = enmap.ndmap(single_map, wcs)
            alm = curvedsky.map2alm(tmap, lmax=lmax, copy=True)
            cls.append(curvedsky.alm2cl(alm))
    else:
        tmap = enmap.ndmap(maps, wcs)
        alm = curvedsky.map2alm(tmap, lmax=lmax, copy=True)
        cls = [curvedsky.alm2cl(alm)]

    plot_cl(core, cls, lmax=lmax, title=title, **kwargs)


def plot_cl_vs(
    core,
    c_ell_a: np.ndarray | list[np.ndarray],
    c_ell_b: np.ndarray | list[np.ndarray],
    title="Angular power spectrum comparison",
    labels=["unlensed", "lensed"],
    xlabel=r"$\ell$",
    ylabel=None,
    scale=True,
    plot_func=plt.semilogy,
    plot_diff_func=plt.plot,
    file_base=None,
    **kwargs,
):
    if ylabel is None:
        ylabel = r"$\ell(\ell+1)/2\pi\;C_{\ell}$" if scale else r"$C_{\ell}$"

    ells = np.arange(2, core.nell)
    scale = (ells * (ells + 1) / 2 / np.pi) if scale else 1

    plot_func(ells, scale * c_ell_a[2:], label=labels[0], linestyle=":")
    plot_func(ells, scale * c_ell_b[2:], label=labels[1], linestyle="--")
    save_file = file_base + "_vs.png" if file_base else None
    finalize_plot(title=title, save_file=save_file, **kwargs)

    delta = (c_ell_b - c_ell_a) / c_ell_a
    plot_diff_func(ells, delta[2:], label="Difference", linestyle="--")
    save_file = file_base + "_diff.png" if file_base else None
    finalize_plot(title=f"{title} - Difference", save_file=save_file, **kwargs)


def plot_predictions(
    truth: np.ndarray,
    preds: np.ndarray,
    title: str = "Predictions",
    x_limit=None,
    data_label=None,
    truth_color="red",
    sigma: float | None = None,
    sigma_color="blue",
    show_sigma_legend=True,
    show_n_sigma=4,
    **kwargs,
):
    if x_limit is not None:
        mask_idx = np.abs(truth) <= x_limit
        truth_ = np.where(mask_idx, truth, np.nan)
        preds_ = np.where(mask_idx, preds, np.nan)
    else:
        truth_ = truth
        preds_ = preds

    df = pd.DataFrame(
        {
            "True": np.array(truth_).flatten(),
            "Predicted": np.array(preds_).flatten(),
        }
    )

    sns.scatterplot(data=df, x=r"True", y=r"Predicted", label=data_label, alpha=0.5)

    # Truth line
    line = [np.nanmin(truth_), np.nanmax(truth_)]
    plt.plot(line, line, color=truth_color, linestyle="--")
    if sigma is not None:
        plt.plot(line, line + sigma, color=sigma_color, linestyle="--")
        plt.plot(line, line - sigma, color=sigma_color, linestyle="--")

        if show_sigma_legend:
            add_sigma_legend(truth_, preds_, sigma, data_label, show_n_sigma)

    finalize_plot(title=title, **kwargs)


def plot_predictions_combined(
    truth: np.ndarray,
    preds: np.ndarray,
    shape_strs: list[str],
    likelihoods: np.ndarray,
    title: str = "Predictions",
    x_limit=None,
    show_sigma_legend=True,
    show_n_sigma=4,
    **kwargs,
):
    sns.set_palette("bright")
    palette = sns.color_palette("dark")

    for i, (t, p, sigma, label) in enumerate(
        zip(truth.T, preds.T, likelihoods, shape_strs)
    ):
        plot_predictions(
            t,
            p,
            title=f"{title}",
            sigma=sigma,
            data_label=label,
            x_limit=x_limit,
            truth_color="red",
            sigma_color=palette[i],
            show_sigma_legend=False,
            finalize=False,
        )

    if show_sigma_legend:
        if x_limit is not None:
            mask_idx = np.abs(truth) <= x_limit
            truth_ = np.where(mask_idx, truth, np.nan)
            preds_ = np.where(mask_idx, preds, np.nan)
        else:
            truth_ = truth
            preds_ = preds
        add_sigma_legend(truth_.T, preds_.T, likelihoods, shape_strs, show_n_sigma)

    finalize_plot(title=title, **kwargs)


def add_sigma_legend(truth, preds, sigma, labels, n_sigma=4):
    truth_ = np.atleast_2d(truth)
    preds_ = np.atleast_2d(preds)
    sigma_ = np.atleast_1d(sigma)
    labels_ = np.atleast_1d(labels)
    row_labels = [f"{i} $\\sigma$" for i in range(1, n_sigma + 1)]

    data = []
    for i in range(1, n_sigma + 1):
        row = []
        for j in range(len(labels_)):
            diff = np.array(preds_[j]).flatten() - np.array(truth_[j]).flatten()
            diff = np.abs(diff[~np.isnan(diff)])
            within = np.sum(diff < i * sigma_[j]) / len(diff) * 100
            row.append(f"{within:.2f}")
        data.append(row)

    plt.table(
        data,
        colWidths=[0.07] * n_sigma,
        rowLabels=row_labels,
        colLabels=labels_,
        loc="best",
        zorder=10,
    )


def plot_histogram(truth: np.ndarray, preds: np.ndarray, title="Predictions", **kwargs):
    """Plot and save a histogram of predictions with mean and std dev as title"""
    # Calculate mean and standard deviation
    truth = truth.flatten()
    preds = preds.flatten()

    # Create a figure with two subplots
    _, axs = plt.subplots(2, figsize=(16, 12))

    # Plot the predictions on the first subplot
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    sns.histplot(preds, ax=axs[0], legend=False)
    axs[0].set_title(
        f"{title} - Mean: {mean_pred:.2f}, Standard Deviation: {std_pred:.2f}"
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


def plot_mollview(maps, title: str, **kwargs):
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
    core,
    alm_l,
    alm_nl,
    index: int | None = None,
    average: int = 0,
    title: str = "Elsner comparison",
    elsner_dir: str = "data/elsner",
    scale_elsner: bool = True,
    plot_func=plt.plot,
    **kwargs,
):
    """Plot comparison between simulated and Elsner alms.

    Parameters:
        alm_l : array-like
            Linear alms from simulations.
        alm_nl : array-like
            Non-linear alms from simulations.
        TCMB : float, optional
            Temperature of the CMB in Kelvin. Default is 2.7255.
        index : int or None, optional
            Index of the Elsner simulation to use. If None, a random index is chosen. Default is None.
        average : int, optional
            Number of Elsner simulations to average. If greater than 0, the specified number of simulations are averaged. Default is 0.
        title : str, optional
            Title of the plot. Default is "Elsner comparison".
        elsner_dir : str, optional
            Directory containing the Elsner simulation files. Default is "data/elsner".
        **kwargs : dict, optional
            Additional keyword arguments passed to the plotting functions.
    Returns:
        None
    """
    # need to set up some inner args, but dont want to remove all options
    inner_kwargs = kwargs.copy()
    inner_kwargs["close"] = False
    inner_kwargs["show"] = False
    inner_kwargs["save_file"] = None

    # get our elsner sims
    if average > 0:
        lsum = 0
        nlsum = 0
        logger.debug("Averaging %s elsner sims, this will take a while", average)
        for i in trange(1, average + 1, desc="Averaging elsner sims"):
            i = str(i).zfill(4)
            lsum += hp.read_alm(f"{elsner_dir}/alm_l_{i}_v3.fits", hdu=(1, 2, 3))
            nlsum += hp.read_alm(f"{elsner_dir}/alm_nl_{i}_v3.fits", hdu=(1, 2, 3))
        elsner_l = np.array(lsum / average)
        elsner_nl = np.array(nlsum / average)
    else:
        if index is None:
            index = np.random.randint(1, 1001)

        idx = str(index).zfill(4)
        elsner_l = np.array(
            hp.read_alm(f"{elsner_dir}/alm_l_{idx}_v3.fits", hdu=(1, 2, 3))
        )
        elsner_nl = np.array(
            hp.read_alm(f"{elsner_dir}/alm_nl_{idx}_v3.fits", hdu=(1, 2, 3))
        )

    if scale_elsner:
        # we should scale the elsner alms to match the simulations
        t_scale = core.cosmo.camb_params.TCMB * 1e6
        elsner_l *= t_scale
        elsner_nl *= t_scale

    elsner_l = elsner_l[core.pol_idxs()]
    elsner_nl = elsner_nl[core.pol_idxs()]

    # we need to reshape either the alms or the elsner to match the same lmax to plot
    if alm_l.shape[-1] > elsner_l.shape[-1]:
        lmax = hp.Alm.getlmax(elsner_l.shape[-1])
        alm_l = trim_alms(alm_l, lmax)
        alm_nl = trim_alms(alm_nl, lmax)
    else:
        lmax = hp.Alm.getlmax(alm_l.shape[-1])
        elsner_l = trim_alms(elsner_l, lmax)
        elsner_nl = trim_alms(elsner_nl, lmax)

    npols = core.npols
    _, axes = plt.subplots(3, npols, figsize=(16, 12))
    if len(axes.shape) == 1:
        # fix for single pol
        axes = axes[:, np.newaxis]

    for data in [(alm_l, alm_nl, "alm"), (elsner_l, elsner_nl, "elsner")]:
        for pol in range(core.npols):
            ratio = np.ma.mean(np.ma.abs(data[0][pol]) / np.ma.abs(data[1][pol]))
            logger.debug("Mean ratio of %s %s: %s", data[2], core.pols[pol], ratio)

    for pol in range(core.npols):
        plt.sca(axes[0, pol])
        plot_elsner_vs(
            core,
            alm_l[pol],
            elsner_l[pol],
            core.pols[pol],
            lmax,
            plot_func=plot_func,
        )
        finalize_plot(**inner_kwargs)

        plt.sca(axes[1, pol])
        plot_elsner_vs(
            core,
            alm_nl[pol],
            elsner_nl[pol],
            core.pols[pol],
            lmax,
            plot_func=plot_func,
        )
        finalize_plot(**inner_kwargs)

        plt.sca(axes[2, pol])
        plot_elsner_vs(
            core,
            alm_l[pol] + alm_nl[pol],
            elsner_l[pol] + elsner_nl[pol],
            core.pols[pol],
            lmax,
            plot_func=plot_func,
        )
        finalize_plot(**inner_kwargs)

    plt.suptitle(title)
    kwargs.pop("plot_func", None)
    finalize_plot(**kwargs)


def plot_elsner_vs(
    core,
    alm: np.ndarray | list[np.ndarray],
    elsner: np.ndarray | list[np.ndarray],
    pol: str,
    lmax: int | None = None,
    lmin: int | None = None,
    scale=True,
    plot_func=plt.semilogy,
):
    lmin = lmin if lmin else core.lmin
    lmax = lmax if lmax else hp.Alm.getlmax(np.shape(alm)[-1])
    ells = np.arange(lmin, lmax + 1)
    scale = (ells * (ells + 1) / 2 / np.pi) if scale else 1
    alm_cl = curvedsky.alm2cl(alm)[lmin : lmax + 1]
    elsner_cl = curvedsky.alm2cl(elsner)[lmin : lmax + 1]

    if np.shape(alm) != np.shape(elsner):
        raise ValueError(
            f"Number of alms must match number of elsner alms. Got {np.shape(alm)} alms and {np.shape(elsner)} elsner alms."
        )

    plot_func(ells, scale * alm_cl, label=f"alm {pol}", linestyle=":")
    plot_func(ells, scale * elsner_cl, label=f"elsner {pol}", linestyle=":")
    plt.legend()


def plot_metrics(history, metrics=["loss"], **kwargs):
    num_metrics = len(metrics)
    _, axs = plt.subplots(num_metrics, figsize=(15, 6 * num_metrics))

    if num_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        axs[i].semilogy(history.history[metric])
        axs[i].semilogy(history.history[f"val_{metric}"])
        axs[i].set_title(f"{metric}")
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Epoch")
        axs[i].legend(["Train", "Validation"], loc="upper right")

    finalize_plot(legend=False, **kwargs)


def make_alm_plots(
    core, shape, alm_l=None, alm_ng=None, alms=None, lensed=False, sim=None
):
    """
    Generate and save various alm plots for comparison and testing.

    Parameters:
        core (object): Core object containing necessary methods and attributes for plotting.
        alm_l (array): Array of linear alm values.
        alm_ng (array): Array of non-Gaussian alm values.
        alms (array): Array of full alm values.
    """
    logger.debug("Generating alm plots")
    sim = sim if sim is not None else core.rng.integers(core.nsims)
    l_str = f"{shape}_lensed" if lensed else f"{shape}_unlensed"
    l_title = f"lensed {shape}" if lensed else f"unlensed {shape}"
    l_title += f" (lmax={core.lmax})"

    # plot a few comparison with different functions to get views
    idx = core.rng.integers(1, 1001)
    if alm_l is not None and alm_ng is not None:
        plot_elsner_comp(
            core,
            alm_l[sim],
            alm_ng[sim],
            index=idx,
            save_file=core.get_plot_file(
                f"{sim}_{l_str}", sub_path="elsner_comp/linear/"
            ),
            plot_func=plt.plot,
        )
        plot_elsner_comp(
            core,
            alm_l[sim],
            alm_ng[sim],
            index=idx,
            save_file=core.get_plot_file(f"{sim}_{l_str}", sub_path="elsner_comp/log/"),
            plot_func=plt.loglog,
        )
        plot_elsner_comp(
            core,
            alm_l[sim],
            alm_ng[sim],
            index=idx,
            save_file=core.get_plot_file(
                f"{sim}_{l_str}", sub_path="elsner_comp/semilogy/"
            ),
            plot_func=plt.semilogy,
        )

    if alms is None and alm_l is not None and alm_ng is not None:
        alms = alm_l + alm_ng

    if alms is not None:
        plot_cl_alm(
            core,
            alms[sim],
            save_file=core.get_plot_file(f"{sim}_{l_str}_alm", sub_path="power"),
            ylabel=r"$C^{tot}_{\ell}$",
            plot_camb=True,
            plot_full_camb=True,
        )
        plot_map_alm(
            core,
            alms[sim, 0],
            title=f"{l_title} full alms",
            save_file=core.get_plot_file(f"{sim}_{l_str}_alm", sub_path="maps"),
        )

    if alm_l is not None:
        plot_cl_alm(
            core,
            alm_l[sim],
            save_file=core.get_plot_file(f"{sim}_{l_str}_alm_l", sub_path="power"),
            ylabel=r"$C_{\ell}^{L}$",
            plot_camb=True,
            plot_full_camb=True,
        )
        plot_map_alm(
            core,
            alm_l[sim, 0],
            title=f"{l_title} alm_l",
            save_file=core.get_plot_file(f"{sim}_{l_str}_alm_l", sub_path="maps"),
        )

    if alm_ng is not None:
        plot_cl_alm(
            core,
            alm_ng[sim],
            save_file=core.get_plot_file(f"{sim}_{l_str}_alm_ng", sub_path="power"),
            ylabel=r"$C_{\ell}^{NG}$",
        )
        plot_map_alm(
            core,
            alm_ng[sim, 0],
            title=f"{l_title} alm_ng",
            save_file=core.get_plot_file(f"{sim}_{l_str}_alm_ng", sub_path="maps"),
        )


def make_trainer_plots(
    core, plot_dir, run_name, history, metrics, truth, preds, likelihoods, **kwargs
):
    # and make our plots
    plot_metrics(
        history,
        metrics=["loss"] + [f"rmse_{s}" for s in core.shapes],
        save_file=core.get_plot_file(f"{run_name}-loss", plot_dir),
        **kwargs
    )

    plot_predictions_combined(
        truth,
        preds,
        likelihoods=likelihoods,
        shape_strs=core.shapes,
        title="Predictions",
        save_file=core.get_plot_file(f"{run_name}-predictions", plot_dir),
        **kwargs
    )

    # plot a reduced range which should better approach the CR bound
    plot_predictions_combined(
        truth,
        preds,
        likelihoods=likelihoods,
        shape_strs=core.shapes,
        x_limit=250.0,
        save_file=core.get_plot_file(f"{run_name}-redux", plot_dir),
        **kwargs
    )

    # individual predictions for each shape
    for s in range(truth.shape[1]):
        s_str = core.shapes[s]
        plot_predictions(
            truth[:, s],
            preds[:, s],
            sigma=likelihoods[s],
            data_label=s_str,
            show_sigma_legend=True,
            title=f"{s_str} RMSE: {metrics[1 + s]:.3f}",
            save_file=core.get_plot_file(f"{run_name}-preds-{s_str}", plot_dir),
            legend=False,
            **kwargs
        )

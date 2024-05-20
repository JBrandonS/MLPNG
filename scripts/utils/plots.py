import logging

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pixell import curvedsky, enmap
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


def plot_patches(patches, n_plots, title="Patches", save_file=None):
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
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
        plt.close()
    else:
        plt.show()


def plot_cl(
    cl,
    lmax,
    title="Angular power spectrum from cl",
    save_file=None,
    plot_func=plt.semilogy,
    plot_camb=False,
    c_ells=None,
    camb_noise=False,
    noise=None,
    beam_width=None,
    scale=True,
):
    """
    Plot the angular power spectrum from cl.

    Args:
        cl (array-like): Array of angular power spectrum values.
        lmax (int): Maximum value of ell.
        title (str, optional): Title of the plot. Defaults to "Angular power spectrum from cl".
        save_file (str, optional): File path to save the plot. Defaults to None.
        plot_func (function, optional): Plotting function to use. Defaults to plt.semilogy.
        plot_camb (bool, optional): Whether to plot the camb values. Defaults to False.
        c_ells (array-like, optional): Array of camb values. Required if plot_camb is True.
        camb_noise (bool, optional): Whether to plot camb values with noise. Defaults to False.
        noise (array-like, optional): Array of noise values. Required if camb_noise is True.
        beam_width (float, optional): Beam width value. Required if camb_noise is True.
        scale (bool, optional): Whether to scale the values. Defaults to True.

    Returns:
        None
    """
    pol = 0
    nell = lmax + 1
    ells = np.arange(2, nell)
    scale = ells * (ells + 1) / 2 / np.pi if scale else 1
    plot_func(ells, scale * cl[2:nell], label="data", linestyle=":")

    if plot_camb:
        if c_ells is None:
            raise ValueError("Need to provide c_ells if plt_camb is True.")

        camb_cl = c_ells[2:nell, pol]
        plot_func(ells, scale * camb_cl, label="camb")

        if camb_noise:
            if noise is None or beam_width is None:
                raise ValueError(
                    "Need to provide noise and beam_width if plotting camb with noise."
                )

            beam = np.exp(-(ells * (ells + 1) * beam_width**2) / (16 * np.log(2)))
            camb_cl_noise = camb_cl * beam**2 + noise[2:nell]
            plot_func(
                ells,
                scale * camb_cl_noise,
                label=r"camb * beam$^2$ + noise",
                linestyle="--",
            )

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\ell(\ell+1)/2\pi\;C_{\ell}$")
    plt.title(title)
    plt.legend()
    plt.grid()

    if save_file is not None:
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()


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
    cl = curvedsky.alm2cl(alm)
    lmax = lmax if lmax else hp.Alm.getlmax(len(alm))
    plot_cl(cl, lmax, title, **kwargs)


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
    tmap = enmap.ndmap(map, wcs)
    alm = curvedsky.map2alm(tmap, lmax=lmax)
    cl = curvedsky.alm2cl(alm)
    plot_cl(cl, lmax, title, **kwargs)


def plot_predictions(
    truth, preds, title="Predictions", fisher=None, scaled_variance=None, save_file=None
):
    """
    Plots the true labels against the predicted labels. If provided will plot the expected deviations from the provided fisher
    and scaled_variance.

    Args:
        truth (array-like): The true labels.
        preds (array-like): The predicted labels.
        title (str, optional): The title of the plot. Defaults to "Predictions".
        fisher (float, optional): The Fisher value. Defaults to None.
        scaled_variance (float, optional): The scaled variance value. Defaults to None.
        save_file (str, optional): The file path to save the plot. Defaults to None.
    """
    df = pd.DataFrame(
        {
            "True Labels": np.array(truth).flatten(),
            "Predicted Labels": np.array(preds).flatten(),
        }
    )

    # Create a scatter plot with seaborn
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="True Labels", y="Predicted Labels")

    # Truth line
    line = [min(truth), max(truth)]
    plt.plot(line, line, color="red", linestyle="--", label="truth")

    if fisher is not None:
        std_dev = np.sqrt(1 / fisher)
        logger.debug(f"Plotting with standard deviation: {std_dev}")
        plt.plot(line, line + std_dev, color="blue", linestyle="--", label="Fisher")
        plt.plot(line, line - std_dev, color="blue", linestyle="--")

    if scaled_variance is not None:
        plt.plot(
            line,
            line + scaled_variance,
            color="green",
            linestyle="--",
            label=r"Scaled Variance (1/$\sqrt{f_{sky} f}$)",
        )
        plt.plot(line, line + scaled_variance, color="green", linestyle="--")
        plt.plot(line, line - scaled_variance, color="green", linestyle="--")

    # Line for perfect fit
    r2 = r2_score(df["True Labels"], df["Predicted Labels"])
    plt.text(min(truth), max(truth), f"$R^2$ = {r2:.2f}", verticalalignment="top")
    plt.title(title)

    if save_file is not None:
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()


def plot_histogram(truth, preds, save_file=None):
    """Plot and save a histogram of predictions with mean and std dev as title"""
    # Calculate mean and standard deviation
    truth = truth.flatten()
    preds = preds.flatten()

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, figsize=(12, 12))

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

    # Save the plot
    if save_file is not None:
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()

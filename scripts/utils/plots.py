import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pixell import enmap, curvedsky

from sklearn.metrics import r2_score
import healpy as hp

import logging
logger = logging.getLogger(__name__)


def plot_patches(patches, n_plots, title="Patches", save_file=None):
    nrows = int(np.ceil(n_plots / 4))
    ncols = min(n_plots, 4)
    idxs = range(min(patches.shape[0], n_plots, nrows*ncols))
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
    c_ells=None
):
    ells = np.arange(2, lmax)
    scale = ells * (ells + 1) / 2 / np.pi
    plot_func(ells, scale * cl[2:lmax], label="data")

    if plot_camb:
        if c_ells is None:
            raise ValueError("Need to provide c_ells if plt_camb is True.")

        camb_cl = c_ells["c_ell"][2:lmax][:, 0]
        camb_inner_plt = scale * camb_cl
        plot_func(ells, camb_inner_plt, label="camb")

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
    cl = curvedsky.alm2cl(alm)
    lmax = lmax if lmax else hp.Alm.getlmax(len(alm))
    plot_cl(cl, lmax, title, **kwargs)


def plot_cl_map(map, wcs, lmax, title="Angular power spectrum from map", **kwargs):
    tmap = enmap.ndmap(map, wcs)
    alm = curvedsky.map2alm(tmap, lmax=lmax)
    cl = curvedsky.alm2cl(alm)
    plot_cl(cl, lmax, title, **kwargs)


def plot_predictions(truth, preds, title="Predictions", fisher=None, scaled_variance=None, save_file=None):
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

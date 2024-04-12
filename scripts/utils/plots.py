import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import r2_score
import healpy as hp


def plot_patches(patches, n_plots, title="Patches", save_file=None):
    idxs = np.random.choice(patches.shape[0], n_plots, replace=False)

    nrows = int(np.ceil(n_plots / 4))
    ncols = min(n_plots, 4)
    fig, axes = plt.subplots(nrows, n_plots, figsize=(20, 20))

    for idx, patch in enumerate(patches[idxs]):
        row = idx // n_plots
        col = idx % n_plots
        if nrows == 1:
            if n_plots == 1:
                ax = axes
            else:
                ax = axes[col]
        else:
            ax = axes[row, col]
        ax.imshow(patch)
        ax.axis("off")

    # Remove empty subplots
    for idx in range(0, nrows * ncols):
        fig.delaxes(axes.flatten()[idx])

    plt.title(title)
    plt.tight_layout()
    plt.show()

    if save_file is not None:
        plt.savefig(save_file)
        plt.close()


def plot_cl(
    cl,
    lmax,
    title="Angular power spectrum from cl",
    save_file=None,
    plot_func=plt.semilogy,
    plot_camb=False,
    c_ells=None,
    plot_camb_noise=False,
    noise=None,
    beam=None,
):
    ells = np.arange(2, lmax)
    scale = ells * (ells + 1) / 2 / np.pi

    plot_func(ells, scale * cl[2:lmax], label="data")

    if plot_camb:
        if c_ells is None:
            raise ValueError("Need to provide c_ells if plt_camb is True.")

        camb_cl = c_ells["c_ell"][2:lmax][:, 0]

        if plot_camb_noise:
            if noise is None or beam is None:
                raise ValueError("Need to provide noise and beam")

            def noise_func(noise, l):
                return noise**2 * np.exp((l * (l + 1) * beam**2) / (8 * np.log(2)))

            noise_ell_b = np.array([noise_func(noise, l) for l in range(2, lmax)])

            camb_n_inner_plt = scale * (camb_cl + noise_ell_b)
            plot_func(ells, camb_n_inner_plt, label="camb + noise")

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


def plot_cl_alm(alm, title="Angular power spectrum from alm", **kwargs):
    from pixell import curvedsky

    cl = curvedsky.alm2cl(alm)
    lmax = hp.Alm.getlmax(len(alm))
    plot_cl(cl, lmax, title, **kwargs)


def plot_cl_map(map, wcs, lmax, title="Angular power spectrum from map", **kwargs):
    from pixell import enmap, curvedsky

    tmap = enmap.ndmap(map, wcs)
    alm = curvedsky.map2alm(tmap, lmax=lmax)
    cl = curvedsky.alm2cl(alm)

    plot_cl(cl, lmax, title, **kwargs)


def plot_predictions(truth, preds, fisher=None, scaled_variance=None, save_file=None):
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

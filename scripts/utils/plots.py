import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import r2_score
import healpy as hp

from astropy import units as u


def plot_patches(patches, n_plots, title="Patches", save_file=None):
    patches = np.random.choice(patches, n_plots, replace=False)

    nrows = int(np.ceil(n_plots / 4))
    ncols = min(n_plots, 4) 
    fig, axes = plt.subplots(nrows, n_plots, figsize=(20, 20))

    for idx, patch in enumerate(patches):
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
    noise_scale_tt=None,
    beam_width=None,
):
    ells = np.arange(2, lmax)
    scale = ells * (ells + 1) / 2 / np.pi

    plot_func(ells, scale * cl[2:lmax], label="data")

    if plot_camb:
        if c_ells is None:
            raise ValueError("Need to provide c_ells if plt_camb is True.")

        camb_cl = c_ells["c_ell"][2:lmax][:, 0]

        if plot_camb_noise:
            if noise_scale_tt is None or beam_width is None:
                raise ValueError(
                    "Need to provide noise_scale_tt and beam_width if plt_camb_noise is True."
                )

            nstt = noise_scale_tt.to_value(u.radian)
            bwr = beam_width.to_value(u.radian)
            noise_ell_b = np.array(
                [
                    nstt**2 * np.exp((l * (l + 1) * bwr**2) / (8 * np.log(2)))
                    for l in range(2, lmax + 1)
                ]
            )

            camb_cls_n = camb_cl + noise_ell_b
            camb_n_inner_plt = scale * camb_cls_n
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


def plot_cl_alm(
    alm,
    title="Angular power spectrum from alm",
    save_file=None,
    plot_func=plt.semilogy,
    plot_camb=False,
    c_ells=None,
    plot_camb_noise=False,
    noise_scale_tt=None,
    beam_width=None,
):
    from pixell import curvedsky

    cl = curvedsky.alm2cl(alm)
    lmax = hp.Alm.getlmax(len(alm))
    plot_cl(
        cl,
        lmax,
        title,
        save_file,
        plot_func,
        plot_camb,
        c_ells,
        plot_camb_noise,
        noise_scale_tt,
        beam_width,
    )


def plot_cl_map(
    map,
    wcs,
    title="Angular power spectrum from map",
    save_file=None,
    plot_func=plt.semilogy,
    plot_camb=False,
    c_ells=None,
    plot_camb_noise=False,
    noise_scale_tt=None,
    beam_width=None,
):
    from pixell import enmap, curvedsky

    tmap = enmap.ndmap(map, wcs)
    lmax = tmap.lmax()
    alm = curvedsky.map2alm(tmap, lmax=lmax)
    cl = curvedsky.alm2cl(alm)

    plot_cl(
        cl,
        lmax,
        title,
        save_file,
        plot_func,
        plot_camb,
        c_ells,
        plot_camb_noise,
        noise_scale_tt,
        beam_width,
    )


def plot_ksw_predictions(fnls, preds, fisher=None, save_file=None):
    df = pd.DataFrame(
        {
            "True Labels": np.array(fnls).flatten(),
            "Predicted Labels": np.array(preds).flatten(),
        }
    )

    # Create a scatter plot with seaborn
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="True Labels", y="Predicted Labels")

    # Truth line
    plt.plot(
        [min(fnls), max(fnls)], [min(fnls), max(fnls)], color="red", linestyle="--"
    )

    if fisher is not None:
        std_dev = np.sqrt(1 / fisher)
        plt.plot(
            [min(fnls), max(fnls)],
            [min(fnls) + std_dev, max(fnls) + std_dev],
            color="blue",
            linestyle="--",
            label="Fisher",
        )
        plt.plot(
            [min(fnls), max(fnls)],
            [min(fnls) - std_dev, max(fnls) - std_dev],
            color="blue",
            linestyle="--",
        )

    # Line for perfect fit
    r2 = r2_score(df["True Labels"], df["Predicted Labels"])
    plt.text(min(fnls), max(fnls), f"R^2 = {r2:.2f}", verticalalignment="top")

    plt.title("Predicted vs True Labels")

    if save_file is not None:
        plt.savefig(save_file)
        plt.close()

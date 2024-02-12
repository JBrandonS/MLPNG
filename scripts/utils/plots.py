import numpy as np
import matplotlib.pyplot as plt


def save_plt(plot_dir, name):
    file = f"{plot_dir}/{name}.png"
    plt.savefig(file)


def plot_cl(
    cl,
    settings,
    plt_func=plt.semilogy,
    plt_camb=True,
    plot_noise=True,
    c_ells=None,
    title="Angular power spectrum from cl",
    save_name=None,
    save=True,
):
    from astropy import units as u

    lmax = settings.lmax
    ells = settings.ells[2:lmax]
    scale = ells * (ells + 1) / 2 / np.pi

    plt.figure()
    plt_func(ells, scale * cl[2:lmax], label="data")

    if plt_camb:
        camb_cl = c_ells["c_ell"][2:lmax][:, 0]

        if plot_noise:
            nstt = settings.noise_scale_tt.to_value(u.radian)
            bwr = settings.beam_width.to_value(u.radian)
            noise_ell_b = np.array(
                [
                    nstt**2 * np.exp((l * (l + 1) * bwr**2) / (8 * np.log(2)))
                    for l in range(settings.nell)
                ]
            )

            camb_cls_n = camb_cl + noise_ell_b[2:lmax]
            camb_n_inner_plt = scale * camb_cls_n
            plt_func(ells, camb_n_inner_plt, label="camb + noise")

        camb_inner_plt = scale * camb_cl
        plt_func(ells, camb_inner_plt, label="camb")

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\ell(\ell+1)/2\pi\;C_{\ell}$")
    plt.title(title)
    plt.legend()
    plt.grid()

    if save:
        save_plt("data/plots", save_name)

    plt.show()
    plt.close()


def plot_cl_alm(
    alm,
    settings,
    plt_func=plt.semilogy,
    plt_camb=True,
    plt_noise=True,
    c_ells=None,
    title="Angular power spectrum from alm",
    save_name="alm",
    save=True,
):
    from pixell import curvedsky

    cl = curvedsky.alm2cl(alm)
    plot_cl(cl, settings, plt_func, plt_camb, plt_noise, c_ells, title, save_name, save)


def plot_cl_map(
    map,
    wcs,
    settings,
    plt_func=plt.semilogy,
    plt_camb=True,
    plt_noise=True,
    c_ells=None,
    title="Angular power spectrum from map",
    save_name="cl",
    save=True,
):
    from pixell import enmap, curvedsky

    tmap = enmap.ndmap(map, wcs)
    almsd = curvedsky.map2alm(tmap, lmax=settings.lmax)
    cl = curvedsky.alm2cl(almsd)

    plot_cl(cl, settings, plt_func, plt_camb, plt_noise, c_ells, title, save_name, save)

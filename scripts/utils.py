import os
import json

import matplotlib.pyplot as plt
import numpy as np

from pixell import enmap, curvedsky

from astropy import units as u

import h5py


def safe_makedirs(dir, verbose=False):
    "Create a directory if it does not exist. Handles a race condition"
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            if verbose:
                print(f"Created directory {dir}")
        except FileExistsError:
            pass


def save_data(file_path, data_dict, verbose=False):
    if verbose:
        print("Saving data to", file_path, end="...")

    with h5py.File(file_path, "a") as hf:
        for key, value in data_dict.items():
            if isinstance(value, dict):
                grp = hf.create_group(key)
                for k, v in value.items():
                    grp[k] = json.dumps(v)

                continue

            if key in hf:
                # Resize the dataset to accommodate the new data
                hf[key].resize((hf[key].shape[0] + value.shape[0],) + value.shape[1:])  # type: ignore
                # Append the new data
                hf[key][-value.shape[0] :] = value  # type: ignore
            else:
                # Create a new dataset for this key
                hf.create_dataset(key, data=value, maxshape=(None,) + value.shape[1:])
    if verbose:
        print("done")


def load_data(data_file, keys, start_index=None, end_index=None, verbose=False):
    if isinstance(keys, str):
        keys = [keys]

    data = {}

    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        for key in keys:
            if verbose:
                print("Loading data", key, "from", data_file)

            kv = hdf.get(key, None)
            if kv is None:
                raise ValueError(f"Key {key} not found in {data_file}")

            if start_index is not None and end_index is not None:
                if verbose:
                    print(" => Loading data from", start_index, "to", end_index)
                data[key] = np.array(kv[start_index:end_index])  # type: ignore
            else:
                data[key] = np.array(kv[()])  # type: ignore

    return data


def load_single_data(data_file, key, index, verbose=False):
    if verbose:
        print("Loading data", key, "from", data_file, "index", index)

    with h5py.File(data_file, "r", swmr=True, locking=False) as hdf:
        kv = hdf.get(key, None)
        if kv is None:
            raise ValueError(f"Key {key} not found in {data_file}")
        else:
            return np.array(kv[index])  # type: ignore


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
    tmap = enmap.ndmap(map, wcs)
    almsd = curvedsky.map2alm(tmap, lmax=settings.lmax)
    cl = curvedsky.alm2cl(almsd)

    plot_cl(cl, settings, plt_func, plt_camb, plt_noise, c_ells, title, save_name, save)


def get_radii(r_min, r_max):
    # For the radii we follow Table 2. of Smith and Zaldarriaga which gives a greater density of points near reionization and recombination.
    #
    # Spacing for all ranges but the last row are linear, with the last row having log spacing.
    #
    # radii are in Mpc
    radii = []
    #          start,  stop, resolution
    ranges = [
        (0, 9500, 150),
        (9500, 11000, 300),
        (11000, 13800, 150),
        (13800, 14600, 400),
        (14600, 16000, 100),
        (16000, 50000, 100),
    ]

    for r in ranges:
        start = max(r_min, r[0])
        end = min(r_max, r[1])

        if start > end:
            continue

        if r == ranges[-1]:  # For the last range, use logspace
            temp_radii = np.logspace(np.log10(start), np.log10(end), num=r[2])
        else:
            temp_radii = np.linspace(start, end, num=r[2], endpoint=False)

        radii.extend(temp_radii)

    radii = np.array([r for r in radii if r_min <= r < r_max])
    drs = np.diff(radii) / 2.
    return radii, drs

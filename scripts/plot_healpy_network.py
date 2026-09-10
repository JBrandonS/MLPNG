#!/usr/bin/env python3
"""
Generate a network architecture diagram for the HEALPix-based deep encoder.

This script uses plotneuralnet to visualize the network architecture used in trainer_unlensed.py
For nside=128 with npol=2, shows the encoder blocks, flattening, and task-specific heads.

Usage:
    python scripts/plot_healpy_network.py
    # Then run: bash plotneuralnet/tikzmake.sh healpy_network
"""

import sys
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from plotneuralnet.pycore.blocks import to_Conv

# Add plotneuralnet to path (assumes it's cloned in the repo root or available)
sys.path.append(os.path.join(os.path.dirname(__file__), "plotneuralnet"))

try:
    from pycore.tikzeng import *
except ImportError:
    print("Error: plotneuralnet not found. Clone it with:")
    print("  git clone https://github.com/HarisIqbal88/plotneuralnet.git")
    sys.exit(1)


# def create_healpy_network_diagram():


def generate_cmb_map_image(nside=128, output_path="cmb_map.png"):
    """Generate a 3D spherical CMB map with random noise.

    Args:
        nside: HEALPix nside resolution
        output_path: Path to save the PNG file
    """
    # Generate random Gaussian spherical map with smoothness
    np.random.seed(42)
    npix = hp.nside2npix(nside)
    cmb_map = np.random.randn(npix)
    cmb_map = hp.smoothing(cmb_map, fwhm=np.radians(30))

    # Create 3D sphere plot
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # Create sphere mesh
    u = np.linspace(0, 2 * np.pi, 64)
    v = np.linspace(0, np.pi, 32)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Map healpy data to sphere (simplified sampling)
    colors = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            theta = v[j]
            phi = u[i]
            pix = hp.ang2pix(nside, theta, phi)
            colors[i, j] = cmb_map[pix]

    # Normalize colors
    norm = Normalize(vmin=colors.min(), vmax=colors.max())
    colors_normalized = cm.viridis(norm(colors))

    # Plot surface with lighting effect
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors_normalized,
        shade=False,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
    )

    # Set viewing angle for 3D effect
    ax.view_init(elev=20, azim=45)
    ax.set_xlim([-0.72, 0.72])
    ax.set_ylim([-0.72, 0.72])
    ax.set_zlim([-0.72, 0.72])
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(
        output_path, bbox_inches="tight", pad_inches=0, dpi=100, transparent=True
    )
    plt.close()

    return output_path


def create_healpy_sphere_column(
    name_prefix, nside, n_channels, start_offset=(0, 0, 0), direction="east", label=""
):
    """Create a column of n HEALPy spheres at a given resolution.

    Args:
        name_prefix: Prefix for layer names (e.g., "input")
        nside: HEALPix nside parameter (resolution)
        n_channels: Number of channels to display
        start_offset: Starting position tuple (x, y, z)
        direction: Direction to stack ("east", "south", "west", "north")
        label: Optional label for the spheres

    Returns:
        List of plotneuralnet layer commands and connections
    """
    layers_code = []
    npix = 12 * nside**2

    # Map direction to offset increment
    dir_offsets = {
        "east": (1.0, 0, 0),
        "west": (-1.0, 0, 0),
        "south": (0, -1.0, 0),
        "north": (0, 1.0, 0),
    }
    dx, dy, dz = dir_offsets.get(direction, (1.0, 0, 0))

    # Create sphere for each channel
    for i in range(n_channels):
        sphere_name = f"{name_prefix}_ch{i}"

        if i == 0:
            # First sphere
            offset = start_offset
            to_layer = None
            height = 20
            depth = 20
        else:
            # Subsequent spheres
            prev_sphere = f"{name_prefix}_ch{i-1}"
            offset = (i * dx, i * dy, i * dz)
            to_layer = f"({prev_sphere}-{direction})"
            height = 20
            depth = 20

        # Create Conv layer to represent sphere
        layer_cmd = to_Conv(
            name=sphere_name,
            s_filer=20,
            n_filer=1,
            offset=f"({offset[0]},{offset[1]},{offset[2]})",
            to=to_layer if to_layer else f"({offset[0]},{offset[1]},{offset[2]})",
            height=height,
            depth=depth,
            width=1,
            caption=f"Ch{i}\nnside={nside}\n({npix})",
        )
        layers_code.append(layer_cmd)

        # Connect to previous sphere
        if i > 0:
            prev_sphere = f"{name_prefix}_ch{i-1}"
            layers_code.append(to_connection(prev_sphere, sphere_name))

    return layers_code, f"{name_prefix}_ch{n_channels-1}"


def create_healpy_network_diagram():
    """Create the HEALPix deep-encoder diagram for trainer_unlensed.py (3 shapes).

    Matches the architecture built by ``build_deep_task_model`` with the
    defaults used in ``main()``:

        nside=128, npol=2, pool_p=2  ->  depth = log_4(128) = 3 encoder blocks

    Per-level schedule (pool_p=2 halves nside by a factor of 4 each block):
        Level nsides:  [128, 32, 8, 2]
        Level npixels: [196608, 12288, 768, 48]
        Channels:      [2, 16, 32, 64]   (fin -> fout per block)
        K schedule:    [5, 5, 5]

    Each DeepEncoderBlock is a double HealpyChebyshev (K=5) convolution
    followed by Dropout + HealpyPool. After 3 blocks the (48, 64) feature map
    is flattened to 3072 and fed to three task-specific heads, one per shape:

        local        -> [32, 32, 1]
        equilateral  -> [64, 64, 32, 32, 1]
        orthogonal   -> [64, 32, 32, 1]
    """

    arch = [
        # Header and coordinate system
        to_head("./plotneuralnet"),
        to_cor(),
        to_begin(),
        "\\usetikzlibrary{calc}",
        # ---- Named colors for legend swatches ----
        "\\colorlet{legconv}{\\ConvColor}",
        "\\colorlet{legpool}{\\PoolColor}",
        "\\colorlet{legdense}{\\FcColor}",
        # ---- Legend (Top Left, drawn first so it sits in the background) ----
        "\\node[anchor=north west,font=\\small,inner sep=0.3cm,fill=white,draw=gray,rounded corners=2pt] at (-8,6) {\\begin{tabular}{cl} "
        "\\tikz{\\fill[legconv] (0,0) rectangle (0.45,0.45);} & Chebyshev Conv (ReLU) \\\\[2pt] "
        "\\tikz{\\fill[legpool,opacity=0.5] (0,0) rectangle (0.45,0.45);} & Pooling \\\\[2pt] "
        "\\tikz{\\fill[legdense] (0,0) rectangle (0.45,0.45);} & Dense / Output \\end{tabular}};",
        # ---- CMB Input Map: Spherical 3D visualization ----
        "\\node[outer sep=0pt,inner sep=0pt,anchor=center] (cmb) at (-3,0,0) {\\includegraphics[width=9.5cm]{cmb_map.png}};",
        # ---- Encoder Block 1: double Chebyshev (K=5) + Pool ----
        # fin=2 -> fout=16   (nside 128 -> 32, npix 196608 -> 12288)
        to_ConvConvRelu(
            name="b1",
            s_filer=" ",
            n_filer=(16, 16),
            offset="(2,0,0)",
            to="(1.5,0,0)",
            width=(3, 3),
            height=40,
            depth=40,
            caption="Block 1",
        ),
        "\\draw [connection] (-0.8,0,0) -- node {\\midarrow} (b1-west);",
        '\\path ($(b1-nearsoutheast)+(0,-0.7,0)$) edge [draw=none,"\\small nside = 128",sloped,pos=0.35,text centered] ($(b1-farsoutheast)+(0,-0.7,0)$);',
        to_Pool(
            name="p1",
            offset="(0,0,0)",
            to="(b1-east)",
            width=1,
            height=35,
            depth=35,
            caption=" ",
        ),
        # ---- Encoder Block 2: double Chebyshev (K=5) + Pool ----
        # fin=16 -> fout=32  (nside 32 -> 8, npix 12288 -> 768)
        to_ConvConvRelu(
            name="b2",
            s_filer=" ",
            n_filer=(32, 32),
            offset="(2.5,0,0)",
            to="(p1-east)",
            width=(3.5, 3.5),
            height=32,
            depth=32,
            caption="Block 2",
        ),
        to_connection("p1", "b2"),
        '\\path ($(b2-nearsoutheast)+(0,-0.7,0)$) edge [draw=none,"\\small nside = 32",sloped,pos=0.4,text centered] ($(b2-farsoutheast)+(0,-0.7,0)$);',
        to_Pool(
            name="p2",
            offset="(0,0,0)",
            to="(b2-east)",
            width=1,
            height=28,
            depth=28,
            caption=" ",
        ),
        # ---- Encoder Block 3: double Chebyshev (K=5) + Pool ----
        # fin=32 -> fout=64  (nside 8 -> 2, npix 768 -> 48)
        to_ConvConvRelu(
            name="b3",
            s_filer=" ",
            n_filer=(64, 64),
            offset="(2.5,0,0)",
            to="(p2-east)",
            width=(5, 5),
            height=24,
            depth=24,
            caption="Block 3",
        ),
        to_connection("p2", "b3"),
        '\\path ($(b3-nearsoutheast)+(0,-0.7,0)$) edge [draw=none,"\\small nside = 8",sloped,pos=0.5,text centered] ($(b3-farsoutheast)+(0,-0.7,0)$);',
        to_Pool(
            name="p3",
            offset="(0,0,0)",
            to="(b3-east)",
            width=1,
            height=20,
            depth=20,
            caption=" ",
        ),
        # ---- Flatten: (48, 64) -> 3072 ----
        # Use the dense (\FcColor) fill so the flatten slab matches the heads.
        "\\pic[shift={(2.5,0,0)}] at (p3-east) {Box={name=flat,caption=Flatten,"
        "xlabel={{ , }},zlabel= ,fill=\\FcColor,height=4,width=2,depth=40}};",
        # 3072 size label, rotated to follow the flatten slab's depth diagonal
        '\\path ($(flat-nearsoutheast)+(0.7,-0.00,0)$) edge [draw=none,"\\small 3072",sloped,pos=0.1text centered] ($(flat-farsoutheast)+(0.7,-0.00,0)$);',
        to_connection("p3", "flat"),
        # ---- Task-specific heads (one per shape) ----
        # Local head: [32, 32, 1]
        "\\pic[shift={(3.5,2.5,0)}] at (flat-east) {Box={name=local,caption=$f_{\\mathrm{NL}}^{\\mathrm{loc}}$,fill=\\FcColor,height=1,width=1,depth=1}};",
        "\\draw [connection] (flat-east) -- (local-west);",
        # Equilateral head: [64, 64, 32, 32, 1]
        "\\pic[shift={(3.5,0,0)}] at (flat-east) {Box={name=equil,caption=$f_{\\mathrm{NL}}^{\\mathrm{equil}}$,fill=\\FcColor,height=1,width=1,depth=1}};",
        "\\draw [connection] (flat-east) -- (equil-west);",
        # Orthogonal head: [64, 32, 32, 1]
        "\\pic[shift={(3.5,-2.5,0)}] at (flat-east) {Box={name=ortho,caption=$f_{\\mathrm{NL}}^{\\mathrm{ortho}}$,fill=\\FcColor,height=1,width=1,depth=1}};",
        "\\draw [connection] (flat-east) -- (ortho-west);",
        to_end(),
    ]

    return arch


def main():
    """Generate the LaTeX file for the network diagram.

    Usage examples for create_healpy_sphere_column():

        # Create a column of 2 HEALPy spheres at nside=128
        sphere_layers, last_layer = create_healpy_sphere_column(
            name_prefix="input",
            nside=128,
            n_channels=2,
            start_offset=(0, 0, 0),
            direction="north"
        )

        # Create a column of 4 spheres stacked east
        sphere_layers, last_layer = create_healpy_sphere_column(
            name_prefix="encoder_out",
            nside=64,
            n_channels=4,
            direction="east"
        )

        # Add to arch with unpacking
        arch = [
            to_head('./plotneuralnet'),
            to_cor(),
            to_begin(),
            *sphere_layers,  # Unpacks all sphere layers
            # ... rest of architecture
            to_end()
        ]

    """
    # Generate CMB map image first
    cmb_image_path = generate_cmb_map_image()
    print(f"Generated CMB map: {cmb_image_path}")

    arch = create_healpy_network_diagram()
    namefile = "healpy_network"
    to_generate(arch, namefile + ".tex")
    print(f"Generated {namefile}.tex")
    print("\nTo compile to PDF, run:")
    print(f"  bash plotneuralnet/tikzmake.sh {namefile}")
    print(f"\nOr compile manually:")
    print(f"  cd plotneuralnet/")
    print(f"  pdflatex -interaction=nonstopmode {namefile}.tex")


if __name__ == "__main__":
    main()

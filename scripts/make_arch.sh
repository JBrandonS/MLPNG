#!/bin/bash
# Generate the HEALPix network architecture diagram.
# Runs plot_healpy_network.py (which also regenerates cmb_map.png) and compiles
# the resulting healpy_network.tex to PDF. Does NOT use tikzmake.sh, which
# assumes the python file is named after the .tex target.

set -e

python plot_healpy_network.py
pdflatex -interaction=nonstopmode healpy_network.tex

# Clean up LaTeX intermediates (keep the .tex so the source is inspectable)
rm -f healpy_network.aux healpy_network.log

if [[ "$OSTYPE" == "darwin"* ]]; then
    open healpy_network.pdf
else
    xdg-open healpy_network.pdf
fi

#! /usr/bin/env python3

# Script downloads the alm_l and alm_nl files from the Heidelberg server
# the files will be saved in current directory, so run this script in the directory where you want the files to be saved

import requests

for num in range(1, 1001):
    URL1 = (
        "https://dc.zah.uni-heidelberg.de/elsnersim/q/s/static/alm_l_"
        + str(num).zfill(4)
        + "_v3.fits"
    )
    URL2 = (
        "https://dc.zah.uni-heidelberg.de/elsnersim/q/s/static/alm_nl_"
        + str(num).zfill(4)
        + "_v3.fits"
    )

    with requests.Session() as s, open(
        "alm_l_" + str(num).zfill(4) + "_v3.fits", "wb"
    ) as f:
        f.write(s.get(URL1).content)

    with requests.Session() as s, open(
        "alm_nl_" + str(num).zfill(4) + "_v3.fits", "wb"
    ) as f:
        f.write(s.get(URL2).content)

    print(f"Finished {num}")

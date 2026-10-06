"""Create a SYNTHETIC demonstration station file so that the pipeline can be run without the
restricted terrestrial data.

The file has the same format as the real input (columns X, Y, LATITUDE, LONGITUDE, CBA; UTM zone
49S, metres; CBA in mGal) and is written to <stage>/data/gravity_requested_area.csv for the three
stages. Station positions are random (43 stations, minimum spacing 450 m, in the 20 x 20 km window
of the study) and the CBA values are one synthetic scene of prior A plus a constant of 125 mGal.
Results obtained with this file are NOT the results of the paper.

Usage (from the repository root):  python demo/make_demo_data.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "3_comparison_ggmplus"))
from geometry import convex_hull  # noqa: E402
from priors_cfg import SCENARIOS  # noqa: E402
from scenes import ProfileBank, SceneGenerator  # noqa: E402

SEED = 2026


def main():
    rng = np.random.default_rng(SEED)
    pts = []
    while len(pts) < 43:
        p = rng.uniform([390_800.0, 9_130_200.0], [410_300.0, 9_150_000.0])
        if all(np.hypot(*(p - q)) >= 450.0 for q in pts):
            pts.append(p)
    xy = np.array(pts)
    gen = SceneGenerator(xy, convex_hull(xy), priors=SCENARIOS["A_clutter"], bank=ProfileBank())
    scene = gen.sample(rng)
    lon, lat = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True).transform(xy[:, 0], xy[:, 1])
    df = pd.DataFrame(dict(X=xy[:, 0].round(3), Y=xy[:, 1].round(3), LATITUDE=np.round(lat, 4),
                           LONGITUDE=np.round(lon, 4), CBA=np.round(125.0 + scene.d, 3)))
    for stage in ("1_detectability", "2_simulation_ml", "3_comparison_ggmplus"):
        d = os.path.join(ROOT, stage, "data")
        os.makedirs(d, exist_ok=True)
        df.to_csv(os.path.join(d, "gravity_requested_area.csv"), index=False)
        with open(os.path.join(d, "DEMO_DATA_README.txt"), "w") as f:
            f.write("gravity_requested_area.csv in this folder is SYNTHETIC demonstration data "
                    "(demo/make_demo_data.py, seed 2026). Replace it with the real terrestrial data "
                    "to reproduce the paper.\n")
    print(f"synthetic demo stations written ({len(df)} stations, {scene.targets.shape[0]} synthetic targets)")


if __name__ == "__main__":
    main()

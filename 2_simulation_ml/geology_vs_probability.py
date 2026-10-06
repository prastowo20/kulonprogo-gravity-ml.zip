"""Compare the reference probability map with the mapped geology (revision of 5 Oct 2026, Section 4.7).

Run in 2_simulation_ml after apply_real.py and after 3_comparison_ggmplus/psg_density.py:
    GEOLOGY_DIR=/path/to/folder python geology_vs_probability.py

Geology: digital geological map of Indonesia (Pusat Penelitian dan Pengembangan Geologi, Geological Agency of
Indonesia), layer Geologi.shp, unit code SANDI. The map data are not redistributed; set GEOLOGY_DIR to the folder
holding Geologi.shp. The model never used geology, so the comparison is independent of the training.

Outputs results/geology_vs_probability.json with
- per unit: share of the station hull (cells and area), mean map probability, cells with p >= 0.3, mean elevation;
- the unit at P1, P2 and the Gajah summit, with the distance to the nearest andesite intrusion (a1);
- two chance levels for the coincidence of the candidate zones with a1:
  (i) two independent random points in the hull both on a1 (share of a1 squared);
  (ii) a random-shift test that keeps the spatial structure of both maps: the geological map is translated
       relative to the probability map by every offset of a 250 m lattice with 2 km <= |offset| <= 6 km, and the
       statistics are recomputed (count of p >= 0.3 cells on a1; both candidate-zone cells on a1; difference of
       mean p on and off a1);
- an elevation check (surface heights of the GGMplus tile, linear interpolation): mean elevation per unit and
  Spearman correlation of p with elevation, to judge whether the coincidence could be a topographic artefact;
- a station-level check that does not involve the model: mean CBA of the stations on and off a1, raw and after
  removing a least-squares plane, under H1 and H2 (3_comparison_ggmplus/data/derived/psg_cba_H1_H2.csv), with the same
  random-shift test and a leave-one-station-out range.
"""
from __future__ import annotations

import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.interpolate import griddata
from scipy.stats import spearmanr
from shapely.geometry import MultiPoint, Point

from features import FeatureExtractor
from geometry import convex_hull, load_psg

GEOLOGY_DIR = os.environ.get("GEOLOGY_DIR", "")   # folder with Geologi.shp (not redistributed)
GGM_XLS = os.environ.get("GGM_XLS", "../3_comparison_ggmplus/data/ggmplus_oasis.xls")
CBA_H1_H2 = os.environ.get("CBA_H1_H2", "../3_comparison_ggmplus/data/derived/psg_cba_H1_H2.csv")  # psg_density.py
SITES = {"P1": (400000.0, 9145000.0), "P2": (402000.0, 9136000.0), "Gajah summit": (401448.0, 9140279.0)}
P_HI = 0.3
SHIFT_STEP, SHIFT_MIN, SHIFT_MAX = 250.0, 2000.0, 6000.0


def elevation_at(cx, cy):
    sys.path.insert(0, "../3_comparison_ggmplus")
    from ggm_real import load_grid
    x, y, z, _ = load_grid(GGM_XLS)
    return griddata(np.column_stack([x.ravel(), y.ravel()]), z.ravel(), (cx, cy), method="linear")


def plane_residual(x, y, c):
    A = np.column_stack([np.ones_like(x), (x - x.mean()) / 1e3, (y - y.mean()) / 1e3])
    coef, *_ = np.linalg.lstsq(A, c, rcond=None)
    return c - A @ coef


def station_check(a1, shifts):
    """CBA of the stations on and off the mapped andesite intrusions (no model involved)."""
    st = pd.read_csv(CBA_H1_H2)
    x, y = st.X.to_numpy(), st.Y.to_numpy()
    on = shapely.contains_xy(a1, x, y)
    out = dict(n_stations=int(len(st)), n_on_a1=int(on.sum()),
               mean_grid_elevation_on_a1_m=round(float(st.h_grid_m[on].mean()), 0),
               mean_grid_elevation_off_a1_m=round(float(st.h_grid_m[~on].mean()), 0))
    for col, lab in (("CBA_H1", "H1"), ("CBA_H2_2670", "H2")):
        c = st[col].to_numpy()
        r = plane_residual(x, y, c)
        d_raw, d_res = c[on].mean() - c[~on].mean(), r[on].mean() - r[~on].mean()
        ge, sd = [], []
        for dx, dy in shifts:
            s = shapely.contains_xy(a1, x - dx, y - dy)
            sd.append(np.nan if s.all() or not s.any() else float(r[s].mean() - r[~s].mean()))
            ge.append(np.nan if np.isnan(sd[-1]) else float(sd[-1] >= d_res))
        jk = []
        for i in range(len(c)):
            m = np.arange(len(c)) != i
            rr = plane_residual(x[m], y[m], c[m])
            jk.append(rr[on[m]].mean() - rr[~on[m]].mean())
        out[lab] = dict(mean_cba_on_a1=round(float(c[on].mean()), 1), mean_cba_off_a1=round(float(c[~on].mean()), 1),
                        diff_raw_mgal=round(float(d_raw), 1), diff_detrended_mgal=round(float(d_res), 1),
                        shift_fraction_ge_detrended=round(float(np.nanmean(ge)), 4),
                        shift_max_detrended_mgal=round(float(np.nanmax(sd)), 1),
                        jackknife_range_detrended_mgal=[round(float(min(jk)), 1), round(float(max(jk)), 1)])
    return out


def main():
    if not os.path.exists(os.path.join(GEOLOGY_DIR, "Geologi.shp")):
        sys.exit("Set GEOLOGY_DIR to the folder holding Geologi.shp of the digital geological map of Indonesia "
                 "(Pusat Penelitian dan Pengembangan Geologi); the map is not redistributed with this code.")
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    maps = np.load("results/real_maps.npz")
    p = maps["pA"]
    X, Y = np.meshgrid(fe.xs, fe.ys)
    cx, cy = X.ravel()[fe.inside], Y.ravel()[fe.inside]
    assert np.allclose(np.column_stack([cx, cy]), maps["cells"])

    g = gpd.read_file(os.path.join(GEOLOGY_DIR, "Geologi.shp"), bbox=(109.94, -7.93, 110.24, -7.63)).to_crs(32749)
    assert set(g.DATA.astype(object)) == {"PUSLITBANG GEOLOGI"}
    g["SANDI"] = g["SANDI"].astype(object)
    cells = gpd.GeoDataFrame({"p": p}, geometry=gpd.points_from_xy(cx, cy), crs=32749)
    j = gpd.sjoin(cells, g[["SANDI", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    assert j.SANDI.notna().all() and len(j) == len(p)
    unit = j.SANDI.to_numpy()
    z = elevation_at(cx, cy)
    assert np.isfinite(z).all()

    hull = MultiPoint([tuple(q) for q in xy]).convex_hull
    gh = g.clip(hull)
    area = gh.assign(a=gh.area / 1e6).groupby("SANDI").a.sum()
    hi = p >= P_HI
    units = {}
    for u in sorted(set(unit)):
        m = unit == u
        units[u] = dict(cells=int(m.sum()), share_of_hull_cells=round(m.mean(), 4),
                        area_km2=round(float(area.get(u, 0.0)), 2),
                        share_of_hull_area=round(float(area.get(u, 0.0) / area.sum()), 4),
                        mean_p=round(float(p[m].mean()), 4), cells_p_ge_0_3=int((m & hi).sum()),
                        mean_elevation_m=round(float(z[m].mean()), 0))

    a1 = shapely.union_all(g[g.SANDI == "a1"].geometry.to_numpy())
    shapely.prepare(a1)
    sites = {}
    for name, (x, y) in SITES.items():
        pt = Point(x, y)
        hit = g[g.contains(pt)]
        sites[name] = dict(unit=list(hit.SANDI), formation=list(hit.FORMASI.astype(object)),
                           distance_to_andesite_intrusion_m=round(float(a1.distance(pt)), 0),
                           units_within_100_m=sorted(set(g[g.distance(pt) < 100].SANDI)))

    # observed statistics
    on = unit == "a1"
    zone_idx = [int(np.argmin(np.hypot(cx - x, cy - y))) for x, y in (SITES["P1"], SITES["P2"])]
    obs = dict(n_hi_on_a1=int((on & hi).sum()), both_zones_on_a1=bool(on[zone_idx].all()),
               dmean_p=float(p[on].mean() - p[~on].mean()))

    # random-shift test: cell (x, y) lies on the shifted map's a1 if (x - dx, y - dy) lies on a1
    k = np.arange(-SHIFT_MAX, SHIFT_MAX + 1, SHIFT_STEP)
    DX, DY = np.meshgrid(k, k)
    r = np.hypot(DX, DY)
    sel = (r >= SHIFT_MIN) & (r <= SHIFT_MAX)
    shifts = np.column_stack([DX[sel], DY[sel]])
    n_hi, both, dmean, share = [], [], [], []
    for dx, dy in shifts:
        s = shapely.contains_xy(a1, cx - dx, cy - dy)
        n_hi.append(int((s & hi).sum()))
        both.append(bool(s[zone_idx].all()))
        dmean.append(float(p[s].mean() - p[~s].mean()) if s.any() and (~s).any() else np.nan)
        share.append(float(s.mean()))
    n_hi, both, dmean, share = map(np.asarray, (n_hi, both, dmean, share))
    shift_test = dict(
        n_shifts=int(len(shifts)), step_m=SHIFT_STEP, min_m=SHIFT_MIN, max_m=SHIFT_MAX,
        p_n_hi_on_a1=round(float(np.mean(n_hi >= obs["n_hi_on_a1"])), 4),
        p_both_zones_on_a1=round(float(np.mean(both)), 4),
        p_dmean=round(float(np.nanmean(dmean >= obs["dmean_p"])), 4),
        n_hi_on_a1_null_median=float(np.median(n_hi)), n_hi_on_a1_null_p95=float(np.percentile(n_hi, 95)),
        share_a1_null_median=round(float(np.median(share)), 4))

    rho_all = spearmanr(p, z)
    rho_off = spearmanr(p[~on], z[~on])
    rho_on = spearmanr(p[on], z[on])
    elev = dict(spearman_p_vs_elevation_all=[round(float(rho_all.statistic), 3), float(rho_all.pvalue)],
                spearman_off_a1=[round(float(rho_off.statistic), 3), float(rho_off.pvalue)],
                spearman_on_a1=[round(float(rho_on.statistic), 3), float(rho_on.pvalue)],
                mean_elevation_hi_cells_m=round(float(z[hi].mean()), 0),
                mean_elevation_hull_m=round(float(z.mean()), 0),
                source="surface heights of the GGMplus tile (linear interpolation to the 0.5 km cells)")

    bodies = sorted((q for q in getattr(a1, "geoms", [a1]) if q.intersects(hull)), key=lambda q: -q.area)
    a1_bodies = [dict(area_km2=round(q.area / 1e6, 1), area_in_hull_km2=round(q.intersection(hull).area / 1e6, 1))
                 for q in bodies if q.intersection(hull).area > 1e6]
    for name, (x, y) in SITES.items():
        sites[name]["distance_to_a1_boundary_m"] = round(float(a1.boundary.distance(Point(x, y))), 0)
        for u in ("Tmj", "da"):
            sites[name][f"distance_to_{u}_m"] = round(float(g[g.SANDI == u].distance(Point(x, y)).min()), 0)

    share_a1 = units["a1"]["share_of_hull_cells"]
    out = dict(source="digital geological map of Indonesia, Pusat Penelitian dan Pengembangan Geologi (DATA attribute)",
               hull_area_km2=round(hull.area / 1e6, 1), cells_in_hull=int(len(p)), cells_p_ge_0_3=int(hi.sum()),
               cells_p_ge_0_3_on_a1=obs["n_hi_on_a1"], observed=obs, units=units, sites=sites, a1_bodies=a1_bodies,
               chance_two_random_points_on_a1=round(share_a1 ** 2, 3), shift_test=shift_test, elevation=elev,
               stations=station_check(a1, shifts))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/geology_vs_probability.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k not in ("units", "sites")}, indent=1, ensure_ascii=False))
    for u, d in units.items():
        print(u, d)


if __name__ == "__main__":
    main()

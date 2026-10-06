"""R80 table: smallest radius with POD >= 0.8 (log-linear interpolation between tested radii)."""
import numpy as np
import pandas as pd

ORDER = ["PSG-43", "Grid 2.83 km", "Grid 2.00 km", "Grid 1.41 km", "Grid 1.00 km", "Grid 0.71 km", "Grid 0.50 km"]


def r_at(pod, R, level=0.8):
    R = np.asarray(R, float)
    pod = np.maximum.accumulate(np.asarray(pod, float))  # guard against MC wiggles
    if pod[0] >= level:
        return R[0], "<="
    if pod[-1] < level:
        return R[-1], ">"
    k = np.argmax(pod >= level)
    x0, x1 = np.log(R[k - 1]), np.log(R[k])
    f = (level - pod[k - 1]) / (pod[k] - pod[k - 1])
    return float(np.exp(x0 + f * (x1 - x0))), "="


def r80_table(df, level=0.8):
    out = []
    for (g, zt, drho, sigma), sub in df.groupby(["geometry", "zt_m", "drho", "sigma"]):
        sub = sub.sort_values("R_m")
        for det in ("D1", "D2"):
            val, flag = r_at(sub[f"pod_{det}"], sub["R_m"], level)
            out.append(dict(geometry=g, n_stations=int(sub.n_stations.iloc[0]), zt_m=zt, drho=drho, sigma=sigma,
                            detector=det, R80_m=round(val), flag=flag))
    t = pd.DataFrame(out)
    t["geometry"] = pd.Categorical(t["geometry"], ORDER, ordered=True)
    return t.sort_values(["detector", "sigma", "drho", "zt_m", "geometry"])


if __name__ == "__main__":
    df = pd.read_csv("results/pod_results.csv")
    t = r80_table(df)
    t.to_csv("results/R80_table.csv", index=False)
    t["R80"] = t.apply(lambda r: f"{r.flag if r.flag != '=' else ''}{r.R80_m / 1000:.2f}", axis=1)
    pd.set_option("display.width", 200)
    print(t.pivot_table(index=["detector", "sigma", "drho", "zt_m"], columns="geometry", values="R80", aggfunc="first", observed=True).to_string())

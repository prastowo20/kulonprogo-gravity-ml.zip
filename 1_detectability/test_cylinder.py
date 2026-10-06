"""Validation of the cylinder forward model (run: python test_cylinder.py)."""
import numpy as np

from cylinder import Profile, gz_bruteforce, gz_cylinder, gz_line, gz_on_axis

rows = []


def check(name, got, ref, rtol):
    err = np.max(np.abs(np.asarray(got) - np.asarray(ref)) / np.maximum(np.abs(ref), 1e-12))
    ok = err <= rtol
    rows.append((name, err, rtol, ok))
    return ok


cases = [(250.0, 100.0, 2100.0), (1000.0, 100.0, 2100.0), (3000.0, 100.0, 2100.0), (500.0, 1000.0, 3000.0)]
for R, zt, zb in cases:
    tag = f"R={R:.0f} zt={zt:.0f} zb={zb:.0f}"
    # 1. on-axis closed form
    check(f"on-axis {tag}", gz_cylinder(0.0, R, zt, zb, 300.0), gz_on_axis(R, zt, zb, 300.0), 1e-10)
    # 2. far field -> vertical line mass
    rf = np.array([30.0, 50.0]) * max(R, zb)
    check(f"far-field line {tag}", gz_cylinder(rf, R, zt, zb, 300.0), gz_line(rf, R, zt, zb, 300.0), 2e-3)
    # 3. independent brute-force quadrature (inside, near edge on both sides, outside)
    rr = np.array([0.3 * R, 0.97 * R, 1.03 * R, 1.8 * R, R + 2000.0])
    cell = min(R / 150.0, zt / 8.0)
    check(f"brute-force {tag}", gz_cylinder(rr, R, zt, zb, 300.0), gz_bruteforce(rr, R, zt, zb, 300.0, cell=cell), 3e-3)
    # 4. continuity across r = R
    e = 1e-6 * R
    a, b = gz_cylinder(np.array([R - e, R + e]), R, zt, zb, 300.0)
    check(f"continuity r=R {tag}", a, b, 1e-4)
    # 5. convergence (double the quadrature)
    rr = np.linspace(0, 3 * R, 61)
    check(f"quadrature convergence {tag}", gz_cylinder(rr, R, zt, zb, 300.0),
          gz_cylinder(rr, R, zt, zb, 300.0, n_theta=8192, n_psi=1024), 1e-5)
    # 6. tabulated profile interpolation
    p = Profile(R, zt, zb)
    rq = np.random.default_rng(0).uniform(0, 20_000, 500)
    check(f"profile interp {tag}", 300.0 * p(rq), gz_cylinder(rq, R, zt, zb, 300.0), 2e-3)

# 7. infinite-slab limit: very wide cylinder, on axis, -> 2*pi*G*drho*H
R, zt, zb = 1e7, 100.0, 2100.0
slab = 2 * np.pi * 6.67430e-11 * 300.0 * (zb - zt) * 1e5
check("slab limit (R->inf)", gz_on_axis(R, zt, zb, 300.0), slab, 1e-3)

w = max(len(r[0]) for r in rows)
print(f"{'test':<{w}}  {'max rel err':>12}  {'tol':>8}  ok")
for name, err, tol, ok in rows:
    print(f"{name:<{w}}  {err:12.2e}  {tol:8.0e}  {'PASS' if ok else 'FAIL'}")
print("ALL PASS" if all(r[3] for r in rows) else "SOME TESTS FAILED")
# reference numbers used in the report
for R in (250, 500, 1000, 2000, 3000):
    print(f"peak (on-axis) R={R:5d} m, zt=100, H=2000, drho=300: {gz_on_axis(R, 100.0, 2100.0, 300.0):6.2f} mGal")

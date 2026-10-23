"""Additional KMC runs: (i) SAR versus packing fraction up to 0.3 for the clinical particle (100 kHz, 4.77 kA/m) and the robust
particle (85 kHz, limit); (ii) clinical particle at 4.0 / 4.5 / 5.0 kA/m for the clinical inversion; (iii) numerical convergence."""
import json, itertools, time
from multiprocessing import Pool
import sar_cluster as S
LIMIT, FE, SIG = 4.77e8, 0.724, 0.15
PART = {"clinical": (12.0, 1.4e4), "robust": (14.0, 2.6e4)}
def job(a):
    tag, part, f, H, ph, kw = a; d, K = PART[part]
    m, e = S.sar("dilute" if ph is None else "loose", ph, d, K, H, f, sig=SIG, **kw)
    return dict(tag=tag, particle=part, f=f, H=H, phi=ph, opts={k: v for k, v in kw.items() if k != "seeds"}, seeds=kw["seeds"], SAR_Fe=m / FE, se_Fe=e / FE)
if __name__ == "__main__":
    t0 = time.time(); J = []
    J += [("phi_curve", "clinical", 1e5, 4770.0, ph, dict(seeds=16, cycles=10)) for ph in (0.25, 0.30)]
    J += [("phi_curve", "robust", 8.5e4, LIMIT / 8.5e4, ph, dict(seeds=16, cycles=10)) for ph in (0.05, 0.15, 0.25, 0.30)]
    J += [("clin_H", "clinical", 1e5, H, ph, dict(seeds=16, cycles=10)) for H in (4000.0, 5000.0) for ph in (None, 0.05, 0.1, 0.15, 0.2)]
    J += [("clin_H", "clinical", 1e5, 4500.0, ph, dict(seeds=16, cycles=10)) for ph in (0.05, 0.15)]
    J += [("conv_ref", "clinical", 1e5, 4770.0, ph, dict(seeds=8, cycles=10)) for ph in (0.1, 0.2)]
    J += [("conv_N120", "clinical", 1e5, 4770.0, ph, dict(seeds=8, cycles=10, N=120)) for ph in (0.1, 0.2)]
    J += [("conv_steps600", "clinical", 1e5, 4770.0, ph, dict(seeds=8, cycles=10, steps=600)) for ph in (0.1, 0.2)]
    J += [("conv_cycles20", "clinical", 1e5, 4770.0, ph, dict(seeds=8, cycles=20)) for ph in (0.1, 0.2)]
    with Pool(2) as p: res = p.map(job, J, chunksize=1)
    json.dump(res, open("../data/extra_kmc.json", "w"), indent=1); print("done", round(time.time() - t0))

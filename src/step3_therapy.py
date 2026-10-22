import json, numpy as np, prostate as P, therapy as Th
SC = {"Planned (dispersed)": 3.80, "Clinical, φ = 0.1": 2.02, "Clinical, φ = 0.2": 0.82,
      "Clinical 50 kHz, φ = 0.1": 3.21, "Clinical 50 kHz, φ = 0.2": 1.16, "Robust 50 kHz, φ = 0.1": 5.77, "Robust 50 kHz, φ = 0.2": 4.38}
res = {}
for w in (2e-3, 3e-3, 5e-3):
    maps = []
    for sd in range(8):
        r = P.run(1.0, seed=sd, w=w); maps.append(r["T"][r["pro"]] - 37.0)      # temperature rise per W/gFe
    for name, s in SC.items():
        rows = []
        for m in maps:
            T = 37.0 + s * m; th = Th.summary(T)
            rows.append(dict(T90=float(np.percentile(T, 10)), T50=float(np.percentile(T, 50)), **th))
        res[f"{name}|{w}"] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
        res[f"{name}|{w}"].update(sd_gain98=float(np.std([x["gain98"] for x in rows])), SAR=s, w=w)
json.dump(res, open("../data/step3_therapy.json", "w"), indent=1)
for k, v in res.items():
    if v["w"] == 3e-3: print(k, {q: round(v[q], 2) for q in ("SAR", "T90", "T50", "TER98", "gain98", "gain50", "sd_gain98")})

"""Thermoradiotherapy effect of the prostate temperature maps (extended LQ, van Leeuwen et al. 2018; Kok et al. 2022).
Prostate cancer: alpha37 = 0.15 Gy^-1, alpha/beta = 1.5 Gy (van Leeuwen et al. 2018, Radiat Oncol 13:96);
thermal ratios and direct-kill Arrhenius parameters of SiHa cells (van Leeuwen et al. 2018, IJH 34:30) as the
only fully published set; decay mu = 0.5 1/h (clinical value). Schedule: 28 x 2.5 Gy, six weekly 60-min sessions,
heating immediately before or after the fraction (t = 0)."""
import numpy as np
from trbio import PAR, lq, k_heat
P = dict(PAR["SiHa"], a37=0.15, ab37=1.5, mu=0.5, rb=1.0)
# beta is kept temperature-independent (rb = 1): the SiHa beta ratio (0.42) was measured in a high alpha/beta line and,
# applied to a low alpha/beta tumour, would make mild heating reduce the EQD; P_SIHA keeps it for a sensitivity check.
P_SIHA = dict(PAR["SiHa"], a37=0.15, ab37=1.5, mu=0.5)
N_FX, N_HT, D = 28, 6, 2.5
def eqd_total(T, tint=0.0, p=P, t_ht=3600.0, direct=True):
    """total EQD2 (Gy) per voxel for the whole schedule; T = session temperature (steady)"""
    a37 = p["a37"]; b37 = a37 / p["ab37"]; den = a37 + b37 * 2.0
    e_rt = (a37 * D + b37 * D * D) / den
    a, b = lq(T, tint, p); e_ht = (a * D + b * D * D) / den
    if direct: e_ht = e_ht + k_heat(T, p) * t_ht / den
    return (N_FX - N_HT) * e_rt + N_HT * e_ht, N_FX * e_rt
def summary(T, tint=0.0, p=P):
    e, e0 = eqd_total(T, tint, p)
    return dict(D98=float(np.percentile(e, 2)), D50=float(np.percentile(e, 50)), EQD_RT=float(e0),
                TER98=float(np.percentile(e, 2) / e0), gain98=float(np.percentile(e, 2) - e0), gain50=float(np.percentile(e, 50) - e0))

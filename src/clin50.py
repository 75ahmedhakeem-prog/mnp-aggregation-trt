import json
import sar_cluster as S
out=[]
for f,H in ((5e4,9.54e3),(1e5,4.77e3)):
    for ph in (None,0.1,0.2):
        m,e=S.sar("dilute" if ph is None else "loose",ph,12.0,1.4e4,H,f,sig=0.15,seeds=16,cycles=12)
        out.append(dict(f=f,H=H,phi=ph,SAR_Fe=m/0.724,se_Fe=e/0.724)); print(out[-1],flush=True)
json.dump(out,open("../data/clin50.json","w"),indent=1); print("done")

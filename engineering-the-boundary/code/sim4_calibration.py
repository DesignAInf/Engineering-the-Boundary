from paths import FIGDIR, RESDIR
import numpy as np, json
import sim4_foraging_agents as S
from multiprocessing import Pool
def cell(args):
    L,k,s=args
    S.HSTRUCT = (L if np.isfinite(L) else 1000.0)
    out,_=S.run_condition(L,k,[s],3000,agents=("PA","BE"))
    return (L,k,out["BE"][0]["net"]-out["PA"][0]["net"])
if __name__=="__main__":
    jobs=[(L,k,s) for L in [25,50,np.inf] for k in [1.0,3.0,10.0] for s in range(12)]
    with Pool(2) as p: r=p.map(cell,jobs)
    res={}
    for L in [25,50,np.inf]:
        for k in [1.0,3.0,10.0]:
            d=np.array([x[2] for x in r if x[0]==L and x[1]==k])
            res[f"{L}_{k}"]=(float(d.mean()),float(d.std(ddof=1)/np.sqrt(len(d))))
            print(L,k,round(d.mean(),3),round(d.std(ddof=1)/np.sqrt(len(d)),3))
    json.dump(res,open(RESDIR + "sim4_calibration.json","w"))

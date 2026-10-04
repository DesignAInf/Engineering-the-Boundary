from paths import FIGDIR, RESDIR
import numpy as np, json
from itertools import permutations
from sim3_inversion import gen
from mini_dmbd import MiniDMBD
out=[]
for seed in range(3):
    Y,lab=gen(seed=seed)
    for init in range(2):
        m=MiniDMBD(seed=10+init).fit(Y,iters=40)
        p=m.q.argmax(1)
        acc=max((p==np.array(pp)[lab]).mean() for pp in permutations(range(3)))
        out.append(dict(seed=seed,init=init,acc=float(acc),mislabeled=int(round((1-acc)*len(lab)))))
    pri=np.full((len(lab),3),0.2); pri[np.arange(len(lab)),lab]=0.6
    for pf in [0.6,0.4]:
        pri=np.full((len(lab),3),(1-pf)/2); pri[np.arange(len(lab)),lab]=pf
        m=MiniDMBD(prior=pri,seed=1).fit(Y,iters=40)
        out.append(dict(seed=seed,focused=pf,acc=float((m.q.argmax(1)==lab).mean())))
for o in out: print(o)
json.dump(out,open(RESDIR + 'flat_prior_results.json','w'),indent=0)

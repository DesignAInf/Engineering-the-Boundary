from paths import FIGDIR, RESDIR
import json, numpy as np
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
r = json.load(open(RESDIR + "inversion_results.json"))
d = defaultdict(list)
for x in r: d[(tuple(x["shield"]), tuple(x["relabel"]))].append(x)
opts = list(d.keys())
br = {k: np.mean([x["breach"] for x in v]) for k, v in d.items()}
cmi = {k: v[0]["cmi"] for k, v in d.items()}
shield_opts = [k for k in opts if not k[1]]
xs = np.array([cmi[k] for k in shield_opts]); ys = np.array([br[k] for k in shield_opts])
rcorr = np.corrcoef(xs, ys)[0, 1]
H = 1000.0
kappas = [1, 5, 10, 20, 50, 100, 200, 500]
choice = []
for kap in kappas:
    G = {k: H * br[k] + kap * (len(k[0]) + len(k[1])) for k in opts}
    best = min(G, key=G.get)
    choice.append((kap, best, cmi[best], G[best]))
    print("kappa", kap, "chosen", best, "true CMI %.3f" % cmi[best])
summary = dict(r_breach_cmi=float(rcorr), choices=[(c[0], [list(c[1][0]), list(c[1][1])], c[2]) for c in choice],
               breach={str(k): br[k] for k in opts}, cmi={str(k): cmi[k] for k in opts})
json.dump(summary, open(RESDIR + "inversion_summary.json", "w"), indent=1)
print("corr breach-CMI", rcorr)
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
none = d[((), ())]; best_k = choice[3][1]
b0 = np.mean([x["breach_leakers"] for x in none], 0); b1 = np.mean([x["breach_leakers"] for x in d[best_k]], 0)
w = 0.38; xi = np.arange(4)
ax[0].bar(xi - w / 2, b0, w, color="#b2182b", label="no intervention")
ax[0].bar(xi + w / 2, b1, w, color="#2166ac", label="selected (κ=20)")
ax[0].axhline(np.mean([x["breach_max_other"] for x in none]), color="k", ls=":", lw=0.8)
ax[0].set_xticks(xi); ax[0].set_xticklabels([str(e) for e in (0.1, 0.4, 0.8, 1.2)], fontsize=7.5); ax[0].set_xlabel(r"leak strength $\varepsilon$")
ax[0].set_ylabel("breach [nats/step]"); ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
ax[0].set_title("(a) breach of leaking elements", fontsize=8.5)
ax[1].scatter(xs, ys, s=12, color="#2166ac")
for k in opts:
    if k[1]: ax[1].scatter(cmi[k], br[k], s=12, color="#999999", marker="^")
ax[1].set_xlabel(r"true factorisation gap $G$ [nats]"); ax[1].set_ylabel("detector breach [nats/step]")
ax[1].set_title("(b) breach vs gap, r=%.3f" % rcorr, fontsize=8.5)
ax[1].set_ylim(0, 0.85); ax[1].text(0.02, 0.78, "recruit options off scale (1.7-3.7)", fontsize=6.5, color="#666666")
ax[2].semilogx([c[0] for c in choice], [c[2] for c in choice], "o-", color="#2166ac", ms=3)
prev = None
for c in choice:
    if c[1] == prev: continue
    prev = c[1]
    ax[2].annotate("{" + ",".join(str(i) for i in c[1][0]) + "}", (c[0], c[2]), fontsize=6, xytext=(2, 3), textcoords="offset points")
ax[2].set_xlabel(r"cost per intervention $\kappa$"); ax[2].set_ylabel("true gap after [nats]")
ax[2].set_title("(c) result of the inversion", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIGDIR + "fig_inversion.pdf")

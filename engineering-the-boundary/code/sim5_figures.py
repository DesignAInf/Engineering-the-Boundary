from paths import FIGDIR, RESDIR
import json, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
R = json.load(open(RESDIR + "thermal_results.json"))
COL = dict(FB="#b2182b", PA="#ef8a62", BE="#2166ac", BEO="#1a9850")
S = {}
main = [r for r in R if r["life"] == 1000 and r["kappa"] == 2.0 and not r["extra"].get("omega_size") and r["seed"] < 16]
for k in ["FB", "PA", "BE", "BEO"]:
    rr = [r for r in main if r["kind"] == k]
    S[k] = {m: (float(np.mean([r[m] for r in rr])), float(1.96 * np.std([r[m] for r in rr], ddof=1) / np.sqrt(len(rr))))
            for m in ["cost", "net", "viab", "moves", "gap", "ctrl"]}
    S[k]["n"] = len(rr)
    print(k, {m: tuple(round(v, 4) for v in S[k][m]) for m in ["cost", "net", "viab", "moves", "gap"]}, len(rr))
tests = {}
for b in ["FB", "PA"]:
    for m in ["net", "viab", "gap"]:
        d = np.array([next(x[m] for x in main if x["kind"] == "BE" and x["seed"] == s) -
                      next(x[m] for x in main if x["kind"] == b and x["seed"] == s) for s in range(16)])
        w = stats.wilcoxon(d)
        tests[f"BE-{b}:{m}"] = dict(mean=float(d.mean()), ci=float(1.96 * d.std(ddof=1) / 4), p=float(w.pvalue),
                                   rel=float(d.mean() / np.mean([x[m] for x in main if x["kind"] == b])),
                                   frac_better=float(np.mean(d < 0)) if m in ("net", "gap") else float(np.mean(d > 0)))
for k, v in tests.items(): print(k, v)
# sweep
lifes = [100, 300, 1000, 3000]; kaps = [0.5, 4.0, 32.0]
adv = np.zeros((len(kaps), len(lifes))); se = np.zeros_like(adv); mv = np.zeros_like(adv)
for i, k in enumerate(kaps):
    for j, L in enumerate(lifes):
        rr = [r for r in R if r["life"] == L and r["kappa"] == k and not r["extra"]]
        seeds = sorted({r["seed"] for r in rr})
        d = []
        for s in seeds:
            fb = [r for r in rr if r["kind"] == "FB" and r["seed"] == s]; be = [r for r in rr if r["kind"] == "BE" and r["seed"] == s]
            if fb and be: d.append((fb[0]["net"] - be[0]["net"]) / fb[0]["net"])
        d = np.array(d); adv[i, j] = d.mean(); se[i, j] = d.std(ddof=1) / np.sqrt(len(d))
        mv[i, j] = np.mean([r["moves"] for r in rr if r["kind"] == "BE"])
print("relative net-cost reduction BE vs FB\n", np.round(adv * 100, 1)); print("se\n", np.round(se * 100, 1)); print("moves\n", mv)
# Occam
oms = [128, 10 ** 3, 10 ** 5, 10 ** 8, 10 ** 12]
occ = []
for om in oms:
    rr = [r for r in R if r["extra"].get("omega_size") == om]
    dl = np.concatenate([r["delays"] for r in rr]) if rr else np.array([])
    occ.append(dict(omega=om, delay_med=float(np.median(dl)), delay_mean=float(np.mean(dl)),
                    miss=float(np.mean(dl >= 1000)), fa=float(np.mean([r["fa_rate"] for r in rr])),
                    net=float(np.mean([r["net"] for r in rr])), moves=float(np.mean([r["moves"] for r in rr])), n=len(rr)))
    print(occ[-1])
json.dump(dict(main=S, tests=tests, sweep=dict(lifes=lifes, kaps=kaps, adv=adv.tolist(), se=se.tolist(), moves=mv.tolist()),
               occam=occ), open(RESDIR + "thermal_summary.json", "w"), indent=1)

fig, ax = plt.subplots(1, 4, figsize=(7.6, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.3, 1.1]))
AG = ["PA", "BE", "BEO"]
for i, (m, lab) in enumerate([("net", "(a) net cost"), ("gap", "(b) weather-induced gap")]):
    mus, cis = [], []
    for a_ in AG:
        d = np.array([(next(x[m] for x in main if x["kind"] == a_ and x["seed"] == s_) -
                       next(x[m] for x in main if x["kind"] == "FB" and x["seed"] == s_)) /
                      next(x[m] for x in main if x["kind"] == "FB" and x["seed"] == s_) for s_ in range(16)
                      if next(x[m] for x in main if x["kind"] == "FB" and x["seed"] == s_) > 0])
        mus.append(100 * d.mean()); cis.append(100 * 1.96 * d.std(ddof=1) / np.sqrt(len(d)))
    print("paired", m, dict(zip(AG, np.round(mus, 1))), dict(zip(AG, np.round(cis, 1))))
    ax[i].bar(range(3), mus, yerr=cis, color=[COL[a_] for a_ in AG], capsize=2)
    ax[i].axhline(0, color="k", lw=0.6)
    ax[i].set_xticks(range(3)); ax[i].set_xticklabels(AG, fontsize=7.5); ax[i].set_title(lab, fontsize=8.5)
    ax[i].set_ylabel("% change (paired)")
im = ax[2].imshow(adv * 100, cmap="RdBu", vmin=-np.abs(adv).max() * 100, vmax=np.abs(adv).max() * 100, origin="lower", aspect="auto")
ax[2].set_xticks(range(len(lifes))); ax[2].set_xticklabels(lifes, fontsize=7); ax[2].set_yticks(range(len(kaps))); ax[2].set_yticklabels(kaps)
ax[2].set_xlabel("mean source lifetime"); ax[2].set_ylabel(r"move cost $\kappa$")
for i in range(len(kaps)):
    for j in range(len(lifes)):
        ax[2].text(j, i, f"{adv[i, j] * 100:.0f}%", ha="center", va="center", fontsize=6.5)
ax[2].set_title("(c) BE vs FB: net-cost cut", fontsize=8.5)
lo_ = np.log10(oms)
ax[3].plot(lo_, [o["delay_mean"] for o in occ], "o-", color="#2166ac", ms=3, label="delay")
ax3b = ax[3].twinx(); ax3b.plot(lo_, [o["fa"] for o in occ], "s--", color="#999999", ms=3, label="false alarms")
ax[3].set_xlabel(r"$\log_{10}|\Omega|$"); ax[3].set_ylabel("mean delay [steps]"); ax3b.set_ylabel("false-alarm rate", color="#666666")
ax[3].set_title("(d) Occam cost in situ", fontsize=8.5)
h1, l1 = ax[3].get_legend_handles_labels(); h2, l2 = ax3b.get_legend_handles_labels(); ax[3].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.5, loc="upper center")
fig.tight_layout(); fig.savefig(FIGDIR + "fig_thermal.pdf")
print("saved")

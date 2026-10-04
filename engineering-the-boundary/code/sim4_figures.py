from paths import FIGDIR, RESDIR
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim4_foraging_agents import *
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
COL = dict(FB="#b2182b", PA="#ef8a62", RND="#999999", BE="#2166ac", OR="#1a9850")
NAMES = dict(FB="Fixed blanket (FB)", PA="Fixed blanket + adaptive precision (PA)", BE="Blanket engineering (BE)", OR="Oracle (OR)")
J = json.load(open(RESDIR + "sim4_summary.json"))
# phase
Ls = J["Ls"]; kaps = J["kaps"]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3))
for a_, key, ttl in [(ax[0], "advPA", "(a) BE $-$ PA"), (ax[1], "advFB", "(b) BE $-$ FB")]:
    A_ = np.array(J[key]); lim = np.abs(A_).max()
    im = a_.imshow(A_, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
    a_.set_xticks(range(len(Ls))); a_.set_xticklabels(["∞" if L == "inf" else str(int(float(L))) for L in Ls])
    a_.set_yticks(range(len(kaps))); a_.set_yticklabels([int(k) for k in kaps])
    a_.set_xlabel(r"mean regime duration $L$ (steps)"); a_.set_ylabel(r"re-wiring cost $\kappa$")
    a_.set_title(ttl + " (net payoff per step)", fontsize=9)
    for i in range(len(kaps)):
        for j in range(len(Ls)):
            v = A_[i, j]
            a_.text(j, i, f"{v:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 * lim else "black")
    plt.colorbar(im, ax=a_, fraction=0.046, pad=0.03)
fig.tight_layout(); fig.savefig(FIGDIR + "fig_phase.pdf")
# bars
S_ = J["summary"]; AG = ["FB", "PA", "RND", "BE", "OR"]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for i, (m, lab) in enumerate([("net", "(a) net payoff per step"),
                              ("surprisal", r"(b) inferential surprisal [nats]"),
                              ("viab", "(c) time in viability set")]):
    mu = [S_[a][m][0] for a in AG]; ci = [S_[a][m][1] for a in AG]
    ax[i].bar(range(len(AG)), mu, yerr=ci, color=[COL[a] for a in AG], capsize=2)
    ax[i].set_xticks(range(len(AG))); ax[i].set_xticklabels(AG, fontsize=8)
    ax[i].set_title(lab, fontsize=9)
fig.tight_layout(); fig.savefig(FIGDIR + "fig_bars.pdf")
# trace
_, tr = run_condition(200, 1.0, [0], 1500, keep_traces=True)
T = len(tr["reg"]); t = np.arange(T)
fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1.2, 1.3]))
for m in range(M):
    inf_mask = np.array([m in INF[r] for r in tr["reg"]])
    ax[0].fill_between(t, m - 0.45, m + 0.45, where=inf_mask, color="#d0d0d0", step="mid", lw=0)
    inS = (tr["BE"]["S"] == m).any(1)
    ax[0].scatter(t[inS], np.full(inS.sum(), m), s=0.6, color=COL["BE"], marker="s")
ax[0].set_yticks(range(M)); ax[0].set_ylabel("channel")
ax[0].set_title("(a) Grey: informative channels; blue: BE agent's sensory blanket", fontsize=9, loc="left")
ax[1].imshow(tr["BE"]["w"].T, aspect="auto", cmap="Blues", vmin=0, vmax=1, extent=[0, T, M - 0.5, -0.5], interpolation="nearest")
ax[1].set_yticks(range(M)); ax[1].set_ylabel("channel")
ax[1].set_title(r"(b) BE agent's belief $q(\iota_m{=}\mathrm{inf})$: perception of blanket membership", fontsize=9, loc="left")
for a in ["FB", "PA", "BE", "OR"]:
    c = np.convolve(tr[a]["reward"], np.ones(50) / 50, mode="valid")
    ax[2].plot(t[49:], c, color=COL[a], lw=1.3, label=NAMES[a])
ax[2].axhline(0.5, color="k", ls=":", lw=0.8)
ax[2].set_ylabel("reward rate"); ax[2].set_xlabel("time step"); ax[2].set_ylim(0, 1.05)
ax[2].legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))
ax[2].set_title("(c) 50-step reward rate; dotted: viability threshold", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(FIGDIR + "fig_trace.pdf")
print("ok")

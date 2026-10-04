from paths import FIGDIR, RESDIR
import json, sys
import numpy as np
from multiprocessing import Pool
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim4_foraging_agents import *

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
COL = dict(FB="#b2182b", PA="#ef8a62", RND="#999999", BE="#2166ac", OR="#1a9850")
NAMES = dict(FB="Fixed blanket", PA="Fixed blanket +\nadaptive precision",
             RND="Random\nre-wiring", BE="Blanket\nengineering", OR="Oracle")
AG = ["FB", "PA", "RND", "BE", "OR"]

# ------------------------------------------------------------ main experiment
L_MAIN, K_MAIN, T_MAIN, NSEED = 200, 1.0, 4000, 50

def one_seed(s):
    out, _ = run_condition(L_MAIN, K_MAIN, [s], T_MAIN)
    return {a: out[a][0] for a in AG}

def phase_cell(args):
    L, kap, s = args
    out, _ = run_condition(L, kap, [s], 3000, agents=("FB", "PA", "BE"))
    return (L, kap, s, out["FB"][0]["net"], out["PA"][0]["net"], out["BE"][0]["net"],
            out["BE"][0]["surprisal"], out["PA"][0]["surprisal"])

if __name__ == "__main__":
    with Pool(2) as pool:
        main = pool.map(one_seed, range(NSEED))
    metrics = ["net", "reward", "surprisal", "viab", "adds", "correctS"]
    summ = {}
    for a in AG:
        summ[a] = {}
        for m in metrics:
            v = np.array([r[a][m] for r in main], float)
            ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            summ[a][m] = (float(v.mean()), float(ci))
    tests = {}
    for b in ["FB", "PA", "RND"]:
        for m in ["net", "surprisal", "viab"]:
            d = np.array([r["BE"][m] - r[b][m] for r in main])
            w = stats.wilcoxon(d) if np.any(d != 0) else None
            tests[f"BE-{b}:{m}"] = dict(mean=float(d.mean()),
                                        ci=float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                                        p=float(w.pvalue) if w else 1.0,
                                        frac_pos=float((d > 0).mean()))
    for a in AG:
        print(a, {m: tuple(round(x, 3) for x in summ[a][m]) for m in metrics})
    for k, v in tests.items():
        print(k, v)

    # ---------------------------------------------------- example trace figure
    _, tr = run_condition(L_MAIN, K_MAIN, [0], 1500, keep_traces=True)
    T = len(tr["reg"]); t = np.arange(T)
    fig, ax = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.2, 1.2, 1.3]))
    # (a) informative channels (truth) vs BE blanket
    for m in range(M):
        inf_mask = np.array([m in INF[r] for r in tr["reg"]])
        ax[0].fill_between(t, m - 0.45, m + 0.45, where=inf_mask, color="#d9d9d9", step="mid", lw=0)
        inS = (tr["BE"]["S"] == m).any(1)
        ax[0].scatter(t[inS], np.full(inS.sum(), m), s=0.6, color=COL["BE"], marker="s")
    ax[0].set_yticks(range(M)); ax[0].set_ylabel("channel")
    ax[0].set_title("(a) Grey: channels carrying information about the hidden cause (true blanket); "
                    "blue: BE agent's sensory blanket", fontsize=8.5, loc="left")
    im = ax[1].imshow(tr["BE"]["w"].T, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      extent=[0, T, M - 0.5, -0.5], interpolation="nearest")
    ax[1].set_yticks(range(M)); ax[1].set_ylabel("channel")
    ax[1].set_title(r"(b) BE agent's posterior $q(\iota_m=\mathrm{informative})$ "
                    "(perception of blanket membership)", fontsize=8.5, loc="left")
    for a in ["FB", "PA", "BE", "OR"]:
        c = np.convolve(tr[a]["reward"], np.ones(50) / 50, mode="same")
        ax[2].plot(t, c, color=COL[a], lw=1.3, label=NAMES[a].replace("\n", " "))
    ax[2].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[2].set_ylabel("reward rate (50-step)"); ax[2].set_xlabel("time step")
    ax[2].legend(frameon=False, fontsize=7, ncol=4, loc="lower center")
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title("(c) Pragmatic performance; dotted line: viability threshold", fontsize=8.5, loc="left")
    fig.tight_layout(); fig.savefig(FIGDIR + "fig_trace.pdf")

    # ------------------------------------------------------- summary bar figure
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for i, (m, lab) in enumerate([("net", "net payoff per step"),
                                  ("surprisal", r"inferential surprisal $-\ln q(\eta_t)$ [nats]"),
                                  ("viab", "time in viability set")]):
        mu = [summ[a][m][0] for a in AG]; ci = [summ[a][m][1] for a in AG]
        ax[i].bar(range(len(AG)), mu, yerr=ci, color=[COL[a] for a in AG], capsize=2)
        ax[i].set_xticks(range(len(AG))); ax[i].set_xticklabels(AG, fontsize=8)
        ax[i].set_title(lab, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIGDIR + "fig_bars.pdf")

    # ------------------------------------------------------------ phase diagram
    Ls = [25, 50, 100, 200, 400, 1000, np.inf]
    kaps = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]
    NS = 12
    jobs = [(L, k, s) for L in Ls for k in kaps for s in range(NS)]
    with Pool(2) as pool:
        cells = pool.map(phase_cell, jobs)
    advPA = np.zeros((len(kaps), len(Ls))); advFB = np.zeros_like(advPA)
    sePA = np.zeros_like(advPA)
    for i, k in enumerate(kaps):
        for j, L in enumerate(Ls):
            rows = [c for c in cells if c[0] == L and c[1] == k]
            d = np.array([c[5] - c[4] for c in rows])
            advPA[i, j] = d.mean(); sePA[i, j] = d.std(ddof=1) / np.sqrt(len(d))
            advFB[i, j] = np.mean([c[5] - c[3] for c in rows])
    print("adv vs PA\n", np.round(advPA, 3)); print("se\n", np.round(sePA, 3))
    print("adv vs FB\n", np.round(advFB, 3))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for a_, A_, ttl in [(ax[0], advPA, "(a) BE minus fixed blanket + adaptive precision"),
                        (ax[1], advFB, "(b) BE minus fixed blanket")]:
        lim = np.abs(A_).max()
        im = a_.imshow(A_, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        a_.set_xticks(range(len(Ls))); a_.set_xticklabels([("∞" if not np.isfinite(L) else int(L)) for L in Ls])
        a_.set_yticks(range(len(kaps))); a_.set_yticklabels([int(k) for k in kaps])
        a_.set_xlabel(r"mean regime duration $L$ (steps)"); a_.set_ylabel(r"re-wiring cost $\kappa$")
        a_.set_title(ttl, fontsize=8.5)
        for i in range(len(kaps)):
            for j in range(len(Ls)):
                a_.text(j, i, f"{A_[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
        plt.colorbar(im, ax=a_, fraction=0.046, pad=0.03, label="Δ net payoff / step")
    fig.tight_layout(); fig.savefig(FIGDIR + "fig_phase.pdf")

    json.dump(dict(summary=summ, tests=tests, Ls=[str(L) for L in Ls], kaps=kaps,
                   advPA=advPA.tolist(), sePA=sePA.tolist(), advFB=advFB.tolist(),
                   params=dict(K=K, M=M, ALPHA=ALPHA, STAY=STAY, GAMMA=GAMMA, HAZ=HAZ,
                               W0=W0, BETA=BETA, HSTRUCT=HSTRUCT, L_MAIN=L_MAIN,
                               K_MAIN=K_MAIN, T_MAIN=T_MAIN, NSEED=NSEED, NS_phase=NS)),
              open(RESDIR + "sim4_summary.json", "w"), indent=1)
    print("done")

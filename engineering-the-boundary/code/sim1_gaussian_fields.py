"""
Simulation 1: the factorisation gap (blanket-dependent free-energy floor) in
Gaussian Markov random fields, for a fixed blanket vs. an engineered blanket.

For a Gaussian joint law with precision L over (Y, B, X), the conditional law of
(Y, X) given B has precision L_{(YX),(YX)} (independent of b). The inclusive gap
I(Y;X|B) and the exclusive (mean-field) gap min_{qY qX} KL(qY qX || p(.,.|b))
are computed and checked to coincide (Lemma 1 in the paper).
"""
from paths import FIGDIR, RESDIR
import itertools, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

def gaps(L, Y, B, X):
    """Return (inclusive CMI, exclusive mean-field KL) in nats."""
    idx = list(Y) + list(X)
    P = L[np.ix_(idx, idx)]                  # conditional precision of (Y,X)|B
    S = np.linalg.inv(P)                     # conditional covariance
    nY = len(Y)
    SY, SX = S[:nY, :nY], S[nY:, nY:]
    PY, PX = P[:nY, :nY], P[nY:, nY:]
    _, ld_S = np.linalg.slogdet(S)
    _, ld_SY = np.linalg.slogdet(SY)
    _, ld_SX = np.linalg.slogdet(SX)
    incl = 0.5 * (ld_SY + ld_SX - ld_S)
    _, ld_P = np.linalg.slogdet(P)
    _, ld_PY = np.linalg.slogdet(PY)
    _, ld_PX = np.linalg.slogdet(PX)
    excl = 0.5 * (ld_PY + ld_PX - ld_P)
    return incl, excl

# ---------------------------------------------------------------- chain + leak
n = 13
rho = 0.45
X = [5, 6, 7]
B0 = [4, 8]
def chain_precision(eps):
    L = np.eye(n)
    for i in range(n - 1):
        L[i, i + 1] = L[i + 1, i] = -rho
    L[1, 6] = L[6, 1] = -eps          # long-range "leak": exterior node 1 -> interior node 6
    L += np.eye(n) * 0.25             # keep PD
    assert np.all(np.linalg.eigvalsh(L) > 0)
    return L

def best_blanket(L, X, B0, budget=2, lam=0.02):
    """Inverted detection: search over blankets B0 + up to `budget` recruited
    exterior nodes, minimising gap + lam*|B| (lam = cost of each boundary element)."""
    ext = [i for i in range(L.shape[0]) if i not in X and i not in B0]
    best = None
    for k in range(budget + 1):
        for add in itertools.combinations(ext, k):
            B = list(B0) + list(add)
            Y = [i for i in ext if i not in add]
            g, _ = gaps(L, Y, B, X)
            obj = g + lam * len(B)
            if best is None or obj < best[0]:
                best = (obj, B, g)
    return best

eps_grid = np.linspace(0, 0.4, 41)
fixed, eng, chosen, check = [], [], [], []
for e in eps_grid:
    L = chain_precision(e)
    Y0 = [i for i in range(n) if i not in X + B0]
    gi, ge = gaps(L, Y0, B0, X)
    check.append(abs(gi - ge))
    fixed.append(gi)
    obj, B, g = best_blanket(L, X, B0)
    eng.append(g)
    chosen.append(B)

print("max |inclusive - exclusive| on chain:", max(check))
print("engineered blanket at eps=0.3:", chosen[30])

# ------------------------------------------------- random sparse GMRF ensemble
def random_gmrf(n=30, p=0.08, w=0.3):
    A = (rng.random((n, n)) < p).astype(float)
    A = np.triu(A, 1); A = A + A.T
    W = A * rng.uniform(0.1, w, size=(n, n)); W = np.triu(W, 1); W = W + W.T
    L = -W
    np.fill_diagonal(L, np.abs(W).sum(1) + 0.3)   # diagonally dominant => PD
    return L, A

def neighbours(A, S):
    return sorted({j for i in S for j in np.nonzero(A[i])[0]} - set(S))

res = []
for trial in range(400):
    L, A = random_gmrf()
    nn = L.shape[0]
    X = list(rng.choice(nn, 4, replace=False))
    B0 = neighbours(A, X)
    if len(B0) == 0 or len(B0) > 12:
        continue
    Y0 = [i for i in range(nn) if i not in X + B0]
    g0, _ = gaps(L, Y0, B0, X)        # exact blanket initially: ~0
    # environmental change: m new random long-range couplings
    L2, A2 = L.copy(), A.copy()
    m = 3
    added = 0
    while added < m:
        i, j = rng.choice(nn, 2, replace=False)
        if A2[i, j] == 0:
            wv = rng.uniform(0.1, 0.3)
            L2[i, j] = L2[j, i] = -wv
            L2[i, i] += wv; L2[j, j] += wv
            A2[i, j] = A2[j, i] = 1
            added += 1
    gfix, gexcl = gaps(L2, Y0, B0, X)
    # engineering agent: re-detect blanket (neighbourhood of X in new coupling graph)
    B1 = neighbours(A2, X)
    Y1 = [i for i in range(nn) if i not in X + B1]
    geng, _ = gaps(L2, Y1, B1, X) if len(Y1) > 0 else (0.0, 0.0)
    res.append(dict(g0=g0, gfix=gfix, geng=geng, nB0=len(B0), nB1=len(B1),
                    excl_err=abs(gfix - gexcl)))

gfix = np.array([r["gfix"] for r in res]); geng = np.array([r["geng"] for r in res])
print("random GMRFs:", len(res), "trials")
print("initial gap max:", max(r["g0"] for r in res))
print("fixed-blanket gap after change: mean %.4f median %.4f frac>1e-6 %.3f" %
      (gfix.mean(), np.median(gfix), (gfix > 1e-6).mean()))
print("engineered gap: max %.2e" % geng.max())
print("blanket size before/after: %.2f -> %.2f" % (np.mean([r['nB0'] for r in res]),
                                                  np.mean([r['nB1'] for r in res])))
print("affected mean gap", gfix[gfix>1e-6].mean(), "n", (gfix>1e-6).sum())
print("max excl/incl discrepancy:", max(r["excl_err"] for r in res))

summary = dict(chain=dict(eps=eps_grid.tolist(), fixed=fixed, eng=eng,
                          chosen_at_0p3=chosen[30]),
               random=dict(n=len(res), gfix_mean=float(gfix.mean()),
                           gfix_median=float(np.median(gfix)),
                           frac_positive=float((gfix > 1e-6).mean()),
                           geng_max=float(geng.max()),
                           nB0=float(np.mean([r['nB0'] for r in res])),
                           nB1=float(np.mean([r['nB1'] for r in res]))))
json.dump(summary, open(RESDIR + "sim1_summary.json", "w"), indent=1)

# ----------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(eps_grid, fixed, color="#b2182b", lw=2, label=r"fixed blanket $\omega_0=\{4,8\}$")
ax[0].plot(eps_grid, eng, color="#2166ac", lw=2, ls="--", label="engineered blanket")
ax[0].set_xlabel(r"leak strength $\varepsilon$ (exterior node 1 $\leftrightarrow$ interior node 6)")
ax[0].set_ylabel(r"factorisation gap $\Delta(\omega)$ [nats $=k_BT$ units]")
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("(a) Chain with a long-range leak", fontsize=9)
aff = gfix[gfix > 1e-6]
bins = np.linspace(0, aff.max(), 30)
ax[1].hist(aff, bins=bins, color="#b2182b", alpha=0.8, label="fixed blanket")
ax[1].axvline(geng.max(), color="#2166ac", ls="--", lw=2, label="engineered (all trials)")
ax[1].set_xlabel(r"$\Delta(\omega)$ after environmental change [nats]")
ax[1].set_ylabel("number of affected systems")
ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("(b) Random GMRFs: %d/%d blankets breached" % (len(aff), len(res)), fontsize=9)
fig.tight_layout()
fig.savefig(FIGDIR + "fig_sim1.pdf")
print("saved")

"""
Simulation 2: emergence of blanket engineering from free-energy minimisation on slow variables.

(a) Gaussian, full control: anti-Hebbian gradient flow  dW/dt = eta * Sigma_{YX|B}
    on the interior-exterior conditional precision block W = P_YX. Check: monotone
    decrease of the gap, positive-definiteness preserved, exponential bound
    V(t) <= V(0) exp(-2 eta t / (lmax(P_XX) lmax(P_YY))),  V = tr(W^T P_YY W).
(b) Gaussian, partial control: gap plateaus at Delta_res > 0; structural perception
    (Bayesian model comparison over partitions on samples) relocates the boundary;
    sample size needed scales like [ln(1/alpha) + ln|Omega|] / Delta_res.
(c) Non-Gaussian check: binary pairwise (Ising) model, gradient descent on the
    exclusive (mean-field) gap and on the inclusive CMI w.r.t. cross couplings.
"""
from paths import FIGDIR, RESDIR
import itertools, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
rng = np.random.default_rng(3)
OUT = {}

def gap_from_P(P, nY):
    S = np.linalg.inv(P)
    return 0.5 * (np.linalg.slogdet(P[:nY, :nY])[1] + np.linalg.slogdet(P[nY:, nY:])[1]
                  - np.linalg.slogdet(P)[1])

def make_P(nY, nX, scale=0.35):
    A = rng.normal(size=(nY, nY)); PY = A @ A.T / nY + np.eye(nY)
    A = rng.normal(size=(nX, nX)); PX = A @ A.T / nX + np.eye(nX)
    while True:
        W = rng.normal(scale=scale, size=(nY, nX))
        P = np.block([[PY, W], [W.T, PX]])
        if np.all(np.linalg.eigvalsh(P) > 1e-3):
            return PY, PX, W

def flow(PY, PX, W0, mask, eta=1.0, dt=0.01, T=12.0):
    nY = PY.shape[0]
    W = W0.copy(); ts, gaps, Vs, mineig = [], [], [], []
    for k in range(int(T / dt) + 1):
        P = np.block([[PY, W], [W.T, PX]])
        S = np.linalg.inv(P)
        ts.append(k * dt); gaps.append(gap_from_P(P, nY))
        Vs.append(np.trace(W.T @ PY @ W)); mineig.append(np.linalg.eigvalsh(P).min())
        W = W + dt * eta * S[:nY, nY:] * mask        # anti-Hebbian: dW = eta Sigma_{YX|B}
    return np.array(ts), np.array(gaps), np.array(Vs), np.array(mineig), W

# ---------------- (a) full control, several random systems
nY, nX = 6, 4
curves = []
for rep in range(8):
    PY, PX, W0 = make_P(nY, nX)
    ts, g, V, me, W = flow(PY, PX, W0, np.ones_like(W0), T=40.0, dt=0.02)
    rate = 2.0 / (np.linalg.eigvalsh(PX).max() * np.linalg.eigvalsh(PY).max())
    bound_ok = np.all(V <= V[0] * np.exp(-rate * ts) * (1 + 1e-6) + 1e-14)
    mono = np.all(np.diff(g) <= 1e-12)
    curves.append(dict(ts=ts, g=g, V=V, rate=rate, bound_ok=bool(bound_ok), mono=bool(mono),
                       pd=bool(me.min() > 0), final_gap=float(g[-1]), g0=float(g[0])))
OUT["full_control"] = [{k: v for k, v in c.items() if k not in ("ts", "g", "V")} for c in curves]
print("full control:", OUT["full_control"])

# ---------------- (b) partial control, then relabelling, then plasticity again
PY, PX, W0 = make_P(nY, nX)
nB = 3
iY = np.arange(nY); iB = np.arange(nY, nY + nB); iX = np.arange(nY + nB, nY + nB + nX)
Lam = np.zeros((nY + nB + nX,) * 2)
Lam[np.ix_(iY, iY)] = PY; Lam[np.ix_(iX, iX)] = PX
W0[0, 1] = 0.6
Lam[np.ix_(iY, iX)] = W0; Lam[np.ix_(iX, iY)] = W0.T
Kyb = rng.normal(scale=0.3, size=(nY, nB)); Kxb = rng.normal(scale=0.3, size=(nX, nB))
Lam[np.ix_(iY, iB)] = Kyb; Lam[np.ix_(iB, iY)] = Kyb.T
Lam[np.ix_(iX, iB)] = Kxb; Lam[np.ix_(iB, iX)] = Kxb.T
Lam[np.ix_(iB, iB)] = np.eye(nB) * 3.0
while np.linalg.eigvalsh(Lam).min() <= 0.05:
    Lam[np.ix_(iY, iY)] += 0.2 * np.eye(nY); Lam[np.ix_(iX, iX)] += 0.2 * np.eye(nX)

def gap_part(L, Y, X):
    idx = list(Y) + list(X); P = L[np.ix_(idx, idx)]
    return gap_from_P(P, len(Y))

def flow_full(L, Y, X, uncontrollable, T, dt=0.02, eta=1.0):
    L = L.copy(); g = []
    for k in range(int(T / dt)):
        idx = list(Y) + list(X); P = L[np.ix_(idx, idx)]
        g.append(gap_from_P(P, len(Y)))
        S = np.linalg.inv(P)[:len(Y), len(Y):]
        for a_, y in enumerate(Y):
            for b_, x in enumerate(X):
                if (y, x) in uncontrollable: continue
                L[y, x] += dt * eta * S[a_, b_]; L[x, y] = L[y, x]
    return L, np.array(g)

Y0 = list(iY); X0 = list(iX)
unc = {(int(iY[0]), int(iX[1]))}
L1, g1 = flow_full(Lam, Y0, X0, unc, T=40.0)
Dres = float(g1[-1])
Y1 = [y for y in Y0 if y != iY[0]]           # relabel: recruit y0 into the boundary
gap_after_relabel = gap_part(L1, Y1, X0)
L2, g2 = flow_full(L1, Y1, X0, set(), T=40.0)
ts_p = np.arange(len(g1) + len(g2)) * 0.02; g_p = np.concatenate([g1, g2])
print("partial: g0 %.4f  plateau %.4f  after relabel %.4f  final %.2e" % (g1[0], Dres, gap_after_relabel, g2[-1]))
OUT["partial_control"] = dict(g0=float(g1[0]), residual_gap=Dres, after_relabel=float(gap_after_relabel), final=float(g2[-1]))
Sig = np.linalg.inv(L1)
def cmi_cov(C, Y, B, X):
    def ld(ix): return np.linalg.slogdet(C[np.ix_(ix, ix)])[1] if len(ix) else 0.0
    return 0.5 * (ld(Y + B) + ld(X + B) - ld(B) - ld(Y + B + X))
Yl, Bl, Xl = list(iY), list(iB), list(iX)
omega0 = (Yl, Bl, Xl)
omega1 = ([i for i in Yl if i != 0], Bl + [0], Xl)         # recruit leaking node y0
I0 = cmi_cov(Sig, *omega0); I1 = cmi_cov(Sig, *omega1)
print("true CMI omega0 %.4f omega1 %.4g" % (I0, I1))
OUT["relabel"] = dict(I_omega0=float(I0), I_omega1=float(I1))

# detection by Bayesian model comparison over partitions (BIC), candidate set = recruit one
# exterior node (|Omega_f| = nY) plus a broad set (any single non-interior element relabelled
# in any of the 3 roles, padded by k irrelevant alternatives) -> study n* vs ln|Omega|
def n_params_blanket(nY_, nB_, nX_):
    # free parameters of Gaussian with Y-X precision block forced to zero
    d = nY_ + nB_ + nX_
    return d * (d + 1) / 2 - nY_ * nX_

Lchol = np.linalg.cholesky(Sig)
def detect_n(prior_logodds, alpha=0.05, reps=150, nmax=6000):
    """smallest n at which the posterior favours omega1 over omega0 by ln(1/alpha)."""
    out = []
    for r in range(reps):
        Z = (Lchol @ rng.normal(size=(Lchol.shape[0], nmax))).T
        hit = nmax
        for n in range(20, nmax + 1, 5):
            C = np.cov(Z[:n].T, bias=True)
            ll = n * (cmi_cov(C, *omega0) - cmi_cov(C, *omega1))   # log-lik advantage of omega1
            k0 = n_params_blanket(len(omega0[0]), len(omega0[1]), len(omega0[2]))
            k1 = n_params_blanket(len(omega1[0]), len(omega1[1]), len(omega1[2]))
            logpost = ll - 0.5 * (k1 - k0) * np.log(n) + prior_logodds
            if logpost > np.log(1 / alpha):
                hit = n; break
        out.append(hit)
    return np.array(out)

Omegas = [2, 8, 32, 128, 512, 2048, 8192]
nstar = []
for Om in Omegas:
    d = detect_n(prior_logodds=-np.log(Om))        # omega1 is one of |Omega| equiprobable alternatives
    nstar.append((float(np.median(d)), float(np.percentile(d, 25)), float(np.percentile(d, 75))))
    print("|Omega|=%d  n* median %.0f" % (Om, nstar[-1][0]))
OUT["occam_detection"] = dict(Omegas=Omegas, nstar=nstar)
lnO = np.log(Omegas); med = np.array([x[0] for x in nstar])
slope = np.polyfit(lnO, med, 1)
print("slope dn*/dln|Omega| = %.1f  ; theory 1/(I0-I1) = %.1f" % (slope[0], 1 / (I0 - I1)))
OUT["occam_detection"]["slope"] = float(slope[0]); OUT["occam_detection"]["inv_gap"] = float(1 / (I0 - I1))

# ---------------- (c) Ising check
def ising_probs(J, h):
    n = J.shape[0]
    S = np.array(list(itertools.product([-1, 1], repeat=n)))
    E = 0.5 * np.einsum("si,ij,sj->s", S, J, S) + S @ h
    p = np.exp(E - E.max()); return S, p / p.sum()

nYi, nBi, nXi = 2, 2, 2
idY, idB, idX = [0, 1], [2, 3], [4, 5]
def cmi_ising(J, h):
    S, p = ising_probs(J, h)
    key = lambda ix: [tuple(s) for s in S[:, ix]]
    def H(ix):
        d = {}
        for k, pv in zip(key(ix), p): d[k] = d.get(k, 0) + pv
        v = np.array(list(d.values())); return -(v * np.log(v)).sum()
    return H(idY + idB) + H(idX + idB) - H(idB) - H(idY + idB + idX)

def mf_gap_ising(J, h, iters=200):
    """exclusive gap averaged over b: E_b min_{qY qX} KL(qY qX || p(y,x|b))."""
    S, p = ising_probs(J, h)
    tot = 0.0
    for b in itertools.product([-1, 1], repeat=2):
        sel = np.all(S[:, idB] == np.array(b), axis=1)
        pb = p[sel].sum(); cond = p[sel] / pb
        Ys = [tuple(s) for s in S[sel][:, idY]]; Xs = [tuple(s) for s in S[sel][:, idX]]
        uY = sorted(set(Ys)); uX = sorted(set(Xs))
        Pm = np.zeros((len(uY), len(uX)))
        for y, x, c in zip(Ys, Xs, cond): Pm[uY.index(y), uX.index(x)] = c
        L = np.log(Pm)
        qY = Pm.sum(1); qX = Pm.sum(0)
        for _ in range(iters):
            qY = np.exp(L @ qX); qY /= qY.sum()
            qX = np.exp(qY @ L); qX /= qX.sum()
        q = np.outer(qY, qX)
        tot += pb * (q * (np.log(q) - L)).sum()
    return tot

def descend(fn, J, iters, lr=0.5):
    Jc = J.copy(); traj = []; Jn = []
    for it in range(iters):
        g0 = fn(Jc, hv); traj.append(g0)
        Jn.append(np.sqrt(sum(Jc[i, j] ** 2 for i, j in cross)))
        grad = {}
        for i, j in cross:
            Jp = Jc.copy(); Jp[i, j] = Jp[j, i] = Jc[i, j] + 1e-5
            grad[(i, j)] = (fn(Jp, hv) - g0) / 1e-5
        for (i, j), gr in grad.items():
            Jc[i, j] = Jc[j, i] = Jc[i, j] - lr * gr
    return np.array(traj), np.array(Jn)
cross = [(i, j) for i in idY for j in idX]
ising_res = {"inclusive": [], "exclusive": []}
ising_traj = {}
for sysk in range(10):
    J = rng.normal(scale=0.4, size=(6, 6)); J = (J + J.T) / 2; np.fill_diagonal(J, 0)
    hv = rng.normal(scale=0.2, size=6)
    for i, j in cross: J[i, j] = J[j, i] = rng.choice([-1, 1]) * rng.uniform(0.5, 0.9)
    for name, fn, it in [("inclusive", cmi_ising, 300), ("exclusive", mf_gap_ising, 300)]:
        tr, Jn = descend(fn, J, it)
        ising_res[name].append(dict(g0=float(tr[0]), gT=float(tr[-1]), J0=float(Jn[0]), JT=float(Jn[-1]),
                                    mono=bool(np.all(np.diff(tr) <= 1e-9)),
                                    incl_at_end=None))
        ising_traj.setdefault(name, []).append((tr, Jn))
for name in ising_res:
    r = ising_res[name]
    print("Ising", name, "decoupled (|J_YX|<0.05):", sum(x["JT"] < 0.05 for x in r), "/", len(r),
          " monotone:", all(x["mono"] for x in r), " final gaps:", [round(x["gT"], 4) for x in r],
          " final |J|:", [round(x["JT"], 2) for x in r])
OUT["ising"] = ising_res
json.dump(OUT, open(RESDIR + "emergence_summary.json", "w"), indent=1)

# ---------------- figure
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.4))
for c in curves:
    ax[0].semilogy(c["ts"], c["V"] / c["V"][0], color="#2166ac", lw=1, alpha=0.8)
    ax[0].semilogy(c["ts"], np.exp(-c["rate"] * c["ts"]), color="#999999", lw=0.6, ls=":")
ax[0].set_xlabel("time $t$ (units of $1/\\eta$)"); ax[0].set_ylabel(r"$V(t)/V(0)$; dotted: bound")
ax[0].set_title("(a) full control", fontsize=8.5); ax[0].set_ylim(1e-10, 1.5)
ax[1].semilogy(ts_p, g_p + 1e-12, color="#b2182b", lw=1.5)
ax[1].axvline(40.0, color="k", ls=":", lw=0.8)
ax[1].set_xlabel("time $t$"); ax[1].set_ylabel(r"gap $\Delta$ [nats]"); ax[1].set_title("(b) partial control + relabel", fontsize=8.5)
ax[1].text(41, Dres * 1.5, "relabel", fontsize=7); ax[1].set_ylim(1e-8, 1)
ax[2].errorbar(lnO, med, yerr=[med - np.array([x[1] for x in nstar]), np.array([x[2] for x in nstar]) - med],
               fmt="o", color="#2166ac", ms=3, capsize=2)
xx = np.linspace(lnO.min(), lnO.max(), 10)
def n_pred(lnOm, dI=I0 - I1, dk=4, alpha=0.05):
    n = 500.0
    for _ in range(50): n = (np.log(1 / alpha) + lnOm + 0.5 * dk * np.log(n)) / dI
    return n
ax[2].plot(xx, [n_pred(v) for v in xx], color="#b2182b", lw=1.0, ls="--", label="prediction")
ax[2].plot(xx, np.polyval(slope, xx), color="#2166ac", lw=0.8, label="fit")
ax[2].legend(frameon=False, fontsize=6.5)
OUT["occam_detection"]["pred"] = [float(n_pred(v)) for v in lnO]
json.dump(OUT, open(RESDIR + "emergence_summary.json", "w"), indent=1)
ax[2].set_xlabel(r"$\ln|\Omega|$ (breadth of structural prior)"); ax[2].set_ylabel(r"samples to relocate $n^\ast$")
ax[2].set_title("(c) Occam cost of relabelling", fontsize=8.5)
for name, col in [("inclusive", "#1a9850"), ("exclusive", "#762a83")]:
    for q_, (tr, Jn) in enumerate(ising_traj[name]):
        ax[3].semilogy(tr + 1e-12, color=col, lw=0.8, alpha=0.8, label=name if q_ == 0 else None)
ax[3].set_xlabel("iteration"); ax[3].set_ylabel("gap [nats]"); ax[3].set_title("(d) binary (Ising) systems", fontsize=8.5)
ax[3].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIGDIR + "fig_emergence.pdf")
print("saved")

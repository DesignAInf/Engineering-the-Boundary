"""
Simulation 5 (physical case): a thermal 'pavilion' on a 2-D heat-conduction lattice.

World: G x G lattice of temperatures, nearest-neighbour conduction, weak loss to ambient,
background thermal noise, and heat sources (sun patches / cold wind spots) that appear at
random positions outside the pavilion (Poisson arrivals, exponential lifetimes, random sign
and amplitude). Nothing in the world is arranged to favour any agent.

Pavilion (agent): interior I (central 4x4); ring R (20 cells around I). The agent owns
  k_p ports  : ring cells fitted with a thermometer (sensory state) and a heater (active state)
  m   shields: ring cells whose conductance to the outside is reduced (insulation)
Its behavioural level is active inference in the linear-Gaussian limit (Kalman filtering +
quadratic preferences = LQG; Baltieri & Buckley 2019; Millidge et al. 2020).

Partition used for the diagnostic gap: X = interior + unported ring cells, B = ported ring
cells, Y = outside. Gap = I(Y; X | B) under the stationary law of the open-loop physics with
the current sources (continuous Lyapunov equation).

Agents
  FB   best fixed design (chosen among symmetric layouts by long pilot runs)
  PA   FB layout + adaptive precision: infers where heat is leaking in and re-weights its
       filter's process-noise model (attention), but cannot move hardware
  BE   same perception + structural action: moves ports/shields towards inferred leaks when
       the structure-level EFE (benefit over anticipated tenure - move cost) favours it
  BEu  BE with an unfocused structural prior (hypotheses on all non-interior cells)
"""
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_lyapunov

G = 12
DT = 0.1
KCOND, AMB, SIG_BG = 1.0, 0.02, 0.05
C_INS = 0.08                      # residual conductance of an insulated edge
RU = 0.5
AMP = (4.0, 8.0)
SRC_CELLS = 'far'
N = G * G
def idx(r, c): return r * G + c
I_CELLS = [idx(r, c) for r in range(4, 8) for c in range(4, 8)]
RING = [idx(r, c) for r in range(3, 9) for c in range(3, 9)
        if not (4 <= r < 8 and 4 <= c < 8)]
OUT_CELLS = [i for i in range(N) if i not in I_CELLS and i not in RING]
FAR_CELLS = [idx(r, c) for r in range(G) for c in range(G)
             if max(abs(r - 5.5), abs(c - 5.5)) >= 3.5]          # >=1 cell away from ring
NEIGH = {i: [] for i in range(N)}
for r in range(G):
    for c in range(G):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < G and 0 <= cc < G:
                NEIGH[idx(r, c)].append(idx(rr, cc))
RING_SET, OUT_SET = set(RING), set(OUT_CELLS)

def conduction_matrix(shields):
    """continuous-time generator L (dT/dt = L T + inputs)."""
    L = np.zeros((N, N))
    sh = set(shields)
    for i in range(N):
        for j in NEIGH[i]:
            c = KCOND
            if (i in sh and j in OUT_SET) or (j in sh and i in OUT_SET):
                c *= C_INS
            L[i, j] += c; L[i, i] -= c
        L[i, i] -= AMB
    return L

# ------------------------------------------------------------------ world
def make_sources(T, rate, life, rng):
    """list of (t0, t1, cell, amplitude)."""
    srcs, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= T: break
        t1 = t + rng.exponential(life)
        pool = FAR_CELLS if SRC_CELLS == 'far' else OUT_CELLS
        cell = pool[rng.integers(len(pool))]
        amp = rng.choice([-1, 1]) * rng.uniform(*AMP)
        srcs.append((int(t), int(min(t1, T)), cell, amp))
    return srcs

def source_field(srcs, T):
    """sparse representation: for each t, list of (cell, amp) -> dense array (T, N) is too big;
    return per-step event lists."""
    on = [[] for _ in range(T)]
    for (t0, t1, c, a) in srcs:
        for t in range(t0, t1):
            on[t].append((c, a))
    return on

# ------------------------------------------------------------------ controller (LQG)
_cache = {}
_WARM = {}
def _sda(A, G, Hq, tol=1e-11, maxit=60):
    """structure-preserving doubling for X = Hq + A^T X (I + G X)^{-1} A."""
    Ak, Gk, Hk = A.copy(), G.copy(), Hq.copy()
    I = np.eye(A.shape[0])
    for it in range(maxit):
        W = I + Gk @ Hk
        WiA = np.linalg.solve(W, Ak)
        WiG = np.linalg.solve(W, Gk)
        Hn = Hk + Ak.T @ Hk @ WiA
        Gk = Gk + Ak @ WiG @ Ak.T
        Ak = Ak @ WiA
        if np.max(np.abs(Hn - Hk)) < tol * (1 + np.max(np.abs(Hn))):
            return Hn
        Hk = Hn
    return Hk
def _riccati_filter(A, H, Q, R, P0=None):
    return _sda(A.T, H.T @ np.linalg.solve(R, H), Q)
def _riccati_control(A, B, Qc, R, X0=None):
    return _sda(A, B @ np.linalg.solve(R, B.T), Qc)
def lqg(ports, shields, qnoise_out):
    key = (tuple(sorted(ports)), tuple(sorted(shields)), tuple(np.round(qnoise_out, 3)))
    if key in _cache: return _cache[key]
    if len(_cache) > 40: _cache.clear()
    L = conduction_matrix(shields)
    A = np.eye(N) + DT * L
    Bm = np.zeros((N, len(ports)))
    for k, p in enumerate(ports): Bm[p, k] = DT
    obs = list(I_CELLS) + list(ports)
    H = np.zeros((len(obs), N))
    for k, o in enumerate(obs): H[k, o] = 1.0
    Q = np.eye(N) * SIG_BG ** 2 * DT
    for k, o in enumerate(OUT_CELLS): Q[o, o] += qnoise_out[k] * DT
    Rn = np.eye(len(obs)) * 0.01 ** 2
    P = _riccati_filter(A, H, Q, Rn, _WARM.get("P"))
    S = H @ P @ H.T + Rn
    K = P @ H.T @ np.linalg.inv(S)
    Qc = np.zeros((N, N))
    for i in I_CELLS: Qc[i, i] = 1.0
    Ru = np.eye(len(ports)) * RU
    X = _riccati_control(A, Bm, Qc, Ru, _WARM.get("X"))
    _WARM["P"], _WARM["X"] = P, X
    F = np.linalg.solve(Ru + Bm.T @ X @ Bm, Bm.T @ X @ A)
    out = dict(A=A, B=Bm, H=H, K=K, F=F, S=S, Sinv=np.linalg.inv(S), obs=obs)
    _cache[key] = out
    return out

# ------------------------------------------------------------------ leak perception
def signatures(ctrl):
    """innovation signature of an unmodelled unit heat inflow at each candidate cell,
    two-step response through the predicted dynamics (inflow -> cell -> neighbours)."""
    if "sig" in ctrl: return ctrl["sig"]
    A, K, H = ctrl["A"], ctrl["K"], ctrl["H"]
    IKH = np.eye(N) - K @ H
    E = np.eye(N) * DT
    M = np.linalg.solve(np.eye(N) - IKH @ A, IKH @ E)      # steady-state estimation bias
    ctrl["sig"] = H @ (A @ M + E)                         # steady-state innovation mean
    return ctrl["sig"]

def _gap(L, ports, Qn):
    S = solve_continuous_lyapunov(L, -Qn)
    B = sorted(ports)
    X = sorted(set(I_CELLS) | (RING_SET - set(ports)))
    Y = OUT_CELLS
    def ld(ix): return np.linalg.slogdet(S[np.ix_(ix, ix)])[1]
    return 0.5 * (ld(Y + B) + ld(X + B) - ld(B) - ld(Y + B + X))

def gap_diagnostic(ports, shields, active_src, extra_var=4.0):
    """source-attributable factorisation gap: I(Y;X|B) with the current sources minus the
    same quantity under background noise only (the part of the breach caused by the weather)."""
    L = conduction_matrix(shields)
    Qn = np.eye(N) * SIG_BG ** 2
    base = _gap(L, ports, Qn)
    if not active_src: return 0.0
    for (c, a) in active_src: Qn[c, c] += extra_var * a ** 2 / 9.0
    return _gap(L, ports, Qn) - base

# ------------------------------------------------------------------ run one agent
PA_THR = 4.0
RHO_P, RHO_S = 0.75, 0.60     # calibrated screening factors (port / shield at the inflow cell)
ELL = 1.5
NEAR_RING = {}
_CN = {}
class _CandNear(dict):
    def __missing__(self, key):
        v = np.array([min(range(len(RING)), key=lambda r: _dist(c, RING[r])) for c in key])
        self[key] = v; return v
CAND_NEAR = _CandNear()

def run(kind, sources_on, T, layout, rng, kappa=2.0, H_str=400.0, beta_val=0.0045,
        window=50, forget=0.7, tau2=4.0, ring_only=False, gap_every=0, omega_size=None,
        max_leaks=4, decay=0.85, track_delay=False, srcs=None, oracle_perception=False, adapt_H=True):
    """kind in FB, PA, BE. Returns performance and diagnostics."""
    ports, shields = list(layout[0]), list(layout[1])
    kp, ms = len(ports), len(shields)
    qn = np.full(len(OUT_CELLS), 0.05)
    ctrl = lqg(ports, shields, qn)
    Ttrue = np.zeros(N); xhat = np.zeros(N)
    cost = 0.0; ctrl_e = 0.0; moves = 0; viab = 0
    cand = list(RING) if ring_only else (OUT_CELLS + RING)
    nOm = len(cand) if omega_size is None else omega_size
    logprior_odds = np.log(2.0 / nOm)          # ~2 leaks expected among |Omega| hypotheses
    near = CAND_NEAR[tuple(cand)]
    def refresh(ctrl):
        sig = signatures(ctrl)[:, cand]
        Ws = ctrl["Sinv"] @ sig                # whitened signatures
        cvec = np.einsum("oc,oc->c", sig, Ws)
        return sig, Ws, cvec
    sig, Ws, cvec = refresh(ctrl)
    esum = np.zeros(len(ctrl["obs"])); cnt = 0
    eacc = np.zeros_like(esum); nacc = 0.0
    pw_bar = np.zeros(len(RING)); effort = np.zeros(kp)
    run_len = np.zeros(len(RING)); ep_sum = H_str / window; ep_n = 1.0     # prior: one episode of length H_str
    gaps = []; detections = []
    Lphys = {}
    def phys(sh):
        k = tuple(sorted(sh))
        if k not in Lphys: Lphys[k] = np.eye(N) + DT * conduction_matrix(sh)
        return Lphys[k]
    Aphys = phys(shields)
    for t in range(T):
        u = -ctrl["F"] @ xhat
        ctrl_e += float(u @ u); effort += u ** 2
        inp = np.zeros(N)
        for (c, a) in sources_on[t]: inp[c] += a
        for k, p in enumerate(ports): inp[p] += u[k]
        Ttrue = Aphys @ Ttrue + DT * inp + SIG_BG * np.sqrt(DT) * rng.normal(size=N)
        y = Ttrue[ctrl["obs"]] + 0.01 * rng.normal(size=len(ctrl["obs"]))
        xpred = ctrl["A"] @ xhat + ctrl["B"] @ u
        e = y - ctrl["H"] @ xpred
        xhat = xpred + ctrl["K"] @ e
        dev = Ttrue[I_CELLS]
        cost += float(np.mean(dev ** 2)); viab += float(abs(dev.mean()) < 0.3)
        if kind in ("PA", "BE"):
            esum += ctrl["Sinv"] @ e; cnt += 1
            if cnt == window:
                eacc = forget * eacc + esum; nacc = forget * nacc + cnt
                # ---- structural perception: sequential Bayesian selection of leak sites
                r = eacc.copy(); chosen = []; pw = np.zeros(len(RING))
                for _ in range(max_leaks):
                    z = sig.T @ r
                    nc = nacc * cvec
                    logbf = -0.5 * np.log1p(tau2 * nc) + 0.5 * (z ** 2 / nc) * (tau2 * nc / (1 + tau2 * nc))
                    if chosen: logbf[chosen] = -np.inf
                    j = int(np.argmax(logbf))
                    if logbf[j] + logprior_odds <= 0: break
                    d = z[j] / nc[j]
                    chosen.append(j); pw[near[j]] += d ** 2
                    r = r - nacc * Ws[:, j] * d
                if oracle_perception:
                    pw = np.zeros(len(RING))
                    for (c_, a_) in sources_on[t]: pw[NEAR_RING[c_]] += a_ ** 2
                if track_delay:
                    detections.append((t, [cand[j] for j in chosen]))
                eff = effort / cnt; effort[:] = 0
                pw_bar = np.maximum(pw, decay * pw_bar)
                # second-order inference: how long do inflow episodes last? (calibrates the
                # structural horizon, cf. Proposition on regret)
                active = pw > 1.0
                ended = (~active) & (run_len > 0)
                ep_sum += run_len[ended].sum(); ep_n += ended.sum()
                run_len = np.where(active, run_len + 1, 0)
                if adapt_H:
                    H_str = window * ep_sum / ep_n
                if kind == "PA":
                    top = {i for i in np.argsort(-pw_bar)[:3] if pw_bar[i] > 1.0}
                    qn_new = np.array([2.0 if NEAR_RING[o] in top else 0.05 for o in OUT_CELLS])
                    if not np.allclose(qn_new, qn):
                        qn = qn_new; ctrl = lqg(ports, shields, qn); sig, Ws, cvec = refresh(ctrl)
                        eacc[:] = 0; nacc = 0.0
                if kind == "BE":
                    # ---- structural action (inverted detection). Target blanket: the layout
                    # that covers the inferred inflow sites; structure-level EFE = expected
                    # interior cost over the anticipated tenure + reconfiguration cost.
                    infl = np.zeros(len(RING))
                    for rr in range(len(RING)):
                        infl[rr] = np.sum(pw_bar * EXPK[rr])
                    def E_of(P_, S_):
                        rho = np.ones(len(RING))
                        for c in P_: rho[RING.index(c)] *= RHO_P
                        for c in S_: rho[RING.index(c)] *= RHO_S
                        return float(np.sum(infl * rho))
                    # greedy single-resource moves while the structure-level EFE decreases
                    nmove = 0
                    for _step in range(3):
                        Ecur = E_of(ports, shields); best = None
                        occupied = set(ports) | set(shields)
                        free = [cc for cc in RING if cc not in occupied]
                        for which in ("p", "s"):
                            lst = ports if which == "p" else shields
                            for i_old, old in enumerate(lst):
                                for tgt in free:
                                    P_ = list(ports); S_ = list(shields)
                                    if which == "p": P_[i_old] = tgt
                                    else: S_[i_old] = tgt
                                    g_ = H_str * beta_val * (Ecur - E_of(P_, S_)) - kappa
                                    if g_ > 0 and (best is None or g_ > best[0]):
                                        best = (g_, P_, S_)
                        if best is None: break
                        ports, shields = best[1], best[2]; nmove += 1
                    if nmove:
                        moves += nmove
                        ctrl = lqg(ports, shields, qn); Aphys = phys(shields); sig, Ws, cvec = refresh(ctrl)
                        eacc[:] = 0; nacc = 0.0; effort = np.zeros(kp)
                esum[:] = 0; cnt = 0
        if gap_every and t % gap_every == 0:
            gaps.append(gap_diagnostic(ports, shields, sources_on[t]))
    return dict(cost=cost / T, ctrl=ctrl_e / T, moves=moves, viab=viab / T,
                gaps=gaps, ports=ports, shields=shields, detections=detections)

_DC = {}
def _dist(a, b):
    k = (a, b)
    if k not in _DC:
        ra, ca = divmod(a, G); rb, cb = divmod(b, G)
        _DC[k] = abs(ra - rb) + abs(ca - cb)
    return _DC[k]

for _o in range(N):
    NEAR_RING[_o] = min(range(len(RING)), key=lambda r: _dist(_o, RING[r]))

EXPK = np.array([[np.exp(-_dist(RING[a], RING[b]) / ELL) for b in range(len(RING))] for a in range(len(RING))])

# symmetric candidate layouts for the best fixed design (4 ports, 8 shields)
def ring_rc(i): return divmod(RING[i], G)
def layout_candidates():
    rc = {RING[i]: divmod(RING[i], G) for i in range(len(RING))}
    corners = [c for c, (r, cc) in rc.items() if r in (3, 8) and cc in (3, 8)]
    mids = [c for c, (r, cc) in rc.items() if (r in (3, 8) and cc in (5, 6)) or (cc in (3, 8) and r in (5, 6))]
    edge_mid1 = [idx(3, 5), idx(8, 6), idx(5, 8), idx(6, 3)]
    rest = [c for c in RING if c not in corners]
    L = []
    L.append((edge_mid1, corners + [idx(3, 6), idx(8, 5), idx(6, 8), idx(5, 3)]))   # ports mid, shields corners+mids
    L.append((corners, edge_mid1 + [idx(3, 6), idx(8, 5), idx(6, 8), idx(5, 3)]))   # ports corners, shields mids
    L.append((edge_mid1, corners + [idx(3, 4), idx(8, 7), idx(4, 8), idx(7, 3)]))
    L.append(([idx(3, 4), idx(8, 7), idx(4, 8), idx(7, 3)], corners + [idx(3, 6), idx(8, 5), idx(6, 8), idx(5, 3)]))
    return L

def oracle_layout(active_src, kp, ms, ell=1.5):
    infl = np.zeros(len(RING))
    for (c, a) in active_src:
        for r, cell in enumerate(RING):
            infl[r] += abs(a) * np.exp(-_dist(c, cell) / ell)
    order = list(np.argsort(-infl))
    ports = [RING[i] for i in order[:kp]]; shields = [RING[i] for i in order[kp:kp + ms]]
    return ports, shields, infl

def run_oracle(sources_on, T, layout, rng, window=50):
    ports, shields = list(layout[0]), list(layout[1])
    kp, ms = len(ports), len(shields)
    qn = np.full(len(OUT_CELLS), 0.05)
    ctrl = lqg(ports, shields, qn); Aphys = np.eye(N) + DT * conduction_matrix(shields)
    Ttrue = np.zeros(N); xhat = np.zeros(N); cost = 0; viab = 0; moves = 0
    for t in range(T):
        if t % window == 0 and sources_on[t]:
            p2, s2, _ = oracle_layout(sources_on[t], kp, ms)
            if set(p2) != set(ports) or set(s2) != set(shields):
                moves += len(set(p2) - set(ports)) + len(set(s2) - set(shields))
                ports, shields = p2, s2
                ctrl = lqg(ports, shields, qn); Aphys = np.eye(N) + DT * conduction_matrix(shields)
                xhat = xhat.copy()
        u = -ctrl["F"] @ xhat
        inp = np.zeros(N)
        for (c, a) in sources_on[t]: inp[c] += a
        for k, p in enumerate(ports): inp[p] += u[k]
        Ttrue = Aphys @ Ttrue + DT * inp + SIG_BG * np.sqrt(DT) * rng.normal(size=N)
        y = Ttrue[ctrl["obs"]] + 0.01 * rng.normal(size=len(ctrl["obs"]))
        xpred = ctrl["A"] @ xhat + ctrl["B"] @ u
        xhat = xpred + ctrl["K"] @ (y - ctrl["H"] @ xpred)
        dev = Ttrue[I_CELLS]; cost += float(np.mean(dev ** 2)); viab += float(abs(dev.mean()) < 0.3)
    return dict(cost=cost / T, viab=viab / T, moves=moves)

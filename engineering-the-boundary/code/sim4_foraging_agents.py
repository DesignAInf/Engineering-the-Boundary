"""
Simulation 4: active inference agents with a fixed vs. an engineerable Markov blanket.

World. A hidden external state eta_t in {0..K-1} (location of a resource) follows a
sticky Markov chain. M candidate micro-channels emit reports o_m. In regime r only
the channels in INF[r] are informative about eta (accuracy ALPHA); all others emit
uniform noise. Regimes switch at random (geometric dwell time, mean L).

Agent. At each step the agent's sensory blanket S_t (k=2 channels) determines which
reports reach its internal states. It updates q(eta), selects a foraging location a
by minimising expected free energy (risk + ambiguity) and receives reward r=[a==eta].
Wiring a new channel into the blanket costs KAPPA (deducted from net payoff).

Agents
  FB   fixed blanket, fixed precision (treats its channels as reliable)
  PA   fixed blanket, adaptive precision (learns channel reliability = attention)
  RND  random re-wiring at the BE agent's switching rate, adaptive precision
  BE   blanket-engineering agent: infers channel roles (perception of blanket
       membership, cf. DMBD) and re-wires its blanket by minimising a
       structure-level expected free energy (inverted detection)
  OR   oracle: always holds the currently informative blanket, knows reliabilities
"""
import numpy as np
from itertools import combinations

K, M, KSENS = 4, 8, 2
ALPHA, STAY = 0.85, 0.75
INF = [(0, 1), (2, 3), (4, 5), (6, 7)]
LOGPREF = np.array([0.0, 3.0])                  # ln p~(r=0), ln p~(r=1) up to const
LOGPREF = LOGPREF - np.log(np.exp(LOGPREF).sum())
GAMMA = 16.0
HAZ = 0.01            # agent's belief in volatility of channel roles
W0 = 0.5              # prior that a channel is informative
BETA = 1.0            # weight of epistemic value about blanket membership
HSTRUCT = 50.0        # temporal depth of structural planning (anticipated tenure of a blanket)

A_INF = np.full((K, K), (1 - ALPHA) / (K - 1)); np.fill_diagonal(A_INF, ALPHA)  # A[o, eta]
B = np.full((K, K), (1 - STAY) / (K - 1)); np.fill_diagonal(B, STAY)            # B[eta, eta']
PAIRS = list(combinations(range(M), KSENS))

def H(p):
    p = np.clip(p, 1e-12, 1); return -(p * np.log(p)).sum(-1)

def make_world(T, L, rng):
    eta = np.zeros(T, int); reg = np.zeros(T, int)
    e, r = rng.integers(K), 0
    for t in range(T):
        if t > 0:
            if rng.random() > STAY:
                e = rng.choice([j for j in range(K) if j != e])
            if np.isfinite(L) and rng.random() < 1.0 / L:
                r = rng.choice([j for j in range(len(INF)) if j != r])
        eta[t], reg[t] = e, r
    obs = rng.integers(K, size=(T, M))
    for t in range(T):
        for m in INF[reg[t]]:
            obs[t, m] = eta[t] if rng.random() < ALPHA else rng.choice([j for j in range(K) if j != eta[t]])
    return eta, reg, obs

def channel_lik(o, w):
    """p(o | eta) for a channel believed informative with prob w (precision-weighted)."""
    return w * A_INF[o, :] + (1 - w) / K

def efe_actions(qs):
    p1 = qs                                    # q(r=1 | a) = q(eta = a)
    pr = np.stack([1 - p1, p1], -1)
    risk = (pr * (np.log(np.clip(pr, 1e-12, 1)) - LOGPREF)).sum(-1)  # KL[q(r|a) || p~(r)]
    return risk                                # ambiguity = 0 (reward deterministic given eta,a)

def run_agent(kind, eta, reg, obs, kappa, rng, sw_rate=0.0):
    T = len(eta)
    q = np.full(K, 1.0 / K)
    w = np.full(M, W0)
    S = INF[0]
    if kind == "FB":
        w[:] = 1.0
    if kind == "OR":
        w = np.array([1.0 if m in INF[reg[0]] else 0.0 for m in range(M)])
    rew = np.zeros(T); surpr = np.zeros(T); adds = 0; correctS = np.zeros(T)
    Ilog = np.zeros((T, M)); Slog = np.zeros((T, KSENS), int)
    for t in range(T):
        pred = B.T @ q
        # ---------------- volatility of blanket roles (DMBD-like dynamic labels)
        if kind in ("PA", "BE", "RND"):
            w = (1 - HAZ) * w + HAZ * W0
        if kind == "OR":
            w = np.array([1.0 if m in INF[reg[t]] else 0.0 for m in range(M)])
        # ---------------- structural action: choose the blanket
        newS = S
        if kind == "OR":
            newS = INF[reg[t]]
        elif kind == "RND":
            if rng.random() < sw_rate:
                newS = PAIRS[rng.integers(len(PAIRS))]
        elif kind == "BE":
            Ieta = H(A_INF @ pred) - (pred * H(A_INF.T)).sum()      # I(eta; o_m) if informative
            mix = w[:, None] * (A_INF @ pred)[None, :] + (1 - w[:, None]) / K   # p(o_m)
            IGiota = H(mix) - (w * H(A_INF @ pred) + (1 - w) * np.log(K))     # I(iota_m; o_m)
            best, bestG = S, None
            for P in PAIRS:
                P_ = list(P)
                added = len(set(P) - set(S))
                # structure-level EFE: instrumental epistemic value about eta over the
                # anticipated tenure + one-shot epistemic value about blanket roles + cost
                G = -HSTRUCT * (w[P_] * Ieta).sum() - BETA * IGiota[P_].sum() + kappa * added
                if bestG is None or G < bestG - 1e-9:
                    best, bestG = P, G
            newS = best
        adds += len(set(newS) - set(S))
        S = tuple(newS)
        Slog[t] = S
        correctS[t] = set(S) == set(INF[reg[t]])
        # ---------------- perception through the blanket
        liks = {m: channel_lik(obs[t, m], w[m]) for m in S}
        qs = pred.copy()
        for m in S:
            qs = qs * liks[m]
        qs /= qs.sum()
        surpr[t] = -np.log(max(qs[eta[t]], 1e-12))
        # ---------------- action: minimise EFE
        G = efe_actions(qs)
        p = np.exp(-GAMMA * (G - G.min())); p /= p.sum()
        a = rng.choice(K, p=p)
        r = int(a == eta[t]); rew[t] = r
        lr = np.full(K, 1.0 - r, float); lr[a] = r
        post = qs * lr
        post = post / post.sum() if post.sum() > 0 else qs
        # ---------------- learning channel reliability (precision / attention)
        if kind in ("PA", "BE", "RND"):
            for m in S:
                loo = pred * lr
                for n in S:
                    if n != m:
                        loo = loo * liks[n]
                if loo.sum() == 0:
                    continue
                loo /= loo.sum()
                p_inf = (A_INF[obs[t, m], :] * loo).sum()
                w[m] = w[m] * p_inf / (w[m] * p_inf + (1 - w[m]) / K)
        Ilog[t] = w
        q = post
    return dict(reward=rew, surprisal=surpr, adds=adds, correctS=correctS,
                w=Ilog, S=Slog, net=(rew.sum() - kappa * adds) / T)

def viability(rew, win=50, thr=0.5):
    c = np.convolve(rew, np.ones(win) / win, mode="valid")
    return (c >= thr).mean()

def run_condition(L, kappa, seeds, T, agents=("FB", "PA", "RND", "BE", "OR"), keep_traces=False):
    out = {a: [] for a in agents}
    traces = None
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        eta, reg, obs = make_world(T, L, rng)
        res_be = run_agent("BE", eta, reg, obs, kappa, np.random.default_rng(s))
        sw = res_be["adds"] / KSENS / T          # BE switching rate (pairs per step)
        for a in agents:
            r = res_be if a == "BE" else run_agent(a, eta, reg, obs, kappa,
                                                   np.random.default_rng(s), sw_rate=sw)
            out[a].append(dict(net=r["net"], reward=r["reward"].mean(),
                               surprisal=r["surprisal"].mean(),
                               viab=viability(r["reward"]), adds=r["adds"],
                               correctS=r["correctS"].mean()))
            if keep_traces and s == seeds[0]:
                traces = traces or dict(reg=reg, eta=eta)
                traces[a] = r
    return out, traces

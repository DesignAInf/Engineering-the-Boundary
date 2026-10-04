"""
Simulation 3: inversion with a DMBD-style detector.

A system of microscopic elements reads out a latent Markov-blanketed dynamical system
(s = environment, b = boundary, z = object). Four elements that the designer *intends* to be
environment (target partition omega*) are physically coupled to the object with strengths
eps = (0.1, 0.4, 0.8, 1.2): they are microscopic leaks. The designer/agent chooses
structural interventions u (shield any subset of the leaking elements, cost kappa each;
or 'recruit' the leaking elements into the boundary, i.e. adopt a larger target blanket at
a cost per boundary element) by the structure-level EFE of Eq. (9):

    G(u) = H * breach(u) + C(u),
where breach(u) is the detector's element-level evidence that the target partition is violated
(the realisation term read through the evidence, since leaks do not change the MAP labels).

Detection = MiniDMBD with a focused prior on the target partition.
"""
from paths import FIGDIR, RESDIR
import itertools, json, sys, time
import numpy as np
from multiprocessing import Pool
from mini_dmbd import MiniDMBD

EPS = np.array([0.1, 0.4, 0.8, 1.2])
nS, nB, nZ = 20, 8, 8
LEAKERS = np.arange(4)            # the first four environment elements

def gen(shield=(), T=2000, seed=0, eps=EPS):
    r = np.random.default_rng(seed)
    D = 6; A = np.zeros((D, D))
    A[:2, :2] = [[0.95, 0.1], [-0.1, 0.95]]; A[2:4, 2:4] = [[0.9, 0.05], [-0.05, 0.9]]
    A[4:, 4:] = [[0.93, -0.15], [0.15, 0.93]]
    rs = np.random.default_rng(12345)              # fixed system (same for all u)
    A[2:4, :2] = 0.08 * rs.normal(size=(2, 2)); A[:2, 2:4] = 0.08 * rs.normal(size=(2, 2))
    A[2:4, 4:] = 0.08 * rs.normal(size=(2, 2)); A[4:, 2:4] = 0.08 * rs.normal(size=(2, 2))
    rho = np.abs(np.linalg.eigvals(A)).max()
    if rho > 0.97: A *= 0.97 / rho
    labels = np.array([0] * nS + [1] * nB + [2] * nZ)
    C = rs.normal(size=(len(labels), 2)); Gz = rs.normal(size=(4, 2))
    x = np.zeros(D); X = np.zeros((T, D))
    for t in range(T):
        x = A @ x + 0.3 * r.normal(size=D); X[t] = x
    Y = np.zeros((T, len(labels)))
    for i, l in enumerate(labels):
        Y[:, i] = X[:, 2 * l:2 * l + 2] @ C[i] + 0.5 * r.normal(size=T)
    for k in LEAKERS:
        if k in shield: continue
        Y[:, k] += eps[k] * (X[:, 4:] @ Gz[k])      # direct coupling of an 'environment' element to the object
    return Y, labels

def true_cmi(shield=(), eps=EPS, relabel=()):
    """exact I(env elements ; object elements | boundary elements) of the stationary
    linear-Gaussian system (discrete Lyapunov equation)."""
    from scipy.linalg import solve_discrete_lyapunov
    D = 6; A = np.zeros((D, D))
    A[:2, :2] = [[0.95, 0.1], [-0.1, 0.95]]; A[2:4, 2:4] = [[0.9, 0.05], [-0.05, 0.9]]
    A[4:, 4:] = [[0.93, -0.15], [0.15, 0.93]]
    rs = np.random.default_rng(12345)
    A[2:4, :2] = 0.08 * rs.normal(size=(2, 2)); A[:2, 2:4] = 0.08 * rs.normal(size=(2, 2))
    A[2:4, 4:] = 0.08 * rs.normal(size=(2, 2)); A[4:, 2:4] = 0.08 * rs.normal(size=(2, 2))
    rho = np.abs(np.linalg.eigvals(A)).max()
    if rho > 0.97: A *= 0.97 / rho
    labels = np.array([0] * nS + [1] * nB + [2] * nZ)
    C = rs.normal(size=(len(labels), 2)); Gz = rs.normal(size=(4, 2))
    Sx = solve_discrete_lyapunov(A, 0.09 * np.eye(D))
    H = np.zeros((len(labels), D))
    for i, l in enumerate(labels): H[i, 2 * l:2 * l + 2] = C[i]
    for k in LEAKERS:
        if k not in shield: H[k, 4:] += eps[k] * Gz[k]
    Sy = H @ Sx @ H.T + 0.25 * np.eye(len(labels))
    Y = [i for i in np.where(labels == 0)[0] if i not in relabel]
    B = list(np.where(labels == 1)[0]) + list(relabel); X = list(np.where(labels == 2)[0])
    ld = lambda ix: np.linalg.slogdet(Sy[np.ix_(ix, ix)])[1]
    return 0.5 * (ld(Y + B) + ld(X + B) - ld(B) - ld(Y + B + X))

def fit_logq(args):
    shield, target_relabel, seed = args
    Y, lab = gen(shield=shield, seed=seed)
    target = lab.copy()
    for k in target_relabel: target[k] = 1      # recruit into the boundary
    pri = np.full((len(lab), 3), 0.2); pri[np.arange(len(lab)), target] = 0.6
    m = MiniDMBD(prior=pri, seed=1).fit(Y, iters=40)
    pred = m.q.argmax(1)
    br = m.breach(target)
    return dict(shield=list(shield), relabel=list(target_relabel), seed=seed, cmi=float(true_cmi(shield, relabel=target_relabel)),
                breach=float(br.sum()), breach_leakers=[float(br[k]) for k in LEAKERS],
                breach_max_other=float(np.max(np.delete(br, LEAKERS))),
                logq=m.log_q_of(target), mislabeled=int((pred != target).sum()),
                leak_labels=[int(pred[k]) for k in LEAKERS])

if __name__ == "__main__":
    t0 = time.time()
    options = []
    for k in range(5):
        for S in itertools.combinations(range(4), k):
            options.append((tuple(S), ()))
    # recruit options: relabel the unshielded strong leakers as boundary
    for R in [(3,), (2, 3), (1, 2, 3), (0, 1, 2, 3)]:
        options.append(((), R))
    jobs = [(S, R, seed) for (S, R) in options for seed in range(3)]
    with Pool(2) as p:
        res = p.map(fit_logq, jobs)
    json.dump(res, open(RESDIR + "inversion_results.json", "w"), indent=0)
    print("done %.0f s" % (time.time() - t0))

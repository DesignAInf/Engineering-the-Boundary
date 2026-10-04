"""
A minimal NumPy re-implementation of the core of Dynamic Markov Blanket Detection
(Beck & Ramstead 2025, arXiv:2502.21217; reference code: github.com/bayesianempirimancer/pyDMBD).

Simplifications relative to the original (documented in the paper):
  * one object, one role per class (environment s, boundary b, object z);
  * assignment labels are static over the analysed window (no label HMM);
  * parameters (A, Q, loadings, offsets, noise) are point-estimated (variational EM)
    instead of carrying Matrix-Normal-Wishart posteriors;
  * element-specific loadings (the original uses role-specific emission matrices).
Kept from the original: latent linear dynamics with Markov-blanket masking of the transition
matrix (A_sz = A_zs = 0), assignment of every microscopic element to s/b/z by a Bayesian
attention step, and coordinate ascent on an evidence lower bound (attend - infer - repeat).
"""
import numpy as np

ROLES = ("s", "b", "z")

class MiniDMBD:
    def __init__(self, dims=(2, 2, 2), prior=None, seed=0):
        self.ds, self.db, self.dz = dims
        self.D = sum(dims)
        self.blocks = {"s": np.arange(0, self.ds),
                       "b": np.arange(self.ds, self.ds + self.db),
                       "z": np.arange(self.ds + self.db, self.D)}
        self.mask = np.ones((self.D, self.D))
        for i in self.blocks["s"]:
            for j in self.blocks["z"]:
                self.mask[i, j] = self.mask[j, i] = 0.0
        self.prior = prior
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ Kalman smoother
    def _smooth(self, Y):
        T, n = Y.shape
        D = self.D
        # mean-field observation model: each (element, role) is a pseudo-observation of the
        # role block with precision q_ir / r_ir
        Hrows, prec, offs, idx = [], [], [], []
        for i in range(n):
            for k, r in enumerate(ROLES):
                q = self.q[i, k]
                if q < 1e-4: continue
                h = np.zeros(D); h[self.blocks[r]] = self.C[i, k]
                Hrows.append(h); prec.append(q / self.R[i, k]); offs.append(self.d[i, k]); idx.append(i)
        Hm = np.array(Hrows); Pm = np.array(prec); off = np.array(offs); idx = np.array(idx)
        J = Hm.T @ (Hm * Pm[:, None])                        # information added per step
        A, Q = self.A, self.Q
        mu_f = np.zeros((T, D)); P_f = np.zeros((T, D, D))
        mu, P = np.zeros(D), np.eye(D)
        loglik = 0.0
        Qi = np.linalg.inv(Q)
        for t in range(T):
            if t > 0:
                mu = A @ mu; P = A @ P @ A.T + Q
            h = Hm.T @ (Pm * (Y[t, idx] - off))
            Pi = np.linalg.inv(P)
            Pn = np.linalg.inv(Pi + J)
            mun = Pn @ (Pi @ mu + h)
            mu, P = mun, Pn
            mu_f[t], P_f[t] = mu, P
        # RTS
        mu_s = mu_f.copy(); P_s = P_f.copy(); Cross = np.zeros((T - 1, D, D))
        for t in range(T - 2, -1, -1):
            Pp = A @ P_f[t] @ A.T + Q
            G = P_f[t] @ A.T @ np.linalg.inv(Pp)
            mu_s[t] = mu_f[t] + G @ (mu_s[t + 1] - A @ mu_f[t])
            P_s[t] = P_f[t] + G @ (P_s[t + 1] - Pp) @ G.T
            Cross[t] = P_s[t + 1] @ G.T                         # Cov(x_{t+1}, x_t)
        return mu_s, P_s, Cross

    # ------------------------------------------------------------ fit
    def fit(self, Y, iters=30, init_labels=None, verbose=False):
        T, n = Y.shape
        Y = (Y - Y.mean(0)) / Y.std(0)
        self.Yn = Y
        if self.prior is None:
            self.logprior = np.log(np.full((n, 3), 1 / 3))
        else:
            self.logprior = np.log(self.prior)
        if init_labels is None:
            self.q = np.exp(self.logprior); self.q /= self.q.sum(1, keepdims=True)
            self.q = 0.7 * self.q + 0.3 * self.rng.dirichlet(np.ones(3), size=n)
        else:
            self.q = np.full((n, 3), 0.05); self.q[np.arange(n), init_labels] = 0.9
        # ---- initialisation from the (focused) prior: PCA of each provisional group
        lab0 = self.q.argmax(1)
        X0 = np.zeros((T, self.D))
        for k, r in enumerate(ROLES):
            bl = self.blocks[r]; members = np.where(lab0 == k)[0]
            if len(members) == 0: members = np.arange(n)
            U, S_, Vt = np.linalg.svd(Y[:, members], full_matrices=False)
            X0[:, bl] = U[:, :len(bl)] * S_[:len(bl)] / np.sqrt(T)
        A = np.zeros((self.D, self.D))
        for i in range(self.D):
            al = np.where(self.mask[i] > 0)[0]
            A[i, al] = np.linalg.lstsq(X0[:-1][:, al], X0[1:, i], rcond=None)[0]
        self.A = A
        self.Q = np.diag(np.maximum(np.var(X0[1:] - X0[:-1] @ A.T, 0), 1e-3))
        self.C = np.zeros((n, 3, max(self.ds, self.db, self.dz))); self.d = np.zeros((n, 3)); self.R = np.ones((n, 3))
        for k, r in enumerate(ROLES):
            bl = self.blocks[r]
            X1 = np.hstack([X0[:, bl], np.ones((T, 1))])
            W = np.linalg.lstsq(X1, Y, rcond=None)[0]
            self.C[:, k, :len(bl)] = W[:len(bl)].T; self.d[:, k] = W[-1]
            self.R[:, k] = np.maximum(np.var(Y - X1 @ W, 0), 1e-3)
        self.history = []
        for it in range(iters):
            mu, P, Cr = self._smooth(Y)
            Exx = np.einsum("ti,tj->ij", mu, mu) + P.sum(0)
            Exx0 = np.einsum("ti,tj->ij", mu[:-1], mu[:-1]) + P[:-1].sum(0)
            Exx1 = np.einsum("ti,tj->ij", mu[1:], mu[1:]) + P[1:].sum(0)
            Ex1x0 = np.einsum("ti,tj->ij", mu[1:], mu[:-1]) + Cr.sum(0)
            # ---- M-step: masked dynamics, row by row (least squares on allowed entries)
            A = np.zeros((self.D, self.D))
            for i in range(self.D):
                al = np.where(self.mask[i] > 0)[0]
                A[i, al] = np.linalg.solve(Exx0[np.ix_(al, al)] + 1e-6 * np.eye(len(al)), Ex1x0[i, al])
            self.A = A
            Qd = np.diag(Exx1 - A @ Ex1x0.T - Ex1x0 @ A.T + A @ Exx0 @ A.T) / (T - 1)
            self.Q = np.diag(np.maximum(Qd, 1e-4))
            # ---- emissions for every element under every role; expected log-likelihoods
            ell = np.zeros((n, 3))
            for k, r in enumerate(ROLES):
                bl = self.blocks[r]
                m = mu[:, bl]; Sxx = Exx[np.ix_(bl, bl)]
                X1 = np.hstack([m, np.ones((T, 1))])
                S1 = np.zeros((len(bl) + 1, len(bl) + 1))
                S1[:len(bl), :len(bl)] = Sxx; S1[:len(bl), -1] = m.sum(0); S1[-1, :len(bl)] = m.sum(0); S1[-1, -1] = T
                W = np.linalg.solve(S1 + 1e-6 * np.eye(len(bl) + 1), X1.T @ Y)       # (len(bl)+1, n)
                c = W[:len(bl)].T; dd = W[-1]
                resid2 = (Y ** 2).sum(0) - 2 * np.einsum("in,in->n", W, X1.T @ Y) + np.einsum("in,ij,jn->n", W, S1, W)
                rr = np.maximum(resid2 / T, 1e-4)
                self.C[:, k, :len(bl)] = c; self.d[:, k] = dd; self.R[:, k] = rr
                ell[:, k] = -0.5 * T * (np.log(2 * np.pi * rr) + 1.0)
            # ---- attend: assignment posteriors
            temp = max(1.0, (T / 20.0) * (1 - it / max(1, iters - 5)))   # annealed attention
            lq = self.logprior + ell / temp
            lq -= lq.max(1, keepdims=True)
            self.q = np.exp(lq); self.q /= self.q.sum(1, keepdims=True)
            self.ell = ell
            self.history.append(float((self.q * (self.logprior + ell)).sum() - (self.q * np.log(self.q + 1e-300)).sum()))
            if verbose: print(it, self.history[-1], np.bincount(self.q.argmax(1), minlength=3))
        return self

    def log_q_of(self, labels):
        """log posterior mass q(omega* | data) of a target labelling (mean-field product)."""
        return float(np.sum(np.log(self.q[np.arange(len(labels)), labels] + 1e-300)))

    def loglik(self, Y=None):
        """exact log-likelihood of the data under the fitted LDS with hard (MAP) labels."""
        Y = self.Yn if Y is None else Y
        T, n = Y.shape
        lab = self.q.argmax(1)
        D = self.D
        H = np.zeros((n, D)); off = np.zeros(n); Rv = np.zeros(n)
        for i in range(n):
            r = ROLES[lab[i]]; H[i, self.blocks[r]] = self.C[i, lab[i], :len(self.blocks[r])]
            off[i] = self.d[i, lab[i]]; Rv[i] = self.R[i, lab[i]]
        mu, P = np.zeros(D), np.eye(D); ll = 0.0
        for t in range(T):
            if t > 0: mu = self.A @ mu; P = self.A @ P @ self.A.T + self.Q
            S = H @ P @ H.T + np.diag(Rv)
            e = Y[t] - H @ mu - off
            L = np.linalg.cholesky(S)
            v = np.linalg.solve(L, e)
            ll += -0.5 * (v @ v) - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi)
            K = P @ H.T @ np.linalg.inv(S)
            mu = mu + K @ e; P = P - K @ H @ P
        return ll

    def breach(self, labels):
        """per-element log-likelihood ratio (nats per time step) of an unconstrained emission
        (loading on s, b and z) against the emission allowed by the target partition. This is
        the evidence, at the level of each microscopic element, that the element violates the
        target blanket: a DMBD-based estimate of where and how much the boundary leaks."""
        Y = self.Yn; T, n = Y.shape
        mu, P, _ = self._smooth(Y)
        Exx = np.einsum("ti,tj->ij", mu, mu) + P.sum(0)
        def ll(bl):
            m = mu[:, bl]; Sxx = Exx[np.ix_(bl, bl)]
            X1 = np.hstack([m, np.ones((T, 1))])
            S1 = np.zeros((len(bl) + 1, len(bl) + 1))
            S1[:len(bl), :len(bl)] = Sxx; S1[:len(bl), -1] = m.sum(0); S1[-1, :len(bl)] = m.sum(0); S1[-1, -1] = T
            W = np.linalg.solve(S1 + 1e-6 * np.eye(len(bl) + 1), X1.T @ Y)
            resid2 = (Y ** 2).sum(0) - 2 * np.einsum("in,in->n", W, X1.T @ Y) + np.einsum("in,ij,jn->n", W, S1, W)
            return -0.5 * (np.log(2 * np.pi * np.maximum(resid2 / T, 1e-6)) + 1.0)
        full = ll(np.arange(self.D))
        per_role = np.stack([ll(self.blocks[r]) for r in ROLES], 1)
        return full - per_role[np.arange(n), labels]

from paths import FIGDIR, RESDIR
import json, sys, time
import numpy as np
from multiprocessing import Pool
import sim5_pavilion as th

# --fixed-horizon reproduces the ablation with a fixed anticipated tenure (adapt_H=False)
FIXED = '--fixed-horizon' in sys.argv
TAG = '_fixed_horizon' if FIXED else ''

LAY = th.layout_candidates()[2]          # best fixed layout (pilot runs, independent seeds)
T = 6000

def world(seed, life):
    rng = np.random.default_rng(10_000 + seed)
    srcs = th.make_sources(T, rate=1 / 300, life=life, rng=rng)
    return srcs, th.source_field(srcs, T)

def job(args):
    kind, seed, life, kappa, extra = args
    srcs, on = world(seed, life)
    kw = dict(kappa=kappa)
    if FIXED: kw['adapt_H'] = False
    kw.update(extra)
    name = kind
    if kind == "BEO":
        kind_ = "BE"; kw["oracle_perception"] = True
    else:
        kind_ = kind
    r = th.run(kind_, on, T, LAY, np.random.default_rng(seed), **kw)
    out = dict(kind=name, seed=seed, life=life, kappa=kappa, extra={k: v for k, v in extra.items() if k != "track_delay"},
               cost=r["cost"], viab=r["viab"], moves=r["moves"], ctrl=r["ctrl"],
               net=r["cost"] + kappa * r["moves"] / T,
               gap=float(np.mean(r["gaps"])) if r["gaps"] else None)
    if extra.get("track_delay"):
        # detection delays and false alarms
        onsets = [(t0, c) for (t0, t1, c, a) in srcs if t1 - t0 > 150]
        delays = []
        for (t0, c) in onsets:
            target = th.NEAR_RING[c]
            d = None
            for (t, chosen) in r["detections"]:
                if t < t0: continue
                if t > t0 + 1000: break
                if any(th._dist(th.RING[th.NEAR_RING[x]], th.RING[target]) <= 1 for x in chosen):
                    d = t - t0; break
            delays.append(d if d is not None else 1000)
        fa = 0; quiet = 0
        for (t, chosen) in r["detections"]:
            if not on[t]:
                quiet += 1; fa += int(len(chosen) > 0)
        out["delays"] = delays; out["fa_rate"] = fa / max(quiet, 1)
    return out

if __name__ == "__main__":
    t0 = time.time()
    jobs = []
    # main condition
    for s in range(16):
        for k in ["FB", "PA", "BE", "BEO"]:
            jobs.append((k, s, 1000, 2.0, dict(gap_every=250)))
    # sweep over source lifetime and re-wiring cost
    for life in [100, 300, 1000, 3000]:
        for kap in [0.5, 4.0, 32.0]:
            for s in range(8):
                for k in ["FB", "BE"]:
                    if life == 1000 and kap == 2.0: continue
                    jobs.append((k, 100 + s, life, kap, {}))
    # Occam study: breadth of structural prior
    for om in [128, 10 ** 3, 10 ** 5, 10 ** 8, 10 ** 12]:
        for s in range(8):
            jobs.append(("BE", 200 + s, 1000, 2.0, dict(omega_size=om, track_delay=True)))
    print("jobs", len(jobs), flush=True)
    done = set()
    try:
        for line in open(RESDIR + "thermal_results" + TAG + ".jsonl"):
            x = json.loads(line); done.add((x["kind"], x["seed"], x["life"], x["kappa"], json.dumps(x["extra"], sort_keys=True)))
    except FileNotFoundError:
        pass
    key = lambda j: (j[0], j[1], j[2], j[3], json.dumps({k: v for k, v in j[4].items() if k != "track_delay"}, sort_keys=True))
    jobs = [j for j in jobs if key(j) not in done]
    print("remaining", len(jobs), flush=True)
    with Pool(2, maxtasksperchild=4) as p, open(RESDIR + "thermal_results" + TAG + ".jsonl", "a") as f:
        for res in p.imap_unordered(job, jobs):
            f.write(json.dumps(res) + "\n"); f.flush()
    res = [json.loads(l) for l in open(RESDIR + "thermal_results" + TAG + ".jsonl")]
    json.dump(res, open(RESDIR + "thermal_results" + TAG + ".json", "w"))
    print("done in %.0f s" % (time.time() - t0))

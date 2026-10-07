"""Noise-filter tuning on TRAIN seeds only (100-105). Output was saved to results_baf_tuning.txt."""
import sys; sys.path.insert(0, '.')
import numpy as np, run_experiments as R, pipeline as P, simulator as S
# Tuning uses TRAIN-set seeds only (100-105); the test seeds 200-205 are never touched here.
R.DUR = 1.5
cache = {}
for lux in (0.01, 0.003):
    for s in (100, 101, 102, 105):
        sc = S.make_scene(s, duration_s=1.5); ev, fr, ft = S.simulate(sc, lux, seed=s); cache[(lux, s)] = (sc, ev, ft)
for dt, k in [(3000, 1), (2000, 2), (1500, 2), (1000, 2), (1000, 3), (600, 2), (2000, 3)]:
    P.BAF_DT = dt; P.BAF_K = k
    row = []
    for lux in (0.01, 0.003):
        recs = {}
        for s in (100, 101, 102, 105):
            sc, ev, ft = cache[(lux, s)]
            eo = P.run_event_pipeline(ev, ft, R.WIN, S.H, S.W)
            recs[s] = dict(scene=sc, ftimes=ft, ev_out=eo)
        X, Y = zip(*[R.collect(recs[s], 'ev') for s in (100, 101, 102)]); X = np.concatenate(X); Y = np.concatenate(Y)
        if Y.sum() < 5:
            row.append((lux, 'too few positives', int(Y.sum()), len(Y))); continue
        clf = R.fit(X, Y)
        o = R.evaluate(recs[105], 'ev', clf, 0.5)[0]
        row.append((lux, round(R.prf(o)[2], 2), int(Y.sum()), len(Y)))
    print(dt, k, row, flush=True)

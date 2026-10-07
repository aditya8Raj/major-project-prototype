"""
Extended experiments on the software prototype (Chapter 7 of the report).

All experiments reuse the simulator, the pipeline and the evaluation functions of
run_experiments.py. The design of the pipeline was frozen BEFORE these runs; they
only measure how sensitive the results are. Nothing here is used to choose a
setting. Train scenes use seeds 100-105 and test scenes 200-205, as before.

Writes results/extended_results.json.   Run:  python extended_experiments.py
"""
import json, os, time
import numpy as np
import simulator as S
import pipeline as P
import run_experiments as R

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
T0 = time.time()
LUX = R.LUX
NAMES = list(LUX)
RES = {}


def log(*a):
    print(round(time.time() - T0), *a, flush=True)


def save():
    with open(os.path.join(OUT, 'extended_results.json'), 'w') as fh:
        json.dump(RES, fh, indent=1)


def build_raw(seed, lux, dur=R.DUR):
    sc = S.make_scene(seed, duration_s=dur)
    ev, fr, ft = S.simulate(sc, lux, seed=seed)
    return dict(scene=sc, ev=ev, fr=fr, ft=ft)


def make_rec(raw, with_frame=True, win=None):
    win = win or R.WIN
    eo = P.run_event_pipeline(raw['ev'], raw['ft'], win, S.H, S.W)
    fo = P.run_frame_pipeline(raw['fr']) if with_frame else None
    return dict(scene=raw['scene'], ftimes=raw['ft'], ev_out=eo, fr_out=fo,
                n_events=len(raw['ev']), dur=R.DUR)


def train_model(recs, which, idx=None):
    X, Y = zip(*[R.collect(r, which) for r in recs])
    X = np.concatenate(X); Y = np.concatenate(Y)
    clf = R.fit(X if idx is None else X[:, idx], Y)
    best = (0, 0.5)
    for thr in np.arange(0.2, 0.85, 0.05):
        agg = dict(tp=0, fp=0, fn=0)
        for r in recs:
            o = R.evaluate(r, which, clf, thr, feat_idx=idx)[0]
            for k in agg: agg[k] += o[k]
        best = max(best, (R.prf(agg)[2], float(thr)))
    return clf, best[1]


def score(clf, thr, test_by_lux, which, idx=None):
    out = {}
    for n in test_by_lux:
        agg = dict(tp=0, fp=0, fn=0)
        for r in test_by_lux[n]:
            o = R.evaluate(r, which, clf, thr, feat_idx=idx)[0]
            for k in agg: agg[k] += o[k]
        p, q, f = R.prf(agg)
        out[n] = dict(precision=p, recall=q, f1=f, **agg)
    return out


def cand_stats(recs_by_lux, which):
    out = {}
    for n in NAMES:
        hit = tot = ncand = nwin = 0
        for r in recs_by_lux[n]:
            for k, te in enumerate(r['ftimes']):
                gt, _ = R.gt_window(r['scene'], te, R.WIN)
                c = r['ev_out'][k][2] if which == 'ev' else r['fr_out'][k][1]
                tot += len(gt); ncand += len(c); nwin += 1
                if len(gt) and len(c):
                    d = np.hypot(c[:, None, 0] - gt[None, :, 0], c[:, None, 1] - gt[None, :, 1])
                    hit += int((d.min(0) <= 6.0).sum())
        out[n] = dict(candidate_recall=hit / max(tot, 1), candidates_per_window=ncand / max(nwin, 1), n_gt=tot)
    return out


# ------------------------------------------------------------------ baseline data
log('simulating train and test scenes')
raw_tr = {n: [build_raw(s, LUX[n]) for s in R.TRAIN_SEEDS] for n in NAMES}
log('train simulated')
raw_te = {n: [build_raw(s, LUX[n]) for s in R.TEST_SEEDS] for n in NAMES}
log('test simulated')
tr = {n: [make_rec(x) for x in raw_tr[n]] for n in NAMES}
te = {n: [make_rec(x) for x in raw_te[n]] for n in NAMES}
train_all = [r for n in NAMES for r in tr[n]]
clf_ev, thr_ev = train_model(train_all, 'ev')
clf_fr, thr_fr = train_model(train_all, 'fr')
base_ev = score(clf_ev, thr_ev, te, 'ev'); base_fr = score(clf_fr, thr_fr, te, 'fr')
published = json.load(open(os.path.join(OUT, 'results.json')))['test']
RES['reproduction_check'] = {n: dict(ev_now=base_ev[n]['f1'], ev_report=published[n]['ev']['f1'],
                                     fr_now=base_fr[n]['f1'], fr_report=published[n]['fr']['f1']) for n in NAMES}
log('baseline', {n: (round(base_ev[n]['f1'], 3), round(base_fr[n]['f1'], 3)) for n in NAMES}); save()

# E1 candidate stage
RES['candidates'] = dict(event=cand_stats(te, 'ev'), frame=cand_stats(te, 'fr'))
for n in NAMES:
    X, Y = zip(*[R.collect(r, 'ev') for r in te[n]])
    RES['candidates']['event'][n]['n_candidates'] = int(sum(len(x) for x in X))
    RES['candidates']['event'][n]['n_moth_candidates'] = int(sum(y.sum() for y in Y))
save(); log('E1 candidates done')

# E2 threshold sweep (precision-recall)
sweep = {}
for which, clf in (('ev', clf_ev), ('fr', clf_fr)):
    sweep[which] = {}
    for n in NAMES:
        rows = []
        for thr in np.arange(0.1, 0.96, 0.05):
            agg = dict(tp=0, fp=0, fn=0)
            for r in te[n]:
                o = R.evaluate(r, which, clf, thr)[0]
                for k in agg: agg[k] += o[k]
            p, q, f = R.prf(agg)
            rows.append(dict(thr=float(thr), precision=p, recall=q, f1=f))
        sweep[which][n] = rows
RES['threshold_sweep'] = sweep; RES['chosen_thresholds'] = dict(ev=thr_ev, fr=thr_fr); save(); log('E2 done')

# E3 noise filter settings (event side only, retrained each time)
baf = {}
for label, (dt, k, use) in {'no filter': (0, 0, False), 'k=1, 1 ms': (1000, 1, True), 'k=2, 1 ms (used)': (1000, 2, True),
                            'k=3, 1 ms': (1000, 3, True), 'k=2, 0.5 ms': (500, 2, True), 'k=2, 2 ms': (2000, 2, True)}.items():
    P.BAF_DT, P.BAF_K = dt, k
    def rebuild(raws):
        out = []
        for x in raws:
            eo = P.run_event_pipeline(x['ev'], x['ft'], R.WIN, S.H, S.W, use_baf=use)
            out.append(dict(scene=x['scene'], ftimes=x['ft'], ev_out=eo, fr_out=None, n_events=len(x['ev']), dur=R.DUR))
        return out
    trb = [r for n in NAMES for r in rebuild(raw_tr[n])]
    teb = {n: rebuild(raw_te[n]) for n in NAMES}
    clf, thr = train_model(trb, 'ev')
    baf[label] = score(clf, thr, teb, 'ev')
    baf[label]['_cand'] = cand_stats(teb, 'ev')
    log('E3', label, {n: round(baf[label][n]['f1'], 3) for n in NAMES})
P.BAF_DT, P.BAF_K = 1000, 2
RES['noise_filter'] = baf; save()

# E4 evaluation window length
win_res = {}
for wms in (16.7, 33.3, 66.7):
    win = int(wms * 1000); R.WIN = win
    trb = [r for n in NAMES for r in [make_rec(x, False, win) for x in raw_tr[n]]]
    teb = {n: [make_rec(x, False, win) for x in raw_te[n]] for n in NAMES}
    clf, thr = train_model(trb, 'ev')
    win_res[str(wms)] = score(clf, thr, teb, 'ev')
    log('E4', wms, {n: round(win_res[str(wms)][n]['f1'], 3) for n in NAMES})
R.WIN = int(1e6 / S.FPS)
RES['window_length'] = win_res; save()

# E5 time constants and EMD shift: full model and model without the T4/T5 feature
dsi_idx = [i for i, f in enumerate(P.FEATURES) if f != 'dsi']
tau_res = {}
for label, (tf, ts, sh) in {'5 ms / 30 ms, shift 2 (used)': (5000, 30000, 2), '2.5 ms / 15 ms, shift 2': (2500, 15000, 2),
                            '10 ms / 60 ms, shift 2': (10000, 60000, 2), '5 ms / 30 ms, shift 1': (5000, 30000, 1),
                            '5 ms / 30 ms, shift 4': (5000, 30000, 4)}.items():
    P.TAU_FAST, P.TAU_SLOW, P.EMD_SHIFT = tf, ts, sh
    trb = [r for n in NAMES for r in [make_rec(x, False) for x in raw_tr[n]]]
    teb = {n: [make_rec(x, False) for x in raw_te[n]] for n in NAMES}
    clf, thr = train_model(trb, 'ev')
    full = score(clf, thr, teb, 'ev')
    clf2, thr2 = train_model(trb, 'ev', dsi_idx)
    nodsi = score(clf2, thr2, teb, 'ev', dsi_idx)
    tau_res[label] = dict(full=full, without_dsi=nodsi)
    log('E5', label, {n: (round(full[n]['f1'], 3), round(nodsi[n]['f1'], 3)) for n in NAMES})
P.TAU_FAST, P.TAU_SLOW, P.EMD_SHIFT = 5000, 30000, 2
RES['time_constants'] = tau_res; save()

# E6 match tolerance (same detections, different scoring radius)
orig_match = R.match
tol_res = {}
for tol in (4.0, 6.0, 10.0, 15.0, 20.0):
    R.match = lambda c, g, tol_=tol: orig_match(c, g, tol_)
    tol_res[str(tol)] = dict(event=score(clf_ev, thr_ev, te, 'ev'), frame=score(clf_fr, thr_fr, te, 'fr'))
R.match = orig_match
RES['match_tolerance'] = {k: {w: {n: v[n]['f1'] for n in NAMES} for w, v in d.items()} for k, d in tol_res.items()}
save(); log('E6 done')

# E7 number of training scenes per light level
ntr = {}
for m in (1, 2, 3, 6):
    sub = [r for n in NAMES for r in tr[n][:m]]
    clf, thr = train_model(sub, 'ev')
    ntr[str(m)] = score(clf, thr, te, 'ev')
    log('E7', m, {n: round(ntr[str(m)][n]['f1'], 3) for n in NAMES})
RES['training_size'] = ntr; save()

# E8 permutation importance of the 12 candidate features (event classifier)
rng = np.random.default_rng(0)
imp = {n: {} for n in NAMES}
for n in NAMES:
    for j, fname in enumerate(P.FEATURES):
        drops = []
        for rep in range(3):
            allf = np.concatenate([o[1] for r in te[n] for o in r['ev_out']])
            perm = rng.permutation(len(allf))
            agg = dict(tp=0, fp=0, fn=0); pos = 0
            for r in te[n]:
                eo2 = []
                for o in r['ev_out']:
                    f = o[1].copy()
                    if len(f):
                        f[:, j] = allf[perm[pos:pos + len(f)], j]
                    pos += len(f)
                    eo2.append((o[0], f, o[2]))
                r2 = dict(r); r2['ev_out'] = eo2
                o = R.evaluate(r2, 'ev', clf_ev, thr_ev)[0]
                for k in agg: agg[k] += o[k]
            drops.append(base_ev[n]['f1'] - R.prf(agg)[2])
        imp[n][fname] = float(np.mean(drops))
    log('E8', n)
RES['permutation_importance'] = imp; save()

# E9 shifted sensor assumptions at 0.01 lux (classifier trained with the default sensor)
name = 'Quarter moon (0.01 lux)'
shift = {}
for label, (attr, val) in {'default': (None, None), 'contrast threshold 0.15': ('C_TH', 0.15), 'contrast threshold 0.35': ('C_TH', 0.35),
                           'photon rate x0.5': ('K_PHOT', 0.5 * S.K_PHOT), 'photon rate x2': ('K_PHOT', 2 * S.K_PHOT)}.items():
    old = getattr(S, attr) if attr else None
    if attr: setattr(S, attr, val)
    raws = [build_raw(s, LUX[name]) for s in R.TEST_SEEDS]
    if attr: setattr(S, attr, old)
    recs = [make_rec(x) for x in raws]
    ev_ = score(clf_ev, thr_ev, {name: recs}, 'ev')[name]
    fr_ = score(clf_fr, thr_fr, {name: recs}, 'fr')[name]
    shift[label] = dict(event=ev_, frame=fr_, events_per_s=float(np.mean([r['n_events'] / R.DUR for r in recs])))
    log('E9', label, round(ev_['f1'], 3), round(fr_['f1'], 3))
RES['sensor_shift'] = shift; save()
RES['runtime_s'] = time.time() - T0; save(); log('ALL DONE')

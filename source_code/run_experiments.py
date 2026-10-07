"""
Runs the full synthetic experiment: simulate scenes, train classifiers on the
training scenes, evaluate event pipeline vs frame-camera baseline on held-out
test scenes. Writes results/*.json and results/*.npz used by make_figures.py.
"""
import json, os, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import simulator as S
import pipeline as P

OUT = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT, exist_ok=True)
LUX = {'Full moon (0.3 lux)': 0.3, 'Quarter moon (0.01 lux)': 0.01, 'Starlight (0.003 lux)': 0.003}
TRAIN_SEEDS = list(range(100, 106))
TEST_SEEDS = list(range(200, 206))
DUR = 2.5
MATCH_PX = 10.0
WIN = int(1e6 / S.FPS)


def gt_window(scene, te, win):
    return S.ground_truth(scene, te - win // 2)


def label(cents, gt):
    if len(cents) == 0:
        return np.zeros(0, bool)
    if len(gt) == 0:
        return np.zeros(len(cents), bool)
    d = np.hypot(cents[:, None, 0] - gt[None, :, 0], cents[:, None, 1] - gt[None, :, 1])
    return d.min(1) <= 6.0


def match(cents, gt, tol=MATCH_PX):
    """Greedy one-to-one matching; returns tp, fp, fn and per-GT hit mask."""
    hit = np.zeros(len(gt), bool)
    if len(cents) == 0:
        return 0, 0, len(gt), hit
    if len(gt) == 0:
        return 0, len(cents), 0, hit
    d = np.hypot(cents[:, None, 0] - gt[None, :, 0], cents[:, None, 1] - gt[None, :, 1])
    tp = 0
    used_c = set()
    for flat in np.argsort(d, axis=None):
        i, j = np.unravel_index(flat, d.shape)
        if d[i, j] > tol:
            break
        if i in used_c or hit[j]:
            continue
        used_c.add(i); hit[j] = True; tp += 1
    return tp, len(cents) - tp, len(gt) - tp, hit


def nms(c, p, r=8.0):
    order = np.argsort(-p); keep = []
    for i in order:
        if all(np.hypot(*(c[i] - c[k])) > r for k in keep):
            keep.append(i)
    return c[keep], p[keep]


def process(seed, lux, timing=None, keep_raw=False):
    sc = S.make_scene(seed, duration_s=DUR)
    ev, frames, ftimes = S.simulate(sc, lux, seed=seed)
    eo = P.run_event_pipeline(ev, ftimes, WIN, S.H, S.W, timing=timing)
    fo = P.run_frame_pipeline(frames)
    rec = dict(scene=sc, ftimes=ftimes, ev_out=eo, fr_out=fo, n_events=len(ev), dur=DUR)
    if keep_raw:
        rec['events'] = ev; rec['frames'] = frames
    return rec


def collect(rec, which):
    X, Y = [], []
    for k, te in enumerate(rec['ftimes']):
        gt, _ = gt_window(rec['scene'], te, WIN)
        f, c = (rec['ev_out'][k][1:] if which == 'ev' else rec['fr_out'][k])
        X.append(f); Y.append(label(c, gt))
    return np.concatenate(X), np.concatenate(Y)


def evaluate(rec, which, clf, thr, feat_idx=None):
    tp = fp = fn = 0
    sp_hit, sp_all = [], []
    dets = []
    for k, te in enumerate(rec['ftimes']):
        gt, spd = gt_window(rec['scene'], te, WIN)
        f, c = (rec['ev_out'][k][1:] if which == 'ev' else rec['fr_out'][k])
        if len(f):
            ff = f if feat_idx is None else f[:, feat_idx]
            p = clf.predict_proba(ff)[:, 1]
            sel = p >= thr
            c2, p2 = nms(c[sel], p[sel])
        else:
            c2 = np.zeros((0, 2))
        a, b, d, hit = match(c2, gt)
        tp += a; fp += b; fn += d
        sp_hit += list(hit); sp_all += list(spd)
        dets.append((te, c2))
    return dict(tp=tp, fp=fp, fn=fn), np.array(sp_hit), np.array(sp_all), dets


def prf(r):
    p = r['tp'] / max(r['tp'] + r['fp'], 1); q = r['tp'] / max(r['tp'] + r['fn'], 1)
    return p, q, 2 * p * q / max(p + q, 1e-9)


def fit(X, Y):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    clf.fit(X, Y)
    return clf


def main():
    t0 = time.time()
    train, test = [], {}
    for name, lux in LUX.items():
        for s in TRAIN_SEEDS:
            train.append(process(s, lux)); print('train', name, s, round(time.time() - t0), flush=True)
    timings = {}
    for name, lux in LUX.items():
        test[name] = []
        for s in TEST_SEEDS:
            tm = {}
            test[name].append(process(s, lux, timing=tm, keep_raw=(s == TEST_SEEDS[0])))
            timings.setdefault(name, []).append(tm)
            print('test', name, s, round(time.time() - t0), flush=True)

    res = {'setup': dict(lux=LUX, train_seeds=TRAIN_SEEDS, test_seeds=TEST_SEEDS, duration_s=DUR,
                         window_ms=WIN / 1000, match_px=MATCH_PX, sensor=[S.W, S.H])}
    clfs = {}
    for which in ('ev', 'fr'):
        X, Y = zip(*[collect(r, which) for r in train]); X = np.concatenate(X); Y = np.concatenate(Y)
        clf = fit(X, Y)
        # pick the decision threshold on the TRAINING scenes only
        best = (0, 0.5)
        for thr in np.arange(0.2, 0.85, 0.05):
            agg = dict(tp=0, fp=0, fn=0)
            for r in train:
                o = evaluate(r, which, clf, thr)[0]
                for k in agg: agg[k] += o[k]
            best = max(best, (prf(agg)[2], float(thr)))
        clfs[which] = (clf, best[1])
        res[f'{which}_train'] = dict(n_samples=int(len(Y)), n_pos=int(Y.sum()), thr=best[1], train_f1=best[0])

    speed_bins = [40, 100, 160, 260]
    res['test'] = {}
    tracks_example = None
    for name in LUX:
        row = {}
        for which in ('ev', 'fr'):
            clf, thr = clfs[which]
            agg = dict(tp=0, fp=0, fn=0); per_scene = []
            H_, A_ = [], []
            for i, r in enumerate(test[name]):
                o, h, a, dets = evaluate(r, which, clf, thr)
                per_scene.append(prf(o)[2])
                for k in agg: agg[k] += o[k]
                H_.append(h); A_.append(a)
                if which == 'ev' and i == 0 and name == list(LUX)[0]:
                    tracks_example = P.track(dets)
            p, q, f1 = prf(agg)
            H_ = np.concatenate(H_); A_ = np.concatenate(A_)
            bins = []
            for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
                m = (A_ >= lo) & (A_ < hi)
                bins.append(dict(lo=lo, hi=hi, recall=float(H_[m].mean()) if m.any() else None, n=int(m.sum())))
            row[which] = dict(**agg, precision=p, recall=q, f1=f1, per_scene_f1=per_scene, recall_by_speed=bins)
        # data volume
        nev = np.mean([r['n_events'] / r['dur'] for r in test[name]])
        kept = np.mean([t['n_kept'] / t['n_in'] for t in timings[name]])
        row['events_per_s'] = float(nev); row['baf_keep_frac'] = float(kept)
        row['event_bytes_per_s'] = float(nev * 4)          # EVT2-style 32-bit words
        row['frame_bytes_per_s'] = float(S.W * S.H * S.FPS)
        pw = np.concatenate([t['per_window_s'] for t in timings[name]])
        bt = np.mean([t['baf_total_s'] / len(t['per_window_s']) for t in timings[name]])
        row['latency_ms'] = dict(mean=float(1000 * (pw.mean() + bt)), p95=float(1000 * (np.percentile(pw, 95) + bt)))
        res['test'][name] = row
        print(name, {w: round(row[w]['f1'], 3) for w in ('ev', 'fr')}, flush=True)

    # ablation on the event side: drop feature groups / BAF (retrain each time)
    abl = {}
    groups = {'all features': None,
              'without motion features (speed, fit_r2, dsi)': [i for i, f in enumerate(P.FEATURES) if f not in ('speed', 'fit_r2', 'dsi')],
              'without T4/T5 detector (dsi)': [i for i, f in enumerate(P.FEATURES) if f != 'dsi']}
    for gname, idx in groups.items():
        X, Y = zip(*[collect(r, 'ev') for r in train]); X = np.concatenate(X); Y = np.concatenate(Y)
        clf = fit(X if idx is None else X[:, idx], Y)
        abl[gname] = {}
        for name in LUX:
            agg = dict(tp=0, fp=0, fn=0)
            for r in test[name]:
                o = evaluate(r, 'ev', clf, clfs['ev'][1], feat_idx=idx)[0]
                for k in agg: agg[k] += o[k]
            abl[gname][name] = prf(agg)[2]
    res['ablation_f1'] = abl
    res['runtime_s'] = time.time() - t0
    with open(os.path.join(OUT, 'results.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    # raw material for figures (one test scene per light level)
    ex = {}
    for name in LUX:
        r = test[name][0]
        ex[name] = dict(events=r['events'], frames=r['frames'], ftimes=r['ftimes'])
    np.savez_compressed(os.path.join(OUT, 'figure_data.npz'),
                        **{f'{k}__{n}': v[k] for n, v in enumerate(ex.values()) for k in ('events', 'frames', 'ftimes')},
                        paths=test[list(LUX)[0]][0]['scene']['paths'],
                        tracks=np.array([np.c_[np.full(len(t), i), t] for i, t in enumerate(tracks_example)], dtype=object) if tracks_example else np.array([]))
    print('done in', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()

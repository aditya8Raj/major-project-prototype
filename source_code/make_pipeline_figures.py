"""
Figures that look inside the event pipeline (Chapter 6) on one simulated scene (test seed 200, 0.01 lux):
  fig_pipeline_example.png : events, filtered events, time surface, motion map, candidates, motion vectors
  fig_feature_hist.png     : distribution of four candidate features for moths and for other candidates
Also writes results/feature_stats.json with the medians used in the text.
Run:  python make_pipeline_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simulator as S
import pipeline as P
import run_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})
LUX = 0.01
WIN = R.WIN

# ---------------------------------------------------------------- one window in detail
sc = S.make_scene(200, duration_s=R.DUR)
ev, fr, ft = S.simulate(sc, LUX, seed=200)
keep = P.baf(ev['t'], ev['x'].astype(np.int64), ev['y'].astype(np.int64), S.H, S.W, P.BAF_DT, P.BAF_K)
evk = ev[keep]
best_k, best_n = 20, -1
for k in range(20, min(70, len(ft) - 1)):
    gt, _ = R.gt_window(sc, ft[k], WIN)
    if len(gt) > best_n:
        best_k, best_n = k, len(gt)
k = best_k
te = ft[k]
gt, _ = R.gt_window(sc, te, WIN)
j = np.searchsorted(evk['t'], te, side='right'); a = np.searchsorted(evk['t'], te - WIN, side='right')
last_on = np.full((S.H, S.W), -10**12, np.int64); last_off = last_on.copy()
c = evk[:j]; on = c['p'] == 1
last_on[c['y'][on], c['x'][on]] = c['t'][on]; last_off[c['y'][~on], c['x'][~on]] = c['t'][~on]
rx, ry, tsf = P.emd_maps(last_on, last_off, te)
win_raw = ev[(ev['t'] > te - WIN) & (ev['t'] <= te)]
win = evk[a:j]
feats, cents = P.candidates(win, rx, ry, tsf, S.H, S.W)


def evimg(e):
    img = np.full((S.H, S.W, 3), 255, np.uint8); o = e['p'] == 1
    img[e['y'][o], e['x'][o]] = (42, 120, 214); img[e['y'][~o], e['x'][~o]] = (235, 104, 52)
    return img


fig, ax = plt.subplots(2, 3, figsize=(7.4, 5.1))
ax[0, 0].imshow(evimg(win_raw)); ax[0, 0].set_title(f'(a) Raw events, 33 ms\n({len(win_raw):,} events)', fontsize=8.5)
ax[0, 1].imshow(evimg(win)); ax[0, 1].set_title(f'(b) After noise filter\n({len(win):,} events)', fontsize=8.5)
from scipy.ndimage import maximum_filter
ax[0, 2].imshow(maximum_filter(tsf, 3), cmap='magma', vmin=0, vmax=max(np.percentile(tsf, 99.8), 1e-3)); ax[0, 2].set_title('(c) Fast time surface\n(ON + OFF)', fontsize=8.5)
mag = np.hypot(rx, ry)
ax[1, 0].imshow(maximum_filter(mag, 3), cmap='viridis', vmin=0, vmax=max(np.percentile(mag, 99.8), 1e-6)); ax[1, 0].set_title('(d) T4/T5-inspired\nmotion map |R|', fontsize=8.5)
ax[1, 1].imshow(evimg(win))
for f_, p_ in zip(feats, cents):
    ax[1, 1].add_patch(plt.Rectangle((p_[0] - f_[2] / 2, p_[1] - f_[3] / 2), f_[2], f_[3], fill=False, ec='#444', lw=0.6))
ax[1, 1].scatter(gt[:, 0], gt[:, 1], s=60, facecolors='none', edgecolors=AQUA, lw=1.4)
ax[1, 1].set_title(f'(e) Candidates (grey boxes)\nand true moths (green rings)', fontsize=8.5)
# zoom on the moth with most events nearby
if len(gt):
    d = [np.sum(np.hypot(win['x'] - g[0], win['y'] - g[1]) < 12) for g in gt]
    g = gt[int(np.argmax(d))]
else:
    g = np.array([160, 200])
x0, x1 = int(max(g[0] - 20, 0)), int(min(g[0] + 20, S.W)); y0, y1 = int(max(g[1] - 20, 0)), int(min(g[1] + 20, S.H))
ax[1, 2].imshow(evimg(win)[y0:y1, x0:x1], extent=(x0, x1, y1, y0), interpolation='nearest')
RX, RY = rx[y0:y1, x0:x1].sum(), ry[y0:y1, x0:x1].sum(); nrm = max(np.hypot(RX, RY), 1e-9)
ax[1, 2].annotate('', xy=(g[0] + 9 * RX / nrm, g[1] + 9 * RY / nrm), xytext=(g[0], g[1]), arrowprops=dict(arrowstyle='-|>', color='k', lw=1.8))
ax[1, 2].scatter([g[0]], [g[1]], s=140, facecolors='none', edgecolors=AQUA, lw=1.4)
ax[1, 2].set_title('(f) Zoom on one moth: events and\nnet T4/T5 motion direction (arrow)', fontsize=8.5)
for a_ in ax.ravel():
    a_.set_xticks([]); a_.set_yticks([])
    for s_ in a_.spines.values(): s_.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig_pipeline_example.png'), dpi=200); plt.close()
print('example window', k, 'events', len(win_raw), '->', len(win), 'candidates', len(feats), 'moths', len(gt))

# ---------------------------------------------------------------- feature distributions (six test scenes, 0.01 lux)
X, Y = [], []
for s in R.TEST_SEEDS:
    scn = S.make_scene(s, duration_s=R.DUR) if s != 200 else sc
    if s != 200:
        e_, f_, t_ = S.simulate(scn, LUX, seed=s)
    else:
        e_, f_, t_ = ev, fr, ft
    rec = dict(scene=scn, ftimes=t_, ev_out=P.run_event_pipeline(e_, t_, WIN, S.H, S.W), fr_out=None)
    x, y = R.collect(rec, 'ev'); X.append(x); Y.append(y)
X = np.concatenate(X); Y = np.concatenate(Y).astype(bool)
names = [('n_ev', 'Number of events', True), ('speed', 'Speed (px/s)', False), ('fit_r2', 'Straight-line fit (fit_r2)', False), ('dsi', 'Direction index (dsi)', False)]
fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.4))
stats = {'n_candidates': int(len(Y)), 'n_moth': int(Y.sum())}
for a_, (nm, lab, logx) in zip(ax, names):
    i = P.FEATURES.index(nm)
    v = X[:, i]; lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    bins = np.geomspace(max(lo, 1), max(hi, 2), 25) if logx else np.linspace(lo, hi, 25)
    a_.hist(v[~Y], bins=bins, density=True, color=ORANGE, alpha=0.65, label='other')
    a_.hist(v[Y], bins=bins, density=True, color=BLUE, alpha=0.65, label='moth')
    if logx: a_.set_xscale('log')
    a_.set_xlabel(lab, fontsize=8); a_.set_yticks([]); a_.tick_params(labelsize=7)
    stats[nm] = dict(median_moth=float(np.median(v[Y])), median_other=float(np.median(v[~Y])))
ax[0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_feature_hist.png'), dpi=200); plt.close()
json.dump(stats, open(os.path.join(HERE, 'results', 'feature_stats.json'), 'w'), indent=1)
print(stats)

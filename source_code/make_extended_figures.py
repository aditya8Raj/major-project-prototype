"""Figures of Chapter 7 from results/extended_results.json.   Run: python make_extended_figures.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#888888'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555'})
r = json.load(open(os.path.join(HERE, 'results', 'extended_results.json')))
NAMES = list(r['threshold_sweep']['ev'])
SHORT = [n.split(' (')[0] + '\n(' + n.split('(')[1] for n in NAMES]

# Fig 7.1 precision-recall curves
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
for a, n, s in zip(ax, NAMES, SHORT):
    for w, col, lab in (('ev', BLUE, 'Event pipeline'), ('fr', ORANGE, 'Frame baseline')):
        rows = r['threshold_sweep'][w][n]
        a.plot([x['recall'] for x in rows], [x['precision'] for x in rows], '-o', color=col, ms=2.5, lw=1.4, label=lab)
        thr = r['chosen_thresholds'][w]
        k = int(np.argmin([abs(x['thr'] - thr) for x in rows]))
        a.plot(rows[k]['recall'], rows[k]['precision'], 'o', ms=7, mfc='none', mec=col, mew=1.4)
    a.set_title(s, fontsize=8.5); a.set_xlabel('Recall'); a.set_xlim(0.5, 1.0); a.set_ylim(0.6, 1.02); a.grid(alpha=0.25)
ax[0].set_ylabel('Precision'); ax[0].legend(fontsize=7, frameon=False, loc='lower left')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_pr_curves.png'), dpi=200); plt.close()

# Fig 7.2 noise-filter settings
nf = r['noise_filter']; labs = list(nf)
fig, ax = plt.subplots(figsize=(7.4, 2.9))
wd = 0.26
for i, (n, col) in enumerate(zip(NAMES, (BLUE, '#5b9be6', '#a9c8f2'))):
    ax.bar(np.arange(len(labs)) + (i - 1) * wd, [nf[l][n]['f1'] for l in labs], wd, color=col, label=SHORT[i].replace('\n', ' '))
ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels([l.replace(' (used)', '\n(used)').replace(', ', '\n') for l in labs], fontsize=7.5)
ax.set_ylabel('F1 of event pipeline'); ax.set_ylim(0, 1.0); ax.legend(fontsize=7, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0)); ax.grid(axis='y', alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_filter_sensitivity.png'), dpi=200); plt.close()

# Fig 7.3 permutation importance
imp = r['permutation_importance']
feats = list(next(iter(imp.values())))
fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
order = np.argsort([-np.mean([imp[n][f] for n in NAMES]) for f in feats])
for a, n, s in zip(ax, NAMES, SHORT):
    vals = [imp[n][feats[i]] for i in order]
    a.barh(range(len(feats)), vals, color=BLUE)
    a.set_yticks(range(len(feats))); a.set_yticklabels([feats[i] for i in order], fontsize=7.5)
    a.invert_yaxis(); a.set_title(s, fontsize=8.5); a.set_xlabel('Drop in F1'); a.axvline(0, color='#555', lw=0.6); a.grid(axis='x', alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_importance.png'), dpi=200); plt.close()

# Fig 7.4 shifted sensor assumptions
sh = r['sensor_shift']; labs = list(sh)
fig, ax = plt.subplots(figsize=(7.4, 2.9))
wd = 0.36
ax.bar(np.arange(len(labs)) - wd / 2, [sh[l]['event']['f1'] for l in labs], wd, color=BLUE, label='Event pipeline')
ax.bar(np.arange(len(labs)) + wd / 2, [sh[l]['frame']['f1'] for l in labs], wd, color=ORANGE, label='Frame baseline')
ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels([l.replace(' ', '\n', 1) if len(l) > 9 else l for l in labs], fontsize=7.5)
ax.set_ylabel('F1 at 0.01 lux'); ax.set_ylim(0, 1.0); ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0)); ax.grid(axis='y', alpha=0.25)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_sensor_shift.png'), dpi=200); plt.close()
print('figures ok')

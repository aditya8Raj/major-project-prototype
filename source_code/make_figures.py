"""Builds the figures used in the report and slides from results/*.json|npz."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simulator as S

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.labelcolor': '#222', 'xtick.color': '#444', 'ytick.color': '#444'})
r = json.load(open(os.path.join(RES, 'results.json')))
D = np.load(os.path.join(RES, 'figure_data.npz'), allow_pickle=True)
names = list(r['test'].keys())
short = [n.split(' (')[0] + '\n(' + n.split('(')[1] for n in names]


def event_frame(ev, t0, t1):
    m = (ev['t'] >= t0) & (ev['t'] < t1)
    img = np.full((S.H, S.W, 3), 255, np.uint8)
    e = ev[m]
    on = e['p'] == 1
    img[e['y'][on], e['x'][on]] = (42, 120, 214)
    img[e['y'][~on], e['x'][~on]] = (235, 104, 52)
    return img


# Fig A: what the two cameras see at three light levels
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.9))
for i, n in enumerate(names):
    fr = D[f'frames__{i}']; ft = D[f'ftimes__{i}']; ev = D[f'events__{i}']
    k = 20
    ax[0, i].imshow(fr[k], cmap='gray', vmin=0, vmax=255)
    ax[0, i].set_title(short[i], fontsize=9)
    ax[1, i].imshow(event_frame(ev, ft[k] - 33000, ft[k]))
    for a in ax[:, i]:
        a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel('Frame camera\n(30 fps)'); ax[1, 0].set_ylabel('Event camera\n(33 ms of events)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_sensor_views.png'), dpi=200); plt.close(fig)

# Fig B: F1 by light level
fig, ax = plt.subplots(figsize=(5.4, 3.3))
w = 0.36; x = np.arange(len(names))
for j, (key, lab, col) in enumerate([('fr', 'Frame-camera baseline', ORANGE), ('ev', 'Event pipeline (this work)', BLUE)]):
    vals = [r['test'][n][key]['f1'] for n in names]
    sd = [np.std(r['test'][n][key]['per_scene_f1']) for n in names]
    b = ax.bar(x + (j - 0.5) * w, vals, w - 0.03, color=col, label=lab, yerr=sd, capsize=3, error_kw=dict(lw=0.8, color='#444'))
    for xi, v, e in zip(x + (j - 0.5) * w, vals, sd):
        ax.text(xi, v + e + 0.025, f'{v:.2f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8); ax.set_ylabel('F1 on held-out synthetic scenes')
ax.set_ylim(0, 1.2); ax.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_f1_by_light.png'), dpi=200); plt.close(fig)

# Fig C: data rate + latency
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
fb = r['test'][names[0]]['frame_bytes_per_s'] / 1e6
ev_rate = [r['test'][n]['event_bytes_per_s'] / 1e6 for n in names]
ax[0].bar(x - 0.18, [fb] * 3, 0.34, color=ORANGE, label='Frame camera (raw)')
ax[0].bar(x + 0.18, ev_rate, 0.34, color=BLUE, label='Event camera (raw events)')
ax[0].set_xticks(x); ax[0].set_xticklabels(short, fontsize=7); ax[0].set_ylabel('Data rate (MB/s)'); ax[0].legend(frameon=False, fontsize=7)
lat = [r['test'][n]['latency_ms']['mean'] for n in names]; p95 = [r['test'][n]['latency_ms']['p95'] for n in names]
ax[1].bar(x - 0.18, lat, 0.34, color=BLUE, label='mean'); ax[1].bar(x + 0.18, p95, 0.34, color=AQUA, label='95th percentile')
ax[1].set_xticks(x); ax[1].set_xticklabels(short, fontsize=7); ax[1].set_ylabel('Software time per window (ms)'); ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_rate_latency.png'), dpi=200); plt.close(fig)

# Fig D: tracks found in one full-moon test scene
paths = D['paths']
tr = D['tracks']
fig, ax = plt.subplots(figsize=(3.6, 3.6))
for m in range(paths.shape[0]):
    ax.plot(paths[m, ::10, 0], paths[m, ::10, 1], color='#bbb', lw=2.5, label='ground truth' if m == 0 else None)
for i, t in enumerate(tr):
    t = np.asarray(t, float)
    ax.plot(t[:, 2], t[:, 3], '-', color=BLUE, lw=1.1, label='pipeline track' if i == 0 else None)
ax.set_xlim(0, S.W); ax.set_ylim(S.H, 0); ax.set_aspect('equal'); ax.legend(frameon=False, fontsize=7, loc='upper right')
ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_tracks.png'), dpi=200); plt.close(fig)
print('figures ok')

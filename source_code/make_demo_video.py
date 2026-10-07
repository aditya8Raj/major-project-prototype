"""
Renders the demo video of the software prototype (synthetic scene).
Three panels per frame: frame camera | event camera | pipeline output (detections, tracks, confidence).
Usage:  python make_demo_video.py            (writes ../demo/demo_software_prototype.mp4)
"""
import os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simulator as S, pipeline as P
import run_experiments as R

OUT = os.path.join(os.path.dirname(__file__), '..', 'demo'); os.makedirs(OUT, exist_ok=True)
DEMO_SEED, DUR_S = 421, 3.0
SEGMENTS = [('Full moon (0.3 lux)', 0.3), ('Starlight (0.003 lux)', 0.003)]
BLUE, ORANGE = (42, 120, 214), (235, 104, 52)

# classifiers trained on a few scenes that are not the demo scene
R.DUR = 2.0
train = [R.process(s, lux) for s in (301, 302, 303) for lux in (0.3, 0.01, 0.003)]
clfs = {}
for which in ('ev', 'fr'):
    X, Y = zip(*[R.collect(r, which) for r in train])
    clfs[which] = R.fit(np.concatenate(X), np.concatenate(Y))
THR = {'ev': 0.5, 'fr': 0.5}

def ev_img(ev, t0, t1):
    m = (ev['t'] >= t0) & (ev['t'] < t1)
    img = np.full((S.H, S.W, 3), 255, np.uint8); e = ev[m]; on = e['p'] == 1
    img[e['y'][on], e['x'][on]] = BLUE; img[e['y'][~on], e['x'][~on]] = ORANGE
    return img

fig, ax = plt.subplots(1, 3, figsize=(10.8, 4.2), dpi=100)
tmp = os.path.join(OUT, '_frames'); os.makedirs(tmp, exist_ok=True)
n = 0
for label, lux in SEGMENTS:
    R.DUR = DUR_S
    sc = S.make_scene(DEMO_SEED, duration_s=DUR_S)
    ev, frames, ft = S.simulate(sc, lux, seed=DEMO_SEED)
    eo = P.run_event_pipeline(ev, ft, R.WIN, S.H, S.W)
    fo = P.run_frame_pipeline(frames)
    dets, hist = [], []
    for k, te in enumerate(ft):
        out = []
        for which, (f, c) in (('ev', eo[k][1:]), ('fr', fo[k])):
            if len(f):
                p = clfs[which].predict_proba(f)[:, 1]; s = p >= THR[which]
                c2, p2 = R.nms(c[s], p[s]) if s.any() else (np.zeros((0, 2)), np.zeros(0))
            else:
                c2, p2 = np.zeros((0, 2)), np.zeros(0)
            out.append((c2, p2))
        dets.append((te, out[0][0])); hist.append(out)
    tracks_so_far = lambda k: P.track(dets[:k + 1], min_len=3)
    for k, te in enumerate(ft):
        for a in ax: a.clear(); a.set_xticks([]); a.set_yticks([])
        ax[0].imshow(frames[k], cmap='gray', vmin=0, vmax=255); ax[0].set_title('Frame camera (30 fps)', fontsize=10)
        c, p = hist[k][1]
        for (x, y), q in zip(c, p): ax[0].add_patch(plt.Circle((x, y), 9, fill=False, ec='#eb6834', lw=1.2))
        ax[1].imshow(ev_img(ev, te - 33000, te)); ax[1].set_title('Event camera (last 33 ms)', fontsize=10)
        ax[2].imshow(np.full((S.H, S.W, 3), 245, np.uint8)); ax[2].set_title('Event pipeline output', fontsize=10)
        gt, _ = S.ground_truth(sc, te)
        ax[2].scatter(gt[:, 0], gt[:, 1], s=60, facecolors='none', edgecolors='#999', linewidths=1, label='true moth position')
        c, p = hist[k][0]
        for tr in tracks_so_far(k):
            ax[2].plot(tr[-25:, 1], tr[-25:, 2], '-', color='#2a78d6', lw=1.2)
        for (x, y), q in zip(c, p):
            ax[2].add_patch(plt.Rectangle((x - 8, y - 8), 16, 16, fill=False, ec='#1baf7a', lw=1.6))
            ax[2].text(x + 10, y - 10, f'{q:.2f}', color='#1b6b4c', fontsize=8)
        ax[2].set_xlim(0, S.W); ax[2].set_ylim(S.H, 0)
        fig.suptitle(f'Simulated night scene: {label}   |   t = {te/1e6:.2f} s   |   SIMULATION, not field data', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(tmp, f'f{n:05d}.png')); n += 1
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '10', '-i', os.path.join(tmp, 'f%05d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(OUT, 'demo_software_prototype.mp4')], check=True)
for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
os.rmdir(tmp); print('video ok', n, 'frames')

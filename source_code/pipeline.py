"""
Event-based detection pipeline (software prototype of the proposed system).

  1. Background-activity filter (BAF)        - removes isolated noise events
  2. Time surfaces, ON and OFF channels       - fast (5 ms) and slow (30 ms) decay
  3. T4/T5-inspired motion detector           - Hassenstein-Reichardt correlator on
                                                the ON (T4) and OFF (T5) time surfaces
  4. Spatial event graph -> connected parts   - groups events into candidate objects
  5. Feature vector + classifier              - moth vs clutter, gives a confidence
  6. Nearest-neighbour tracker                - links detections into flight tracks

The full design replaces step 4-5 with a graph neural network and runs on an
FPGA; this prototype uses a light-weight graph grouping + gradient-boosted trees
so it can be trained on a laptop without a GPU.
"""
import numpy as np
from numba import njit
from scipy import ndimage

TAU_FAST = 5_000      # us
TAU_SLOW = 30_000     # us
BAF_DT = 1_000       # us (chosen on training scenes, see results_baf_tuning.txt)
BAF_K = 2           # supporting neighbours required
EMD_SHIFT = 2         # px
FEATURES = ['n_ev', 'n_px', 'w', 'h', 'fill', 'on_frac', 'ev_per_px', 't_std_ms',
            'speed', 'fit_r2', 'dsi', 'ts_mean']


@njit(cache=True)
def baf(t, x, y, H, W, dt, k):
    """Background-activity filter (Delbruck 2008): keep an event only if at least
    k of its 8 neighbours fired within dt."""
    last = np.full((H + 2, W + 2), -10**12, np.int64)
    keep = np.zeros(t.shape[0], np.bool_)
    for i in range(t.shape[0]):
        xi = x[i] + 1; yi = y[i] + 1; ti = t[i]
        n = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                if ti - last[yi + dy, xi + dx] <= dt:
                    n += 1
        keep[i] = n >= k
        last[yi, xi] = ti
    return keep


def emd_maps(last_on, last_off, t_now):
    """T4 (ON) / T5 (OFF) Hassenstein-Reichardt correlator on time surfaces.
    Returns Rx, Ry (signed motion maps) and the fast time surface."""
    rx = np.zeros(last_on.shape); ry = np.zeros(last_on.shape); tsf_all = np.zeros(last_on.shape)
    s = EMD_SHIFT
    for last in (last_on, last_off):
        age = (t_now - last).astype(np.float64)
        tsf = np.exp(-age / TAU_FAST)
        tss = np.exp(-age / TAU_SLOW)
        tsf_all += tsf
        # delayed arm (slow surface) x direct arm (fast surface) at the neighbour
        rx[:, :-s] += tss[:, :-s] * tsf[:, s:] - tsf[:, :-s] * tss[:, s:]
        ry[:-s, :] += tss[:-s, :] * tsf[s:, :] - tsf[:-s, :] * tss[s:, :]
    return rx, ry, tsf_all


STRUCT = np.ones((5, 5), bool)   # radius-2 neighbourhood = edges of the event graph


def candidates(ev, rx, ry, tsf, H, W):
    """Group window events into objects and compute one feature row per object."""
    if len(ev) == 0:
        return np.zeros((0, len(FEATURES))), np.zeros((0, 2))
    occ = np.zeros((H, W), bool)
    occ[ev['y'], ev['x']] = True
    # radius-2 graph edges: dilate by 1 px, label 8-connected, keep only event pixels
    lab, n = ndimage.label(ndimage.binary_dilation(occ, np.ones((3, 3), bool)), structure=np.ones((3, 3), bool))
    lab = lab * occ
    if n == 0:
        return np.zeros((0, len(FEATURES))), np.zeros((0, 2))
    c = lab[ev['y'], ev['x']]
    m = n + 1
    x = ev['x'].astype(float); y = ev['y'].astype(float)
    t = (ev['t'] - ev['t'].min()).astype(float) / 1000.0  # ms
    cnt = np.bincount(c, minlength=m).astype(float)
    sx = np.bincount(c, x, m); sy = np.bincount(c, y, m); st = np.bincount(c, t, m)
    mx, my, mt = sx / np.maximum(cnt, 1), sy / np.maximum(cnt, 1), st / np.maximum(cnt, 1)
    dt_ = t - mt[c]; dx_ = x - mx[c]; dy_ = y - my[c]
    vtt = np.bincount(c, dt_ * dt_, m); vtx = np.bincount(c, dt_ * dx_, m); vty = np.bincount(c, dt_ * dy_, m)
    vxx = np.bincount(c, dx_ * dx_, m); vyy = np.bincount(c, dy_ * dy_, m)
    vx = vtx / np.maximum(vtt, 1e-9); vy = vty / np.maximum(vtt, 1e-9)       # px / ms
    expl = (vx * vtx + vy * vty)
    r2 = expl / np.maximum(vxx + vyy, 1e-9)
    on = np.bincount(c, ev['p'].astype(float), m) / np.maximum(cnt, 1)
    objs = ndimage.find_objects(lab)
    npx = np.bincount(lab.ravel(), minlength=m).astype(float)
    feats, cents = [], []
    for k in range(1, m):
        sl = objs[k - 1]
        if cnt[k] < 5:
            continue
        h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
        Rx = rx[sl].sum(); Ry = ry[sl].sum(); E = tsf[sl].sum()
        dsi = np.hypot(Rx, Ry) / (E + 1e-6)
        feats.append([cnt[k], npx[k], w, h, npx[k] / (w * h), on[k], cnt[k] / npx[k],
                      np.sqrt(vtt[k] / cnt[k]), 1000 * np.hypot(vx[k], vy[k]), r2[k], dsi,
                      E / npx[k]])
        cents.append([mx[k], my[k]])
    return np.array(feats).reshape(-1, len(FEATURES)), np.array(cents).reshape(-1, 2)


def run_event_pipeline(events, eval_times, window_us, H, W, use_baf=True, timing=None):
    """Yields (t_eval, features, centroids) for every evaluation instant."""
    import time
    t0 = time.perf_counter()
    if use_baf:
        keep = baf(events['t'], events['x'].astype(np.int64), events['y'].astype(np.int64), H, W, BAF_DT, BAF_K)
        ev = events[keep]
    else:
        ev = events
    t_baf = time.perf_counter() - t0
    last_on = np.full((H, W), -10**12, np.int64); last_off = last_on.copy()
    idx = 0
    out = []
    per = []
    for te in eval_times:
        t1 = time.perf_counter()
        j = np.searchsorted(ev['t'], te, side='right')
        chunk = ev[idx:j]; idx = j
        on = chunk['p'] == 1
        last_on[chunk['y'][on], chunk['x'][on]] = chunk['t'][on]
        last_off[chunk['y'][~on], chunk['x'][~on]] = chunk['t'][~on]
        a = np.searchsorted(ev['t'], te - window_us, side='right')
        win = ev[a:j]
        rx, ry, tsf = emd_maps(last_on, last_off, te)
        f, c = candidates(win, rx, ry, tsf, H, W)
        per.append(time.perf_counter() - t1)
        out.append((te, f, c))
    if timing is not None:
        timing['baf_total_s'] = t_baf
        timing['per_window_s'] = per
        timing['n_in'] = len(events); timing['n_kept'] = len(ev)
    return out


# ----------------------------------------------------------------- frame baseline
FRAME_FEATURES = ['area', 'z_peak', 'z_mean', 'w', 'h', 'fill', 'bright']


def run_frame_pipeline(frames):
    """Classical night-camera baseline: running background, z-score, blobs."""
    f = frames.astype(np.float64)
    bg = f[0].copy(); var = np.full(bg.shape, 4.0)
    out = []
    for i in range(len(f)):
        d = ndimage.gaussian_filter(f[i] - bg, 1.0)
        z = d / np.sqrt(var + 1.0)
        fg = z > 2.0
        lab, n = ndimage.label(fg, structure=np.ones((3, 3), bool))
        feats, cents = [], []
        if n:
            objs = ndimage.find_objects(lab)
            idx = np.arange(1, n + 1)
            area = ndimage.sum(np.ones_like(z), lab, idx)
            zmax = ndimage.maximum(z, lab, idx); zmean = ndimage.mean(z, lab, idx)
            br = ndimage.mean(f[i], lab, idx)
            com = ndimage.center_of_mass(np.maximum(z, 0), lab, idx)
            for k in range(n):
                if area[k] < 2:
                    continue
                sl = objs[k]
                h = sl[0].stop - sl[0].start; w = sl[1].stop - sl[1].start
                feats.append([area[k], zmax[k], zmean[k], w, h, area[k] / (w * h), br[k]])
                cents.append([com[k][1], com[k][0]])
        out.append((np.array(feats).reshape(-1, len(FRAME_FEATURES)), np.array(cents).reshape(-1, 2)))
        # update background where nothing is detected
        alpha = 0.1 if i > 0 else 0.0
        upd = ~fg
        diff = f[i] - bg
        bg[upd] += alpha * diff[upd]
        var[upd] = (1 - alpha) * var[upd] + alpha * diff[upd] ** 2
    return out


# ----------------------------------------------------------------- tracking
def track(dets, gate=25.0, min_len=5):
    """Greedy nearest-neighbour linking. dets: list of (t, Nx2 positions)."""
    tracks, active = [], []
    for t, P in dets:
        used = set()
        new_active = []
        for tr in active:
            if len(P) == 0:
                continue
            d = np.hypot(*(P - tr[-1][1:]).T)
            for u in used:
                d[u] = np.inf
            j = int(np.argmin(d))
            if d[j] < gate:
                tr.append((t, *P[j])); used.add(j); new_active.append(tr)
            elif t - tr[-1][0] < 100_000:
                new_active.append(tr)
            else:
                tracks.append(tr)
        for j in range(len(P)):
            if j not in used:
                new_active.append([(t, *P[j])])
        active = new_active
    tracks += active
    return [np.array(tr) for tr in tracks if len(tr) >= min_len]

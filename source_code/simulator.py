"""
Synthetic night-field scene + event-camera (DVS) + frame-camera simulator.

Everything here is SYNTHETIC. It exists so the pipeline can be developed and
tested before real event-camera hardware and field data are available.

Scene: a 320x320 view (same size as the Prophesee GenX320 sensor) of a crop
canopy under the night sky. It contains
  * moths  : small bright blobs flying along random, turning paths, with a
             wing-beat flicker (the targets),
  * leaves : large blobs swaying slowly in the wind (clutter),
  * stars  : point sources that scintillate in place (flicker clutter).
Illumination is set in lux. Photon shot noise is modelled for both sensors,
so the scene gets noisier as the light drops.

Event camera: per-pixel photoreceptor low-pass (slower in the dark), log
intensity, ON/OFF threshold crossing with per-pixel threshold mismatch, plus
leak noise events. This follows the standard DVS pixel model used by ESIM and
v2e (Rebecq et al. 2018; Hu et al. 2021), simplified.

Frame camera: 30 fps, 33 ms exposure (motion blur), shot + read noise, 8-bit.
"""
import numpy as np
from numba import njit, prange

W = H = 320
DT_US = 1000            # simulation step: 1 ms
K_PHOT = 2.0e6          # photons / s / pixel per (lux x reflectance)  [assumed]
DARK = 50.0             # dark-current equivalent photons / s          [assumed]
C_TH = 0.25             # DVS contrast threshold (log units)            [assumed]
C_SIGMA = 0.03          # threshold mismatch between pixels
LEAK_HZ = 0.05          # leak/background events per pixel per second
FPS = 30
READ_NOISE = 3.0        # electrons rms, frame camera


def _smooth_noise(rng, shape, sigma):
    from scipy.ndimage import gaussian_filter
    n = gaussian_filter(rng.standard_normal(shape), sigma)
    return (n - n.min()) / (n.max() - n.min() + 1e-9)


def make_scene(seed, duration_s=3.0, n_moths=None):
    """Return a dict describing one random scene (all units: px, s)."""
    rng = np.random.default_rng(seed)
    horizon = int(H * rng.uniform(0.30, 0.45))
    base = np.full((H, W), 0.03)
    canopy = 0.06 + 0.16 * _smooth_noise(rng, (H, W), 3.0)
    base[horizon:] = canopy[horizon:]
    # leaves (clutter that really moves, but only by a few pixels)
    nl = 14
    leaves = np.zeros((nl, 7))
    leaves[:, 0] = rng.uniform(0, W, nl)                    # x0
    leaves[:, 1] = rng.uniform(horizon + 10, H, nl)         # y0
    leaves[:, 2] = rng.uniform(5, 11, nl)                   # sigma
    leaves[:, 3] = rng.uniform(0.06, 0.12, nl)              # reflectance add
    leaves[:, 4] = rng.uniform(1.5, 4.0, nl)                # sway amplitude px
    leaves[:, 5] = rng.uniform(0.3, 1.5, nl)                # sway freq Hz
    leaves[:, 6] = rng.uniform(0, 2 * np.pi, nl)            # phase
    # stars (flicker in place)
    ns = 18
    stars = np.zeros((ns, 4))
    stars[:, 0] = rng.integers(0, W, ns)
    stars[:, 1] = rng.integers(0, max(horizon - 5, 6), ns)
    stars[:, 2] = rng.uniform(0.2, 0.6, ns)                 # brightness
    stars[:, 3] = rng.integers(0, 1 << 30, ns)              # flicker seed
    # moths
    if n_moths is None:
        n_moths = int(rng.integers(2, 6))
    steps = int(duration_s * 1000)
    paths = np.zeros((n_moths, steps, 2))
    params = np.zeros((n_moths, 4))
    for m in range(n_moths):
        speed = rng.uniform(40, 260)                        # px / s
        heading = rng.uniform(0, 2 * np.pi)
        p = np.array([rng.uniform(30, W - 30), rng.uniform(30, H - 30)])
        turn = rng.uniform(1.0, 4.0)                        # rad/s noise
        for k in range(steps):
            heading += turn * np.sqrt(1e-3) * rng.standard_normal()
            # soft turn-back near the borders keeps moths in view most of the time
            c = np.array([W / 2, H / 2]) - p
            if np.any(p < 15) or np.any(p > W - 15):
                heading = np.arctan2(c[1], c[0]) + rng.uniform(-0.6, 0.6)
            p = p + speed * 1e-3 * np.array([np.cos(heading), np.sin(heading)])
            paths[m, k] = p
        params[m] = [rng.uniform(1.2, 2.2),                 # sigma px
                     rng.uniform(0.35, 0.6),                # reflectance
                     rng.uniform(25, 45),                   # wing-beat Hz
                     speed]
    return dict(seed=seed, base=base, leaves=leaves, stars=stars,
                paths=paths, moth_params=params, steps=steps, horizon=horizon)


@njit(cache=True)
def _blob(img, cx, cy, sig, amp):
    r = int(4 * sig) + 1
    x0 = max(int(cx) - r, 0); x1 = min(int(cx) + r + 1, img.shape[1])
    y0 = max(int(cy) - r, 0); y1 = min(int(cy) + r + 1, img.shape[0])
    inv = 1.0 / (2 * sig * sig)
    for y in range(y0, y1):
        for x in range(x0, x1):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            img[y, x] += amp * np.exp(-d2 * inv)


@njit(cache=True)
def render(base, leaves, stars, paths, mparams, k):
    """Reflectance image at step k (1 ms steps)."""
    t = k * 1e-3
    img = base.copy()
    for i in range(leaves.shape[0]):
        dx = leaves[i, 4] * np.sin(2 * np.pi * leaves[i, 5] * t + leaves[i, 6])
        _blob(img, leaves[i, 0] + dx, leaves[i, 1] + 0.3 * dx, leaves[i, 2], leaves[i, 3])
    for i in range(stars.shape[0]):
        # scintillation: smooth pseudo-random brightness, changes every ~20 ms
        a = np.sin(t * 2 * np.pi * 3.1 + stars[i, 3] % 97) + 0.6 * np.sin(t * 2 * np.pi * 7.7 + stars[i, 3] % 31)
        b = stars[i, 2] * (1.0 + 0.35 * a)
        _blob(img, stars[i, 0], stars[i, 1], 0.7, b)
    for m in range(paths.shape[0]):
        x = paths[m, k, 0]; y = paths[m, k, 1]
        if x < -10 or x > img.shape[1] + 10 or y < -10 or y > img.shape[0] + 10:
            continue
        wing = 1.0 + 0.3 * np.sin(2 * np.pi * mparams[m, 2] * t)
        _blob(img, x, y, mparams[m, 0], (mparams[m, 1] - 0.1) * wing)
    return img


@njit(parallel=True, cache=True)
def _dvs_step(flux, state, ref, thr, tau_us, t_us, out_x, out_y, out_t, out_p, cnt, seed_off):
    h, w = flux.shape
    for y in prange(h):
        np.random.seed(seed_off + y)
        for x in range(w):
            phi = flux[y, x]
            n = np.random.poisson(phi * DT_US * 1e-6)
            inst = n / (DT_US * 1e-6)
            tau = tau_us[y, x]
            a = DT_US / (tau + DT_US)
            state[y, x] += a * (inst - state[y, x])
            L = np.log(state[y, x] + DARK)
            d = L - ref[y, x]
            th = thr[y, x]
            # leak noise
            if np.random.random() < LEAK_HZ * DT_US * 1e-6:
                d = th + 1e-6
            nev = 0
            while abs(d) >= th and nev < 4:
                pol = 1 if d > 0 else 0
                ref[y, x] += th if pol else -th
                d = L - ref[y, x]
                c = cnt[y]
                if c < out_x.shape[1]:
                    out_x[y, c] = x; out_y[y, c] = y
                    out_t[y, c] = t_us + int(np.random.random() * DT_US)
                    out_p[y, c] = pol
                    cnt[y] = c + 1
                nev += 1


def simulate(scene, lux, seed=0):
    """Simulate both sensors. Returns (events structured array, frames, frame_times_us)."""
    rng = np.random.default_rng(seed + 12345)
    thr = np.clip(C_TH + C_SIGMA * rng.standard_normal((H, W)), 0.12, None)
    steps = scene['steps']
    sc = (scene['base'], scene['leaves'], scene['stars'], scene['paths'], scene['moth_params'])
    flux0 = K_PHOT * lux * render(*sc, 0)
    state = flux0.copy()
    ref = np.log(state + DARK)
    # photoreceptor bandwidth drops in the dark (assumed model)
    tau_us = np.clip(3000.0 * np.sqrt(6.0e4 / np.maximum(flux0, 1.0)), 3000.0, 40000.0)
    cap = 64
    out_x = np.zeros((H, cap), np.int16); out_y = np.zeros((H, cap), np.int16)
    out_t = np.zeros((H, cap), np.int64); out_p = np.zeros((H, cap), np.int8)
    ev_chunks = []
    exp_steps = int(round(1000 / FPS))
    frames, ftimes = [], []
    acc = np.zeros((H, W))
    for k in range(steps):
        flux = K_PHOT * lux * render(*sc, k)
        cnt = np.zeros(H, np.int64)
        _dvs_step(flux, state, ref, thr, tau_us, k * DT_US, out_x, out_y, out_t, out_p, cnt,
                  (seed * 7919 + k * 331) % (2**31 - H))
        for y in np.nonzero(cnt)[0]:
            c = cnt[y]
            ev_chunks.append(np.stack([out_t[y, :c], out_x[y, :c], out_y[y, :c], out_p[y, :c]], 1))
        acc += flux
        if (k + 1) % exp_steps == 0:
            e = rng.poisson(acc * 1e-3) + READ_NOISE * rng.standard_normal((H, W))
            frames.append(e.astype(np.float32))
            ftimes.append((k + 1) * DT_US)
            acc[:] = 0
    ev = np.concatenate(ev_chunks) if ev_chunks else np.zeros((0, 4), np.int64)
    ev = ev[np.argsort(ev[:, 0], kind='stable')]
    events = np.zeros(len(ev), dtype=[('t', 'i8'), ('x', 'i2'), ('y', 'i2'), ('p', 'i1')])
    events['t'] = ev[:, 0]; events['x'] = ev[:, 1]; events['y'] = ev[:, 2]; events['p'] = ev[:, 3]
    # 8-bit output with a fixed gain per illumination level
    gain = 200.0 / max(np.percentile(frames[0], 99.5), 1.0)
    frames = [np.clip(np.round(f * gain), 0, 255).astype(np.uint8) for f in frames]
    return events, np.array(frames), np.array(ftimes)


def ground_truth(scene, t_us):
    """Moth positions (Nx2) visible at time t_us, plus their speeds."""
    k = min(int(t_us // DT_US), scene['steps'] - 1)
    p = scene['paths'][:, k]
    ok = (p[:, 0] >= 2) & (p[:, 0] < W - 2) & (p[:, 1] >= 2) & (p[:, 1] < H - 2)
    return p[ok], scene['moth_params'][ok, 3]

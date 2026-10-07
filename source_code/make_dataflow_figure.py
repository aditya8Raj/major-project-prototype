"""Figure 6.1 of the report: flow of data through the software prototype."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif']})
BLUE, ORANGE, GREY, AQUA = '#2a78d6', '#eb6834', '#555555', '#1baf7a'
fig, ax = plt.subplots(figsize=(7.4, 4.2)); ax.set_xlim(-1, 107); ax.set_ylim(0, 58); ax.axis('off')


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc='white', ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.0, linespacing=1.25)


def arrow(x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=9, color=color, lw=1.2))


W, H = 14.5, 11
xs = [20, 38, 56, 74, 92]
# row 1: event path (y = 40), row 2: frame path (y = 18)
labels1 = ['Event camera\nmodel\n(1 ms steps)', 'Noise filter\n(k = 2, 1 ms)', 'Time surfaces\nand T4/T5-\ninspired map', 'Event grouping\n12 features', 'Tree\nclassifier']
labels2 = ['Frame camera\nmodel\n(30 fps)', 'Background\nsubtraction', 'Blob grouping\n7 features', 'Tree\nclassifier']
for x, t in zip(xs, labels1): box(x, 40, W, H, t, BLUE)
for x, t in zip(xs, labels2): box(x, 18, W, H, t, ORANGE)
box(0, 24.5, 15, 16, 'Simulator\n(random scene,\n3 light levels,\ntrue moth\npositions)', GREY)
box(60, 0.5, 44, 10, 'Matching with truth (10 px):\nprecision, recall, F1; tracker;\ndata rate; software timing', AQUA)
for x in xs[:-1]:
    arrow(x + W + 0.6, 45.5, x + W + 3.0, 45.5, BLUE)
for x in xs[:3]:
    arrow(x + W + 0.6, 23.5, x + W + 3.0, 23.5, ORANGE)
arrow(15.6, 36, 19.4, 43.5, GREY); arrow(15.6, 29, 19.4, 21.5, GREY)
arrow(99.2, 39.4, 99.2, 11.0, AQUA); arrow(80, 17.4, 80, 11.0, AQUA)
ax.text(63, 56, 'Event pipeline', color=BLUE, fontsize=8.5, ha='center', weight='bold')
ax.text(45, 34, 'Frame baseline', color=ORANGE, fontsize=8.5, ha='center', weight='bold')
plt.savefig(os.path.join(OUT, 'fig_dataflow.png'), dpi=200, bbox_inches='tight', pad_inches=0.08)
print('ok')

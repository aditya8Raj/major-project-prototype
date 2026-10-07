"""Block diagram of the proposed system (Figure 3.1)."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')
plt.rcParams['font.family'] = ['DejaVu Serif']
fig, ax = plt.subplots(figsize=(7.6, 5.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis('off')

def box(x, y, w, h, text, fc, ec, fs=7.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', linespacing=1.25)

def arrow(a, b, label=None, rad=0.0, color='#333', ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10, lw=1.1, color=color, ls=ls, connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text((a[0] + b[0]) / 2 + 1.5, (a[1] + b[1]) / 2, label, ha='left', va='center', fontsize=6.8, color=color, style='italic')

ax.add_patch(FancyBboxPatch((1, 21), 98, 49, boxstyle='round,pad=0.3,rounding_size=1.5', fc='#f4f7fb', ec='#2a78d6', lw=1.4, ls='--'))
ax.text(3, 67.5, 'Solar-powered field node (all processing on the device, no cloud needed)', fontsize=7.8, color='#2a78d6', fontweight='bold', va='center')
W_, G_ = 19, 6
xs = [4 + i * (W_ + G_) for i in range(4)]
y1, y2, h = 45, 28, 14
box(xs[0], y1, W_, h, 'Stereo event\ncameras\n(left + right)', '#fff', '#2a78d6', bold=True)
box(xs[1], y1, W_, h, 'Noise filter\n(background-\nactivity filter)', '#fff', '#2a78d6')
box(xs[2], y1, W_, h, 'Time-surface\nencoder\n(ON / OFF)', '#fff', '#2a78d6')
box(xs[3], y1, W_, h, 'T4/T5-inspired\nmotion detector', '#fff', '#2a78d6')
box(xs[3], y2 - 3, W_, h - 3 + 3, 'Event graph\nbuilder', '#fff', '#1baf7a')
box(xs[2], y2 - 3, W_, h, 'Graph neural\nnetwork: pest vs\nnon-pest', '#fff', '#1baf7a', bold=True)
box(xs[1], y2 - 3, W_, h, 'Stereo matching\n+ 4D tracks\n+ confidence', '#fff', '#1baf7a')
box(xs[0], y2 - 3, W_, h, 'Alert logic\n(threshold on\nconfidence)', '#fff', '#1baf7a')
for i in range(3):
    arrow((xs[i] + W_ + 0.4, y1 + h / 2), (xs[i + 1] - 0.4, y1 + h / 2))
arrow((xs[3] + W_ / 2, y1 - 0.4), (xs[3] + W_ / 2, y2 - 3 + h + 0.4))
for i in (3, 2, 1):
    arrow((xs[i] - 0.4, y2 - 3 + h / 2), (xs[i - 1] + W_ + 0.4, y2 - 3 + h / 2))
ax.text(3, 63.6, 'Planned FPGA target: green stages run as a hardware pipeline, latency goal < 50 ms', fontsize=7, ha='left', color='#1b6b4c', style='italic')
box(4, 3, 30, 12, 'Farmer dashboard on phone\n(alert + confidence score,\nfarmer decides)', '#eef9f4', '#1baf7a', bold=True)
box(60, 3, 36, 12, 'Solar panel + battery +\npower manager\n(supplies every block)', '#fff6e5', '#eda100')
arrow((14, 24.6), (19, 15.4))
ax.text(21.5, 18.2, 'alert (LoRa / GSM, planned)', fontsize=6.8, color='#333', style='italic')
arrow((78, 15.4), (78, 20.2), None, color='#8a5a00', ls='--')
fig.savefig(os.path.join(OUT, 'fig_block_diagram.png'), dpi=220, bbox_inches='tight', facecolor='white')
print('ok')

from pathlib import Path
import csv
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'report/figures'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.4), sharey=True)
for ax, run, title in zip(axes, ['dcgan128', 'r1'], ['Baseline (100 epochs)', 'R1 (200 epochs)']):
    rows = list(csv.DictReader(open(ROOT / 'output' / run / 'losses.csv')))
    epochs = [int(r['epoch']) for r in rows]
    ax.plot(epochs, [float(r['loss_g']) for r in rows], color='#b65328', lw=1.1, label='Generator')
    ax.plot(epochs, [float(r['loss_d']) for r in rows], color='#28698b', lw=1.1, label='Discriminator')
    ax.set(title=title, xlabel='Epoch', ylim=(0, 13), xlim=(0, max(epochs)))
    ax.grid(alpha=.18, linewidth=.5)
axes[0].set_ylabel('BCE loss')
axes[1].legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(pad=.25, w_pad=1.5)
fig.savefig(FIG / 'losses.pdf', bbox_inches='tight')
plt.close(fig)

# First 16 samples in the saved 8x8 grid, rearranged into four columns.
source = Image.open(ROOT / 'output/test_r1_final/samples.png').convert('RGB')
assert source.size == (1042, 1042)
grid = Image.new('RGB', (526, 526), 'white')
for i in range(16):
    row, col = divmod(i, 8)
    tile = source.crop((2 + col*130, 2 + row*130, 130 + col*130, 130 + row*130))
    dest_row, dest_col = divmod(i, 4)
    grid.paste(tile, (2 + dest_col*131, 2 + dest_row*131))
grid.save(FIG / 'generated.png')

selection = json.loads((ROOT / 'output/final_selection.json').read_text())
baseline = json.loads((ROOT / 'output/val_e100/metrics.json').read_text())
reference = json.loads((ROOT / 'output/reference_cuda/metrics.json').read_text())
lines = []
for i, row in enumerate([baseline] + selection['validation_best']):
    lines.append(f'\\newcommand{{\\ValFID{chr(65+i)}}}{{{float(row["fid_ignite"]):.4f}}}')
    lines.append(f'\\newcommand{{\\ValIS{chr(65+i)}}}{{{float(row["inception_score"]):.2f}}}')
for key, val, fmt in [('TestFID',selection['test']['fid_ignite'],'.4f'),
                      ('TestIS',selection['test']['inception_score'],'.2f'),
                      ('RealValFID',reference['fid_ignite']['train_vs_val'],'.4f'),
                      ('RealTestFID',reference['fid_ignite']['train_vs_test'],'.4f'),
                      ('RealTestIS',reference['inception_score']['test'],'.2f')]:
    lines.append(f'\\newcommand{{\\{key}}}{{{val:{fmt}}}}')
rows = list(csv.DictReader(open(ROOT / 'output/val_r1/summary.csv')))
fid100 = float(next(r for r in rows if int(r['epoch']) == 100)['fid_ignite'])
lines.append(f'\\newcommand{{\\RoneHundredFID}}{{{fid100:.4f}}}')
(ROOT / 'report/values.tex').write_text('\n'.join(lines)+'\n')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.patches import Patch

# Load & sort
df = pd.read_csv('./ligand_summary.csv')
df_sorted = df.sort_values('num_sequences', ascending=False).reset_index(drop=True)

num_ligands = len(df_sorted)
x = np.arange(1, num_ligands + 1)
y = df_sorted['num_sequences'].values

# Split indices
idx_over_end = 46
idx_under_end = idx_over_end + 123

fig, ax = plt.subplots(figsize=(10, 4))

baseline = 0.8

# Plot curve
ax.plot(x, y, color='firebrick', linewidth=1, label='Ligand counts')

# Shaded regions with labels for legend
ax.fill_between(x[:idx_over_end], y[:idx_over_end], baseline,
                color='orange', alpha=0.25)
ax.fill_between(x[idx_over_end-1:idx_under_end], y[idx_over_end-1:idx_under_end], baseline,
                color='royalblue', alpha=0.25)
ax.fill_between(x[idx_under_end:], y[idx_under_end:], baseline,
                color='green', alpha=0.25)

# Numeric labels for region sizes
label_y2 = baseline * 1.5
ax.text(8, label_y2, '45', ha='center', va='bottom', fontsize=12, color='black')
ax.text(95, label_y2, '123', ha='center', va='bottom', fontsize=12, color='black')
ax.text(500, label_y2, '5612', ha='center', va='bottom', fontsize=12, color='black')

# Draw double-headed arrows under region labels
arrow_y = baseline * 1.5
ax.annotate('', xy=(idx_over_end, arrow_y), xytext=(1, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='black', linewidth=0.8, shrinkA=0, shrinkB=0))
ax.annotate('', xy=(idx_under_end, arrow_y), xytext=(idx_over_end, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='black', linewidth=0.8, shrinkA=0, shrinkB=0))
ax.annotate('', xy=(num_ligands, arrow_y), xytext=(idx_under_end, arrow_y),
            arrowprops=dict(arrowstyle='<->', color='black', linewidth=0.8, shrinkA=0, shrinkB=0))

# Horizontal dashed reference lines truncated at curve
for val, txt in zip([199, 100, 20], ['200', '', '20']):
    # find first index where y drops below the threshold
    cross_idxs = np.where(y < val)[0]
    x_end = x[cross_idxs[0]] if cross_idxs.size > 0 else x[-1]
    ax.hlines(val, x[0], x_end, linestyle=':', color='black', linewidth=0.8)
    ax.text(0.78, val, txt, va='bottom', ha='left', fontsize=8, color='gray')

# Vertical category borders
# Compute intersections for truncated vertical lines
y_over = y[idx_over_end-1]
y_under = y[idx_under_end-1]
ax.vlines(idx_over_end, baseline, y_over, linestyle='--', color='black', linewidth=0.8)
ax.vlines(idx_under_end, baseline, y_under, linestyle='--', color='black', linewidth=0.8)
ax.vlines(5780, baseline, 1.5, linestyle='--', color='black', linewidth=0.8)

# Log scale
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(bottom=baseline)
ax.set_xlim(left=1, right=10000)

# Custom y-axis ticks: replace '10^0' with '1'
yticks = [1, 10, 100, 1000]
ytick_labels = ['1', '10', '100', '1,000']
ax.yaxis.set_major_locator(FixedLocator(yticks))
ax.yaxis.set_major_formatter(FixedFormatter(ytick_labels))

# Custom x ticks
ticks = [1, 10, 100, 1000, 10000]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FixedFormatter(['1', '10', '100', '1,000', '10,000']))

# Axis labels and title
ax.set_xlabel('Ligand index')
ax.set_ylabel('# Protein sequences')
# ax.set_title('Log–log distribution of protein‑sequence counts per ligand')

# Split indices below axis
label_y = baseline * 0.85
ax.text(idx_over_end, label_y, '46', va='top', ha='center', fontsize=8, color='gray')
ax.text(idx_under_end, label_y, '169', va='top', ha='center', fontsize=8, color='gray')
ax.text(num_ligands, label_y, '5780', va='top', ha='right', fontsize=8, color='gray')

# Region labels
# ax.text(idx_over_end/2, 2000, 'Over‑represented\n(45 ligands)', ha='center', va='center', fontsize=9)
# mid_under = (idx_over_end + idx_under_end) / 2
# ax.text(mid_under-20, 300, 'Under‑represented\n(123 ligands)', ha='center', va='center', fontsize=9)
# mid_zero = (idx_under_end + num_ligands) / 2
# ax.text(mid_zero * 0.40, 30, 'Zero‑shot\n(5612 ligands)', ha='center', va='center', fontsize=9)

# Build custom legend with patches
legend_handles = [
    Patch(facecolor='orange', alpha=0.25, label='Over‑represented'),
    Patch(facecolor='royalblue', alpha=0.25, label='Under‑represented'),
    Patch(facecolor='green', alpha=0.25, label='Zero‑shot')
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=True)

fig.tight_layout()

out_path = './ligand_samples_distribution.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
# plot
plt.show()

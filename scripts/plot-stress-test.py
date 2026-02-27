# AI note: this script is entirely generated using AI (claude code)
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="tab10")

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv_file>")
    sys.exit(1)

csv_path = Path(sys.argv[1])
matrix_name = csv_path.stem

df = pd.read_csv(csv_path)

df["triplet"] = list(zip(df["nev"], df["ncv"], df["p"]))
order = sorted(df["triplet"].unique())
x_positions = {t: i for i, t in enumerate(order)}
x_labels = [str(t) for t in order]

metrics = [
    ("elapsed", "elapsed (s)"),
    ("matvecs", "matvecs"),
    ("restarts", "restarts"),
]

which_values = sorted(df["which"].unique())
n_rows = len(metrics)
n_cols = len(which_values)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True)

# Ensure axes is always 2D
if n_cols == 1:
    axes = axes[:, None]

for col_idx, which in enumerate(which_values):
    df_which = df[df["which"] == which]
    for method, group in df_which.groupby("method"):
        group = group.copy()
        group["x"] = group["triplet"].map(x_positions)
        group = group.sort_values("x")
        for row_idx, (metric, _) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            ax.plot(group["x"], group[metric], marker="o", label=method, linewidth=2, markersize=7)

    axes[0, col_idx].set_title(f"which={which}")
    axes[-1, col_idx].set_xticks(range(len(order)))
    axes[-1, col_idx].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[-1, col_idx].set_xlabel("(nev, ncv, p)")

for row_idx, (_, ylabel) in enumerate(metrics):
    axes[row_idx, 0].set_ylabel(ylabel)

axes[0, 0].legend()
fig.suptitle(matrix_name)
fig.tight_layout()
plt.show()

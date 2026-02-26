# AI note: this script was generated using AI (claude code)
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="tab10")

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv_file>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

df["triplet"] = list(zip(df["nev"], df["ncv"], df["p"]))
order = sorted(df["triplet"].unique())
x_positions = {t: i for i, t in enumerate(order)}
x_labels = [str(t) for t in order]

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

metrics = [
    ("elapsed", "elapsed (s)"),
    ("matvecs", "matvecs"),
    ("restarts", "restarts"),
]

for method, group in df.groupby("method"):
    group = group.copy()
    group["x"] = group["triplet"].map(x_positions)
    group = group.sort_values("x")
    for ax, (col, ylabel) in zip(axes, metrics):
        ax.plot(group["x"], group[col], marker="o", label=method, linewidth=2, markersize=7)

for ax, (_, ylabel) in zip(axes, metrics):
    ax.set_ylabel(ylabel)

axes[0].legend()
axes[-1].set_xticks(range(len(order)))
axes[-1].set_xticklabels(x_labels, rotation=45, ha="right")
axes[-1].set_xlabel("(nev, ncv, p)")
fig.tight_layout()
plt.show()

from pathlib import Path
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE
with (DATA / "overall_results.csv").open(newline="") as f:
    overall = list(csv.DictReader(f))
with (DATA / "task_group_results.csv").open(newline="") as f:
    categories = list(csv.DictReader(f))
with (DATA / "per_task_results.csv").open(newline="") as f:
    tasks = list(csv.DictReader(f))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
COLORS = {
    "paper": "#D5A95F",
    "official": "#9AA3AD",
    "gpu": "#3975A5",
    "qcs": "#2E8B68",
}
SYSTEMS = [
    ("official_release_percent", "Official released results", COLORS["official"]),
    ("rtx4090_bf16_percent", "RTX 4090 BF16", COLORS["gpu"]),
    ("qcs8550_qnn_percent", "QCS8550 QNN", COLORS["qcs"]),
]

def save(fig, stem):
    files = {
        "figure1_overall_success": (
            "figure1_overall_success.png",
            "figure1_overall_success.svg",
            "figure1_overall_success.pdf",
        ),
        "figure2_task_families": (
            "figure2_task_families.png",
            "figure2_task_families.svg",
            "figure2_task_families.pdf",
        ),
        "figure3_per_task": (
            "figure3_per_task.png",
            "figure3_per_task.svg",
            "figure3_per_task.pdf",
        ),
    }
    png_file, svg_file, pdf_file = files[stem]
    fig.savefig(OUT / png_file, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / svg_file, bbox_inches="tight")
    fig.savefig(OUT / pdf_file, bbox_inches="tight")
    plt.close(fig)

# Figure 1: headline result, with the paper's reported headline distinguished
# from the exact count reconstructed from its released episode-level results.
fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
labels = [
    "Official paper headline",
    "Official episode-level results",
    "RTX 4090 BF16 reproduction",
    "QCS8550 QNN migration",
]
values = [74.5, 1781/2400*100, 1782/2400*100, 1793/2400*100]
colors = [COLORS["paper"], COLORS["official"], COLORS["gpu"], COLORS["qcs"]]
suffixes = ["paper-reported", "1,781 / 2,400", "1,782 / 2,400", "1,793 / 2,400"]
y = np.arange(len(labels))
ax.barh(y, values, color=colors, height=0.58, edgecolor="none")
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xlabel("Task success rate (%)")
ax.set_title("RoboCasa benchmark: overall success rate", loc="left", pad=10, weight="bold")
ax.grid(axis="x", color="#E8EBEF", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for yi, val, suffix in zip(y, values, suffixes):
    ax.text(val - 1.2, yi, f"{val:.2f}%  |  {suffix}", va="center", ha="right", fontsize=7.4, color="white", weight="semibold")
fig.text(0.01, -0.02, "24 tasks × 100 episodes. The paper headline is shown separately from the released episode-level count.", fontsize=7.5, color="#525D68")
save(fig, "figure1_overall_success")

# Figure 2: aggregate task-family comparison.
fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
x = np.arange(len(categories))
width = 0.23
fields = [
    ("official_release_percent", "Official released", COLORS["official"]),
    ("rtx4090_bf16_percent", "RTX 4090 BF16", COLORS["gpu"]),
    ("qcs8550_qnn_percent", "QCS8550 QNN", COLORS["qcs"]),
]
for j, (field, label, color) in enumerate(fields):
    vals = [float(row[field]) for row in categories]
    bars = ax.bar(x + (j-1)*width, vals, width, color=color, label=label, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2.8, f"{val:.1f}", ha="center", va="bottom", fontsize=7.2, color="#27313B")
ax.set_ylim(0, 106)
ax.set_ylabel("Task success rate (%)")
ax.set_xticks(x, [f"{row['task_group']}\n(n={row['episodes']})" for row in categories])
ax.set_title("Success rate by task family", loc="left", pad=10, weight="bold")
ax.grid(axis="y", color="#E8EBEF", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
save(fig, "figure2_task_families")

# Figure 3: task-level results for all 24 tasks; QCS-vs-GPU difference is
# printed in percentage points at the right edge.
group_order = ["Object transfer", "Doors and drawers", "Appliance controls", "Coffee tasks"]
ordered = []
for group in group_order:
    ordered.extend([row for row in tasks if row["task_group"] == group])
positions = []
group_ranges = []
cursor = 0.0
for group in group_order:
    block = [row for row in ordered if row["task_group"] == group]
    start = cursor
    for row in block:
        positions.append(cursor)
        cursor += 1.0
    end = cursor - 1.0
    group_ranges.append((group, start, end))
    cursor += 0.8
fig, ax = plt.subplots(figsize=(7.2, 12.0), layout="constrained")
for idx, (row, y0) in enumerate(zip(ordered, positions)):
    if idx % 2 == 0:
        ax.axhspan(y0-0.43, y0+0.43, color="#F7F8FA", zorder=0)
    gpu = float(row["rtx4090_bf16_percent"])
    qcs = float(row["qcs8550_qnn_percent"])
    official = float(row["official_release_percent"])
    ax.plot([gpu, qcs], [y0, y0], color="#BCC4CC", linewidth=1.2, zorder=1)
    ax.scatter([official], [y0], s=24, marker="o", color=COLORS["official"], edgecolor="white", linewidth=0.35, zorder=3)
    ax.scatter([gpu], [y0], s=29, marker="D", color=COLORS["gpu"], edgecolor="white", linewidth=0.35, zorder=4)
    ax.scatter([qcs], [y0], s=33, marker="s", color=COLORS["qcs"], edgecolor="white", linewidth=0.35, zorder=5)
    delta = int(row["qcs_vs_rtx4090_pp"])
    dc = COLORS["qcs"] if delta > 0 else ("#B45454" if delta < 0 else "#68727D")
    ax.text(104.0, y0, f"{delta:+d}", va="center", ha="center", fontsize=7.4, color=dc, weight="bold")
for group, start, end in group_ranges[:-1]:
    ax.axhline(end+0.55, color="#DCE1E6", linewidth=0.8)
ax.set_yticks(positions, [row["task"] for row in ordered])
ax.invert_yaxis()
ax.set_xlim(0, 109)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xlabel("Task success rate (%)")
ax.set_title("Task-level success rates", loc="left", pad=10, weight="bold")
ax.grid(axis="x", color="#E8EBEF", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
ax.text(104, -0.9, "Δ pp", ha="center", va="bottom", fontsize=7.7, color="#525D68", weight="bold", clip_on=False)
handles = [
    plt.Line2D([0],[0], marker="o", linestyle="none", markerfacecolor=COLORS["official"], markeredgecolor="white", markersize=6, label="Official released"),
    plt.Line2D([0],[0], marker="D", linestyle="none", markerfacecolor=COLORS["gpu"], markeredgecolor="white", markersize=6, label="RTX 4090 BF16"),
    plt.Line2D([0],[0], marker="s", linestyle="none", markerfacecolor=COLORS["qcs"], markeredgecolor="white", markersize=6, label="QCS8550 QNN"),
]
ax.legend(handles=handles, ncols=3, loc="lower center", bbox_to_anchor=(0.5, -0.065))
save(fig, "figure3_per_task")

print("Rendered 3 figures in PNG, SVG, and PDF.")

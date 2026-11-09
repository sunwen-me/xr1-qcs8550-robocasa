from pathlib import Path
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "latency_and_model_storage.csv"

with DATA.open(newline="") as f:
    rows = {row["metric"]: row for row in csv.DictReader(f)}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

FLOAT = "#8B969F"
W8 = "#2E8B68"

def add_panel_label(ax, label):
    offset = ScaledTranslation(-10 / 72, 4 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + offset, fontsize=8,
            fontweight="bold", ha="left", va="bottom", clip_on=False)

def metric(name, field):
    return float(rows[name][field])

fig, (ax_latency, ax_storage, ax_board_memory) = plt.subplots(
    1, 3, figsize=(9.5, 3.35), layout="constrained",
    gridspec_kw={"width_ratios": [1, 1, 1]},
)

# Panel a: warm end-to-end latency; bars show means and labels retain the
# median/P95 summaries. The first cold request is reported separately below.
add_panel_label(ax_latency, "a")
labels = ["Qwen FLOAT", "Qwen W8A16"]
means = [metric("warm_elapsed_mean", "float_qwen"), metric("warm_elapsed_mean", "w8a16_qwen")]
medians = [metric("warm_elapsed_median", "float_qwen"), metric("warm_elapsed_median", "w8a16_qwen")]
p95 = [metric("warm_elapsed_p95", "float_qwen"), metric("warm_elapsed_p95", "w8a16_qwen")]
colors = [FLOAT, W8]
y = np.arange(2)
bars = ax_latency.barh(y, means, color=colors, height=0.52, edgecolor="none")
ax_latency.set_yticks(y, labels)
ax_latency.invert_yaxis()
ax_latency.set_xlim(0, 2.55)
ax_latency.set_xticks(np.arange(0, 2.6, 0.5))
ax_latency.set_xlabel("Client end-to-end latency (s)")
ax_latency.set_title("Warm request latency", loc="left", pad=9, weight="bold")
ax_latency.grid(False)
ax_latency.set_axisbelow(True)
for bar, mean, median, tail in zip(bars, means, medians, p95):
    ax_latency.text(mean - 0.08, bar.get_y() + bar.get_height()/2,
                    f"{mean:.2f} s", va="center", ha="right",
                    fontsize=7.3, color="white", weight="bold")
ax_latency.text(
    0.02, -0.24,
    f"First cold request: {metric('cold_elapsed','float_qwen'):.2f} s → {metric('cold_elapsed','w8a16_qwen'):.2f} s",
    transform=ax_latency.transAxes, fontsize=7.2, color="#525D68", ha="left", va="top",
)

# Panel b: Qwen context files report persistent model storage, not runtime RAM.
add_panel_label(ax_storage, "b")
float_size = metric("qwen_context_total", "float_qwen")
w8_size = metric("qwen_context_total", "w8a16_qwen")
sizes = [float_size, w8_size]
bars = ax_storage.barh([0, 1], sizes, color=[FLOAT, W8], height=0.52, edgecolor="none")
ax_storage.set_yticks([0, 1], ["Qwen FLOAT", "Qwen W8A16"])
ax_storage.invert_yaxis()
ax_storage.set_xlim(0, 9.2)
ax_storage.set_xticks(np.arange(0, 9.5, 2))
ax_storage.set_xlabel("Total Qwen context files (GiB)")
ax_storage.set_title("On-disk context size", loc="left", pad=9, weight="bold")
ax_storage.grid(False)
ax_storage.set_axisbelow(True)
for bar, val in zip(bars, sizes):
    ax_storage.text(val - 0.12, bar.get_y()+bar.get_height()/2, f"{val:.2f} GiB",
                    ha="right", va="center", fontsize=7.3, color="white", weight="bold")
ax_storage.text(0.02, -0.24, "W8A16 context files are 44.8% smaller.",
                transform=ax_storage.transAxes, fontsize=7.0, color="#525D68", ha="left", va="top")

# Panel c: board-wide available memory sampled after each warm inference request.
add_panel_label(ax_board_memory, "c")
available = [
    metric("warm_mem_available_median", "float_qwen") / 1024.0,
    metric("warm_mem_available_median", "w8a16_qwen") / 1024.0,
]
bars = ax_board_memory.barh([0, 1], available, color=[FLOAT, W8], height=0.52, edgecolor="none")
ax_board_memory.set_yticks([0, 1], ["Qwen FLOAT", "Qwen W8A16"])
ax_board_memory.invert_yaxis()
ax_board_memory.set_xlim(0, 8.5)
ax_board_memory.set_xticks(np.arange(0, 8.5, 2))
ax_board_memory.set_xlabel("Board MemAvailable (GiB)")
ax_board_memory.set_title("Remaining board memory", loc="left", pad=9, weight="bold")
ax_board_memory.grid(False)
ax_board_memory.set_axisbelow(True)
for bar, val in zip(bars, available):
    ax_board_memory.text(val - 0.10, bar.get_y()+bar.get_height()/2, f"{val:.2f} GiB",
                         ha="right", va="center", fontsize=7.3, color="white", weight="bold")
delta_mib = (metric("warm_mem_available_median", "w8a16_qwen")
             - metric("warm_mem_available_median", "float_qwen"))
ax_board_memory.text(0.02, -0.24, f"W8A16 run: {delta_mib / 1024.0:.2f} GiB more available (warm median, n=23).",
                     transform=ax_board_memory.transAxes, fontsize=6.8, color="#525D68", ha="left", va="top")

fig.text(
    0.01, -0.035,
    "Paired QCS8550 test: 24 fixed requests; latency and board MemAvailable summaries exclude the first cold request (n=23).",
    fontsize=7.0, color="#525D68",
)

if require_matplotlib_panel_alignment is not None:
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(HERE / "figure4_runtime_performance.alignment.json"),
        overlay_svg=str(HERE / "figure4_runtime_performance.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )

fig.savefig(HERE / "figure4_runtime_performance.png", dpi=600, bbox_inches="tight")
fig.savefig(HERE / "figure4_runtime_performance.svg", bbox_inches="tight")
fig.savefig(HERE / "figure4_runtime_performance.pdf", bbox_inches="tight")
plt.close(fig)
print("Rendered figure4_runtime_performance in PNG, SVG, and PDF.")

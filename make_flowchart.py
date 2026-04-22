"""Generate a simplified pipeline flowchart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")

# Subtle grid dots for texture
for gx in range(0, 17):
    for gy in range(0, 10):
        ax.plot(gx, gy, ".", color="#161b22", markersize=1.5)

def box(x, y, w, h, text, color="#161b22", edge="#30363d", fontsize=12, bold=False,
        shadow=True, glow=None):
    if shadow:
        s = mpatches.FancyBboxPatch((x + 0.06, y - 0.06), w, h, boxstyle="round,pad=0.2",
                                     facecolor="#000000", edgecolor="none", alpha=0.3)
        ax.add_patch(s)
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                     facecolor=color, edgecolor=edge, linewidth=2.5)
    ax.add_patch(rect)
    if glow:
        g = mpatches.FancyBboxPatch((x - 0.05, y - 0.05), w + 0.1, h + 0.1,
                                      boxstyle="round,pad=0.25",
                                      facecolor="none", edgecolor=glow, linewidth=1, alpha=0.3)
        ax.add_patch(g)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color="white", weight=weight, linespacing=1.4)

def curved_arrow(x1, y1, x2, y2, color="#58a6ff", style="->", lw=2.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0",
                                mutation_scale=15))

def label(x, y, text, fontsize=9, color="#8b949e"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)

# ── Title ──
ax.text(8, 8.3, "Treeshrew Stereo Pipeline", ha="center", va="center",
        fontsize=22, color="#f0f6fc", weight="bold", family="sans-serif")
ax.text(8, 7.8, "Monocular Video  \u2192  Stereo Pair for Neuroscience",
        ha="center", va="center", fontsize=12, color="#8b949e")

# Divider line
ax.plot([1, 15], [7.5, 7.5], color="#21262d", lw=1.5)

# ── Step numbers ──
step_color = "#58a6ff"

# ── Row 1: Main pipeline ──
# Step 1: Input
ax.text(1.85, 6.85, "1", ha="center", fontsize=10, color=step_color, weight="bold",
        bbox=dict(boxstyle="circle,pad=0.15", facecolor="#0d1117", edgecolor=step_color, lw=1.5))
box(0.8, 5.5, 2.2, 1.2, "Monocular\nVideo", color="#1c3a5e", edge="#58a6ff", fontsize=13, bold=True, glow="#58a6ff")

# Step 2: Depth
ax.text(4.65, 6.85, "2", ha="center", fontsize=10, color=step_color, weight="bold",
        bbox=dict(boxstyle="circle,pad=0.15", facecolor="#0d1117", edgecolor=step_color, lw=1.5))
box(3.6, 5.5, 2.2, 1.2, "MiDaS Depth\nEstimation", color="#161b22", edge="#30363d", fontsize=12)
curved_arrow(3.0, 6.1, 3.6, 6.1)
label(3.3, 6.45, "frame", fontsize=8)

# Step 3: Disparity
ax.text(7.45, 6.85, "3", ha="center", fontsize=10, color=step_color, weight="bold",
        bbox=dict(boxstyle="circle,pad=0.15", facecolor="#0d1117", edgecolor=step_color, lw=1.5))
box(6.4, 5.5, 2.2, 1.2, "Disparity\nCalibration", color="#161b22", edge="#30363d", fontsize=12)
curved_arrow(5.8, 6.1, 6.4, 6.1)
label(6.1, 6.45, "depth map", fontsize=8)

# Step 4: Warp
ax.text(10.25, 6.85, "4", ha="center", fontsize=10, color=step_color, weight="bold",
        bbox=dict(boxstyle="circle,pad=0.15", facecolor="#0d1117", edgecolor=step_color, lw=1.5))
box(9.2, 5.5, 2.2, 1.2, "Stereo Warp\n(Left + Right)", color="#161b22", edge="#30363d", fontsize=12)
curved_arrow(8.6, 6.1, 9.2, 6.1)
label(8.9, 6.45, "disparity", fontsize=8)

# Step 5: Refinement
ax.text(13.05, 6.85, "5", ha="center", fontsize=10, color=step_color, weight="bold",
        bbox=dict(boxstyle="circle,pad=0.15", facecolor="#0d1117", edgecolor=step_color, lw=1.5))
box(12.0, 5.5, 2.2, 1.2, "ResNet18\nRefinement", color="#161b22", edge="#30363d", fontsize=12)
curved_arrow(11.4, 6.1, 12.0, 6.1)
label(11.7, 6.45, "warped + masks", fontsize=8)

# Down arrow to output
curved_arrow(13.1, 5.5, 13.1, 4.6)

# Output box
box(11.3, 3.4, 3.6, 1.1, "Stereo Output (side-by-side)",
    color="#1c3a5e", edge="#58a6ff", fontsize=13, bold=True, glow="#58a6ff")
label(13.1, 4.85, "refined L & R", fontsize=8)

# ── Divider ──
ax.plot([1, 15], [2.8, 2.8], color="#21262d", lw=1.5)

# ── Bottom: Simple workflow ──
ax.text(8, 2.35, "Quick Start", ha="center", fontsize=14, color="#f0f6fc", weight="bold")

bx_y = 1.3
# input folder
box(1.5, bx_y, 2.4, 0.8, "input/", color="#1c3a5e", edge="#58a6ff", fontsize=13, bold=True)
label(2.7, bx_y - 0.25, "drop .mp4 files here", fontsize=8, color="#58a6ff")

curved_arrow(3.9, bx_y + 0.4, 5.3, bx_y + 0.4, color="#58a6ff")

# command
box(5.3, bx_y, 4.0, 0.8, "python run.py infer", color="#161b22", edge="#da3633", fontsize=13, bold=True)

curved_arrow(9.3, bx_y + 0.4, 10.7, bx_y + 0.4, color="#58a6ff")

# output folder
box(10.7, bx_y, 2.8, 0.8, "output/", color="#1c3a5e", edge="#58a6ff", fontsize=13, bold=True)
label(12.1, bx_y - 0.25, "stereo videos here", fontsize=8, color="#58a6ff")

# Config note
ax.text(8, 0.55, "All settings in  configs/default.yaml", ha="center", fontsize=10, color="#484f58",
        family="monospace")

plt.tight_layout()
plt.savefig("pipeline_flowchart.png", dpi=180, bbox_inches="tight", facecolor="#0d1117")
print("Saved pipeline_flowchart.png")

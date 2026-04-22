"""Generate a training metrics summary chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Training Metrics Summary", fontsize=20, color="#f0f6fc", weight="bold", y=0.98)

for ax in axes:
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=10)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#21262d", linewidth=0.8)

# ── Chart 1: Val SSIM by run ──
ax1 = axes[0]
runs = ["Run 1\n(40 ep)", "Run 2\n(60 ep)", "Run 3\n(tanh fix)"]
ssim_vals = [0.3648, 0.3649, None]
colors = ["#da3633", "#da3633", "#30363d"]
bars = ax1.bar([0, 1, 2], [0.3648, 0.3649, 0], color=colors, edgecolor="#30363d", linewidth=1.5, width=0.6)
ax1.bar(2, 0.05, color="#30363d", edgecolor="#58a6ff", linewidth=1.5, width=0.6, linestyle="--")
ax1.text(2, 0.15, "TBD", ha="center", fontsize=12, color="#58a6ff", weight="bold")
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(runs, fontsize=10, color="#8b949e")
ax1.set_ylabel("SSIM", fontsize=12, color="#8b949e")
ax1.set_title("Validation SSIM", fontsize=14, color="#f0f6fc", pad=10)
ax1.set_ylim(0, 1.0)
ax1.axhline(y=0.7, color="#3fb950", linestyle="--", linewidth=1, alpha=0.7)
ax1.text(2.4, 0.71, "decent", fontsize=9, color="#3fb950")
ax1.axhline(y=0.9, color="#58a6ff", linestyle="--", linewidth=1, alpha=0.7)
ax1.text(2.4, 0.91, "great", fontsize=9, color="#58a6ff")
for i, v in enumerate(ssim_vals):
    if v is not None:
        ax1.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10, color="#f0f6fc", weight="bold")

# ── Chart 2: Val PSNR by run ──
ax2 = axes[1]
psnr_vals = [13.57, 13.57, None]
colors2 = ["#da3633", "#da3633", "#30363d"]
ax2.bar([0, 1, 2], [13.57, 13.57, 0], color=colors2, edgecolor="#30363d", linewidth=1.5, width=0.6)
ax2.bar(2, 1, color="#30363d", edgecolor="#58a6ff", linewidth=1.5, width=0.6, linestyle="--")
ax2.text(2, 5, "TBD", ha="center", fontsize=12, color="#58a6ff", weight="bold")
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(runs, fontsize=10, color="#8b949e")
ax2.set_ylabel("PSNR (dB)", fontsize=12, color="#8b949e")
ax2.set_title("Validation PSNR", fontsize=14, color="#f0f6fc", pad=10)
ax2.set_ylim(0, 35)
ax2.axhline(y=20, color="#3fb950", linestyle="--", linewidth=1, alpha=0.7)
ax2.text(2.4, 20.5, "decent", fontsize=9, color="#3fb950")
ax2.axhline(y=30, color="#58a6ff", linestyle="--", linewidth=1, alpha=0.7)
ax2.text(2.4, 30.5, "great", fontsize=9, color="#58a6ff")
for i, v in enumerate(psnr_vals):
    if v is not None:
        ax2.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10, color="#f0f6fc", weight="bold")

# ── Chart 3: Mean Disparity ──
ax3 = axes[2]
disp_vals = [0.4, 0.5, None]
colors3 = ["#da3633", "#da3633", "#30363d"]
ax3.bar([0, 1, 2], [0.4, 0.5, 0], color=colors3, edgecolor="#30363d", linewidth=1.5, width=0.6)
ax3.bar(2, 0.5, color="#30363d", edgecolor="#58a6ff", linewidth=1.5, width=0.6, linestyle="--")
ax3.text(2, 3, "TBD", ha="center", fontsize=12, color="#58a6ff", weight="bold")
ax3.set_xticks([0, 1, 2])
ax3.set_xticklabels(runs, fontsize=10, color="#8b949e")
ax3.set_ylabel("Pixels", fontsize=12, color="#8b949e")
ax3.set_title("Mean Disparity", fontsize=14, color="#f0f6fc", pad=10)
ax3.set_ylim(0, 20)
ax3.axhspan(5, 15, color="#3fb950", alpha=0.08)
ax3.text(2.4, 9.5, "target\nrange", fontsize=9, color="#3fb950", va="center")
for i, v in enumerate(disp_vals):
    if v is not None:
        ax3.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=10, color="#f0f6fc", weight="bold")

# ── Legend at bottom ──
fig.text(0.5, 0.02,
         "Run 1 & 2: disparity collapsed (residual canceled base shift)  |  "
         "Run 3: residual constrained to \u00b12px via tanh — should fix collapse",
         ha="center", fontsize=10, color="#8b949e")

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig("metrics_chart.png", dpi=180, bbox_inches="tight", facecolor="#0d1117")
print("Saved metrics_chart.png")

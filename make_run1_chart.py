"""Generate training charts for Run 1 (40-epoch run)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

with open("/tmp/run1_data.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("Training Run 1 — 40 Epochs", fontsize=22, color="#f0f6fc", weight="bold", y=0.98)

def style_ax(ax, title, ylabel):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#21262d", linewidth=0.8)
    ax.set_title(title, fontsize=14, color="#f0f6fc", pad=10)
    ax.set_ylabel(ylabel, fontsize=11, color="#8b949e")
    ax.set_xlabel("Step", fontsize=10, color="#8b949e")

def smooth(vals, window=500):
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="valid")

# ── 1. Total Loss ──
ax = axes[0, 0]
style_ax(ax, "Total Loss", "Loss")
steps, vals = zip(*data["train/loss"])
steps, vals = np.array(steps), np.array(vals)
ax.plot(steps, vals, color="#58a6ff", alpha=0.08, linewidth=0.5)
s_vals = smooth(vals)
s_steps = steps[:len(s_vals)]
ax.plot(s_steps, s_vals, color="#58a6ff", linewidth=2)
ax.set_ylim(0, min(np.percentile(vals, 99), 10))

# ── 2. L1 Loss ──
ax = axes[0, 1]
style_ax(ax, "L1 Loss (Pixel Error)", "L1")
steps, vals = zip(*data["train/l1"])
steps, vals = np.array(steps), np.array(vals)
ax.plot(steps, vals, color="#3fb950", alpha=0.08, linewidth=0.5)
s_vals = smooth(vals)
ax.plot(steps[:len(s_vals)], s_vals, color="#3fb950", linewidth=2)
ax.set_ylim(0, min(np.percentile(vals, 99), 1))

# ── 3. Perceptual Loss ──
ax = axes[0, 2]
style_ax(ax, "Perceptual Loss (VGG)", "Loss")
steps, vals = zip(*data["train/perc"])
steps, vals = np.array(steps), np.array(vals)
ax.plot(steps, vals, color="#d2a8ff", alpha=0.08, linewidth=0.5)
s_vals = smooth(vals)
ax.plot(steps[:len(s_vals)], s_vals, color="#d2a8ff", linewidth=2)
ax.set_ylim(0, min(np.percentile(vals, 99), 5))

# ── 4. Val SSIM ──
ax = axes[1, 0]
style_ax(ax, "Validation SSIM", "SSIM")
ax.set_xlabel("Epoch", fontsize=10, color="#8b949e")
steps, vals = zip(*data["val/ssim"])
ax.plot(steps, vals, color="#f78166", linewidth=2.5, marker="o", markersize=4)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.7, color="#3fb950", linestyle="--", linewidth=1, alpha=0.5)
ax.text(max(steps) * 0.85, 0.72, "decent", fontsize=9, color="#3fb950")

# ── 5. Val PSNR ──
ax = axes[1, 1]
style_ax(ax, "Validation PSNR", "dB")
ax.set_xlabel("Epoch", fontsize=10, color="#8b949e")
steps, vals = zip(*data["val/psnr"])
ax.plot(steps, vals, color="#ffa657", linewidth=2.5, marker="o", markersize=4)
ax.set_ylim(0, 35)
ax.axhline(y=20, color="#3fb950", linestyle="--", linewidth=1, alpha=0.5)
ax.text(max(steps) * 0.85, 21, "decent", fontsize=9, color="#3fb950")

# ── 6. Disparity Supervision ──
ax = axes[1, 2]
style_ax(ax, "Disparity Supervision Loss", "Loss")
steps, vals = zip(*data["train/disp_sup"])
steps, vals = np.array(steps), np.array(vals)
ax.plot(steps, vals, color="#79c0ff", alpha=0.15, linewidth=0.5)
s_vals = smooth(vals, window=200)
ax.plot(steps[:len(s_vals)], s_vals, color="#79c0ff", linewidth=2)
ax.set_ylim(0, min(np.percentile(vals, 99), 20))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("run1_chart.png", dpi=180, bbox_inches="tight", facecolor="#0d1117")
print("Saved run1_chart.png")

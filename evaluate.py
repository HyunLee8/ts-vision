"""Quick evaluation: compare predicted right view against ground truth."""
import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from inference.infer import load_model, frame_to_tensor, tensor_to_bgr
from utils.config import load_config

cfg = load_config("configs/default.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("checkpoints/best.pt", cfg, device)

left_gt = cv2.imread("data/processed/left/0000050.png")
right_gt = cv2.imread("data/processed/right/0000050.png")

t = frame_to_tensor(left_gt, device)
with torch.no_grad():
    out = model.synthesize(t, direction="to_right")
    pred = out["output"].squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    disp = out["disparity"].squeeze().cpu().numpy()

right_gt_rgb = cv2.cvtColor(right_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

mse = np.mean((pred - right_gt_rgb) ** 2)
psnr_val = 10 * np.log10(1.0 / mse)
ssim_val = structural_similarity(pred, right_gt_rgb, channel_axis=2, data_range=1.0)

print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")
print(f"Disparity range: {disp.min():.1f} to {disp.max():.1f} pixels")
print(f"Mean disparity: {disp.mean():.1f} pixels")

pred_bgr = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

h, w = left_gt.shape[:2]
bar_h = 40
labels = ["LEFT INPUT", "RIGHT GROUND TRUTH", "RIGHT PREDICTED"]
panels = [left_gt, right_gt, pred_bgr]

rows = []
for label, panel in zip(labels, panels):
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
    tx = (w - text_size[0]) // 2
    ty = (bar_h + text_size[1]) // 2
    cv2.putText(bar, label, (tx, ty), font, 0.7, (255, 255, 255), 2)
    rows.append(np.concatenate([bar, panel], axis=0))

comp = np.concatenate(rows, axis=1)

info_bar = np.zeros((bar_h, comp.shape[1], 3), dtype=np.uint8)
info = f"PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}  |  Mean Disp: {disp.mean():.1f} px"
text_size = cv2.getTextSize(info, font, 0.6, 1)[0]
tx = (comp.shape[1] - text_size[0]) // 2
cv2.putText(info_bar, info, (tx, 28), font, 0.6, (0, 255, 255), 1)
comp = np.concatenate([comp, info_bar], axis=0)

cv2.imwrite("comparison.png", comp)
print("Saved comparison.png (labeled: left_input | right_gt | right_predicted)")

"""Remove treeshrew dichromat color transform from a stereo video."""
import cv2
import numpy as np
import torch
import sys
from models.color_transform import TreeshrewDichromatTransform

input_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace(".mp4", "_normal.mp4")

color = TreeshrewDichromatTransform()
proj = color.brettel_lms_proj.numpy()
rgb2lms = color.rgb2lms_human.numpy()
lms2rgb = color.lms2rgb_human.numpy()

# Forward is: rgb -> lms -> proj -> rgb_out
# Full forward matrix: lms2rgb @ proj @ rgb2lms
forward_matrix = lms2rgb @ proj @ rgb2lms
inverse_matrix = np.linalg.inv(forward_matrix)

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    flat = rgb.reshape(-1, 3)
    corrected = (flat @ inverse_matrix.T).reshape(rgb.shape)
    corrected = np.clip(corrected, 0, 1)
    bgr = cv2.cvtColor((corrected * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    writer.write(bgr)
    count += 1

cap.release()
writer.release()
print(f"Saved {output_path} ({count} frames)")

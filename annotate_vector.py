"""Annotate comparison video with a disparity vector arrow at center."""
import cv2
import numpy as np
import sys

input_path = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
stereo_path = sys.argv[2] if len(sys.argv) > 2 else "stereo.mp4"
out_path = sys.argv[3] if len(sys.argv) > 3 else "comparison_vector.mp4"
max_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

cap_in = cv2.VideoCapture(input_path)
cap_st = cv2.VideoCapture(stereo_path)
fps = cap_in.get(cv2.CAP_PROP_FPS)
w_in = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_st = int(cap_st.get(cv2.CAP_PROP_FRAME_WIDTH))
max_frames = int(max_seconds * fps)

panel_w = w_st // 2
out_w = w_in + w_st
font = cv2.FONT_HERSHEY_SIMPLEX

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, h))

frame_idx = 0
while frame_idx < max_frames:
    ret1, orig = cap_in.read()
    ret2, stereo = cap_st.read()
    if not ret1 or not ret2:
        break

    pred_left = stereo[:, panel_w:]
    pred_right = stereo[:, :panel_w]

    cx, cy = w_in // 2, h // 2
    patch = 64
    y1, y2 = max(cy - patch, 0), min(cy + patch, h)
    x1, x2 = max(cx - patch, 0), min(cx + patch, w_in)

    orig_gray = cv2.cvtColor(orig[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    left_gray = cv2.cvtColor(pred_left[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(pred_right[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    def get_shift(ref_patch, target_patch):
        flow = cv2.calcOpticalFlowFarneback(
            ref_patch, target_patch, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        return flow[patch, patch]

    dx_left, dy_left = get_shift(orig_gray, left_gray)
    dx_right, dy_right = get_shift(orig_gray, right_gray)

    scale = 20.0
    arrow_color_left = (0, 255, 0)
    arrow_color_right = (0, 128, 255)

    def draw_vector_overlay(panel, dx, dy, color, label, side_label):
        cx_p, cy_p = panel.shape[1] // 2, panel.shape[0] // 2
        cv2.drawMarker(panel, (cx_p, cy_p), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        end_x = int(cx_p + dx * scale)
        end_y = int(cy_p + dy * scale)
        cv2.arrowedLine(panel, (cx_p, cy_p), (end_x, end_y), color, 2, tipLength=0.3)
        mag = np.sqrt(dx**2 + dy**2)
        txt = f"{label}: ({dx:+.1f}, {dy:+.1f}) px  mag={mag:.1f}"
        cv2.putText(panel, txt, (10, panel.shape[0] - 15), font, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(panel, side_label, (panel.shape[1]//2 - 60, 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return panel

    orig_panel = orig.copy()
    cv2.putText(orig_panel, "ORIGINAL", (w_in//2 - 50, 25), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cx_o, cy_o = w_in // 2, h // 2
    cv2.drawMarker(orig_panel, (cx_o, cy_o), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)

    left_panel = pred_left.copy()
    draw_vector_overlay(left_panel, dx_left, dy_left, arrow_color_left, "shift", "PRED LEFT")

    right_panel = pred_right.copy()
    draw_vector_overlay(right_panel, dx_right, dy_right, arrow_color_right, "shift", "PRED RIGHT")

    comp = np.concatenate([orig_panel, left_panel, right_panel], axis=1)
    writer.write(comp)
    frame_idx += 1

cap_in.release()
cap_st.release()
writer.release()

tmp = out_path.replace(".mp4", "_tmp.mp4")
import os
os.rename(out_path, tmp)
os.system(f'ffmpeg -y -i "{tmp}" -c:v libx264 -crf 18 -preset fast "{out_path}" 2>/dev/null')
os.remove(tmp)

print(f"Saved {out_path} ({frame_idx} frames, {frame_idx/fps:.1f}s)")

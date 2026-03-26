# --- visualize_path.py ---
import cv2
import torch
import numpy as np
import win32gui
import mss
import time
from collections import deque
from train_vision import SplitBrainVAE, setup_device
from train_pathfinder import Pathfinder  # Import class structure
from vision import VisionSystem
import queue

# --- CONFIG ---
WINDOW_TITLE = "Crossy Road"
VAE_CP = "checkpoints/crossy_vae_latest.pth"
PATH_CP = "checkpoints/pathfinder_latest.pth"
DEVICE = setup_device()


def process_frame(frame):
    h, w, _ = frame.shape
    crop_h = int(h * 0.16)
    cropped = frame[crop_h:, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
    return resized / 255.0


def main():
    # Load Models
    vae = SplitBrainVAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_CP, map_location=DEVICE))
    vae.eval()

    pathfinder = Pathfinder().to(DEVICE)
    pathfinder.load_state_dict(torch.load(PATH_CP, map_location=DEVICE))
    pathfinder.eval()

    # Setup Vision
    vision_q = queue.Queue(maxsize=1)
    vis = VisionSystem(WINDOW_TITLE, vision_q, show_preview=False)
    vis.start()

    frame_buffer = deque(maxlen=4)

    print("--- PATH VISUALIZER ---")

    try:
        while True:
            try:
                data = vision_q.get(timeout=1.0)
            except:
                continue

            frame = data['frame']
            pos = data['chicken_pos']
            h, w, _ = frame.shape

            display = frame.copy()

            # VAE Processing
            processed = process_frame(frame)
            frame_buffer.append(processed)
            if len(frame_buffer) < 4: continue

            stack = np.array(frame_buffer, dtype=np.float32)
            tensor_in = torch.FloatTensor(stack).unsqueeze(0).to(DEVICE)

            # Prediction
            if pos:
                # Normalize Current Pos
                cx_norm = pos[0] / w
                cy_norm = pos[1] / h
                pos_tensor = torch.FloatTensor([[cx_norm, cy_norm]]).to(DEVICE)

                with torch.no_grad():
                    _, _, mu_c, _, mu_t, _ = vae(tensor_in)
                    latents = torch.cat([mu_c, mu_t], dim=1)

                    # PREDICT PATH (6 outputs -> 3 coords)
                    path_pred = pathfinder(latents, pos_tensor).cpu().numpy()[0]

                # Draw the Dotted Line
                points = [(pos[0], pos[1])]  # Start at chicken

                # Unpack predictions (x1, y1, x2, y2, x3, y3)
                for i in range(0, 6, 2):
                    px = int(path_pred[i] * w)
                    py = int(path_pred[i + 1] * h)
                    points.append((px, py))

                    # Draw node
                    cv2.circle(display, (px, py), 4, (255, 255, 0), -1)  # Cyan Dots

                # Draw Lines
                for i in range(len(points) - 1):
                    cv2.line(display, points[i], points[i + 1], (255, 255, 0), 2)

            cv2.imshow("Ghost Path", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        vis.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
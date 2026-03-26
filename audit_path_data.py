import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

DATA_DIR = "data_pathfinder"


def main():
    print("--- PATH DATA AUDITOR ---")
    files = glob.glob(os.path.join(DATA_DIR, "*.npy"))
    files.sort(key=os.path.getmtime)

    if not files:
        print("No data found!")
        return

    print(f"Found {len(files)} chunks. Press SPACE to pause, ESC to quit.")

    total_samples = 0
    gaps_detected = 0

    # Trail history for visualization
    trail = []

    for f_idx, f_path in enumerate(files):
        print(f"Playing Chunk {f_idx + 1}/{len(files)}: {os.path.basename(f_path)}")

        try:
            # Data format: (InputStack, CurrentXY, FutureFlatVector)
            chunk = np.load(f_path, allow_pickle=True)
        except Exception as e:
            print(f"Corrupt file: {e}")
            continue

        for i in range(len(chunk)):
            total_samples += 1

            # 1. Unpack
            stack, current_pos, future_vector = chunk[i]

            # Stack is (4, 160, 160). We want the latest frame (index 3)
            # It's normalized 0-1, so scale up to 255
            img = (stack[3] * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            h, w, _ = img.shape

            # 2. Coordinates
            cx, cy = current_pos
            px, py = int(cx * w), int(cy * h)

            # 3. Gap Detection (Teleport check)
            if len(trail) > 0:
                last_px, last_py = trail[-1]
                dist = np.sqrt((px - last_px) ** 2 + (py - last_py) ** 2)

                # If chicken moved > 50 pixels in 1 frame (impossible), it's a recording gap
                if dist > 50:
                    gaps_detected += 1
                    trail = []  # Reset trail
                    # Visual Flash
                    cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 3)

            trail.append((px, py))
            if len(trail) > 20: trail.pop(0)  # Keep tail short

            # 4. Draw The Recorded Past (The Tail)
            for j in range(1, len(trail)):
                cv2.line(img, trail[j - 1], trail[j], (0, 255, 0), 2)

            # 5. Draw The Recorded Future (The Prediction Target)
            # future_vector is [x1, y1, x2, y2, x3, y3]
            future_pts = []
            for j in range(0, 6, 2):
                fx = int(future_vector[j] * w)
                fy = int(future_vector[j + 1] * h)
                cv2.circle(img, (fx, fy), 3, (255, 0, 255), -1)  # Magenta dots
                future_pts.append((fx, fy))

            # Connect current to future
            if len(future_pts) > 0:
                cv2.line(img, (px, py), future_pts[0], (255, 0, 255), 1)
                cv2.line(img, future_pts[0], future_pts[1], (255, 0, 255), 1)
                cv2.line(img, future_pts[1], future_pts[2], (255, 0, 255), 1)

            # 6. Info HUD
            cv2.putText(img, f"Sample: {total_samples}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img, f"Gaps: {gaps_detected}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Scale up for easy viewing
            display = cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Path Data Audit", display)

            key = cv2.waitKey(30)  # 30ms playback speed
            if key == 32:  # Space to pause
                cv2.waitKey(0)
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Audit Complete.")


if __name__ == "__main__":
    main()
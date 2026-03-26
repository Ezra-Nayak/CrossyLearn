# --- collect_path_data.py ---
import cv2
import numpy as np
import time
import os
import queue
from collections import deque
from vision import VisionSystem

# --- CONFIG ---
DATA_DIR = "data_pathfinder"  # Separate folder for path data
IMG_SIZE = 160
STACK_SIZE = 4
TARGET_FPS = 15
SAVE_INTERVAL = 1000

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def process_frame(frame):
    h, w, _ = frame.shape
    crop_h = int(h * 0.16)  # 16% Crop
    cropped = frame[crop_h:, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized / 255.0


def main():
    print("--- PATHFINDER DATA COLLECTOR ---")
    print("PLAY PERFECTLY. The AI learns the path YOU take.")
    print("Recording starts when Score > 0.")

    vision_q = queue.Queue(maxsize=1)
    # No preview from vision system, we draw our own
    vis = VisionSystem("Crossy Road", vision_q, show_preview=False)
    vis.start()

    # Buffer stores: (ProcessedFrame, (x, y))
    frame_buffer = deque(maxlen=STACK_SIZE + 20)  # Look 20 frames into the future
    data_cache = []
    total_saved = 0

    frame_duration = 1.0 / TARGET_FPS

    try:
        while True:
            loop_start = time.time()
            try:
                data = vision_q.get(timeout=2.0)
            except queue.Empty:
                continue

            raw_frame = data['frame']
            score = data['score']
            pos = data['chicken_pos']  # (x, y)

            # Draw HUD
            display = raw_frame.copy()

            if score is not None and pos is not None:
                # Normalize Pos to 0-1 range
                h, w, _ = raw_frame.shape
                norm_pos = (pos[0] / w, pos[1] / h)

                processed = process_frame(raw_frame)

                # Append tuple: (Frame, Coords)
                frame_buffer.append((processed, norm_pos))

                # LOGIC: We need history (t-3 to t) AND future (t+5, t+10, t+15)
                # We can only save a training sample if we have enough FUTURE data in the buffer
                if len(frame_buffer) >= STACK_SIZE + 15:
                    # Current State is at index = STACK_SIZE - 1
                    # History: 0, 1, 2, 3 (Current)
                    current_idx = STACK_SIZE - 1

                    # Extract Stack
                    img_stack = np.array([x[0] for x in list(frame_buffer)[:STACK_SIZE]])
                    current_xy = frame_buffer[current_idx][1]

                    # Extract Future Targets (The Path)
                    # We want positions at t+5, t+10, t+15 relative to current
                    future_idx = [current_idx + 5, current_idx + 10, current_idx + 15]
                    future_pts = []
                    for f_idx in future_idx:
                        pt = frame_buffer[f_idx][1]
                        future_pts.extend([pt[0], pt[1]])  # Flatten x, y

                    # Save: (InputStack, CurrentXY, FutureFlatVector)
                    data_cache.append((img_stack, current_xy, np.array(future_pts)))

                    # Visual Feedback (Green Dot on Chicken)
                    cv2.circle(display, pos, 5, (0, 255, 0), -1)

            else:
                # Clear buffer if track lost or menu
                if len(frame_buffer) > 0: frame_buffer.clear()
                cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), (0, 0, 255), 5)

            cv2.imshow("Path Collector", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            # Save to Disk
            if len(data_cache) >= SAVE_INTERVAL:
                ts = int(time.time())
                np.save(f"{DATA_DIR}/path_{ts}.npy", np.array(data_cache, dtype=object))
                print(f"[SAVE] Saved {len(data_cache)} paths.")
                total_saved += len(data_cache)
                data_cache = []  # Only clear cache, keep buffer for continuity

            elapsed = time.time() - loop_start
            time.sleep(max(0, frame_duration - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        if data_cache:
            ts = int(time.time())
            np.save(f"{DATA_DIR}/path_{ts}.npy", np.array(data_cache, dtype=object))
        vis.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
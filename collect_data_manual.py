# --- collect_data_manual.py ---
import cv2
import numpy as np
import time
import os
import queue
from collections import deque
from vision import VisionSystem

# --- CONFIG ---
DATA_DIR = "data"
IMG_SIZE = 160  # 160x160 for VAE
STACK_SIZE = 4  # 4 Past frames
TARGET_FPS = 15  # Locked recording speed
SAVE_INTERVAL = 1000  # Save to disk every N sequences

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def process_frame(frame):
    """
    1. Grayscale
    2. Resize to 160x160
    3. Normalize 0-1
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized / 255.0


def main():
    print("--- Manual Data Collector ---")
    print("1. Play the game naturally.")
    print("2. Recording starts AUTOMATICALLY when a Score is visible.")
    print("3. Recording stops when Score disappears (Death/Menu).")
    print("Press 'q' in the preview window to quit/save.")

    # Initialize Vision (Disable internal preview so we can draw our own HUD)
    vision_queue = queue.Queue(maxsize=1)
    vision = VisionSystem("Crossy Road", vision_queue, show_preview=False)
    vision.start()

    # Data Structures
    frame_buffer = deque(maxlen=STACK_SIZE + 1)  # Holds [t-3, t-2, t-1, t, t+1]
    data_cache = []
    total_frames_saved = 0

    # Timing
    frame_duration = 1.0 / TARGET_FPS

    try:
        while True:
            loop_start = time.time()

            # 1. Get latest frame from Vision System
            try:
                # Wait briefly for data, but don't block forever
                data = vision_queue.get(timeout=2.0)
            except queue.Empty:
                print("[WARN] Vision system lagging or game closed...")
                continue

            # 2. Extract Data
            raw_frame = data['frame']
            score = data['score']

            # 3. Logic: Is the run active?
            is_recording = False

            if score is not None:
                # Active Run: Process and Stack
                processed = process_frame(raw_frame)
                frame_buffer.append(processed)
                is_recording = True

                # Generate Training Pair if we have enough history
                if len(frame_buffer) == STACK_SIZE + 1:
                    seq = np.array(frame_buffer, dtype=np.float32)
                    input_stack = seq[:-1]  # t-3 to t
                    target_frame = seq[-1]  # t+1
                    data_cache.append((input_stack, target_frame))
            else:
                # Menu/Dead: Clear buffer to prevent mixing separate runs
                if len(frame_buffer) > 0:
                    frame_buffer.clear()
                    # Optional: Print when a run ends
                    # print(f"[REC] Run ended. Buffered {len(data_cache)} sequences total.")

            # 4. Visualization (Custom HUD)
            display = raw_frame.copy()
            h, w, _ = display.shape

            # Draw Status Border
            color = (0, 255, 0) if is_recording else (0, 0, 255)  # Green vs Red
            cv2.rectangle(display, (0, 0), (w, h), color, 10)

            # Text Stats
            status_text = f"REC | Score: {score}" if is_recording else "IDLE (No Score)"
            cv2.putText(display, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display, f"Cache: {len(data_cache)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            cv2.putText(display, f"Saved: {total_frames_saved}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            cv2.imshow("Manual Collector", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 5. Periodic Save
            if len(data_cache) >= SAVE_INTERVAL:
                timestamp = int(time.time())
                save_path = os.path.join(DATA_DIR, f"chunk_{timestamp}.npy")
                np.save(save_path, np.array(data_cache, dtype=object))
                print(f"[SAVE] Dumped {len(data_cache)} frames to disk.")
                total_frames_saved += len(data_cache)
                data_cache = []

            # 6. FPS Lock
            # We sleep to force the loop to run at ~15Hz
            # This effectively downsamples the high-speed VisionSystem stream
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_duration - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        # Final Save
        if data_cache:
            timestamp = int(time.time())
            np.save(os.path.join(DATA_DIR, f"chunk_{timestamp}.npy"), np.array(data_cache, dtype=object))
            print(f"[SAVE] Final dump of {len(data_cache)} frames.")

        vision.stop()
        cv2.destroyAllWindows()
        print(f"--- Session Complete. Total Frames: {total_frames_saved + len(data_cache)} ---")


if __name__ == "__main__":
    main()
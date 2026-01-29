# --- collect_data.py ---
import cv2
import numpy as np
import time
import os
import random
from collections import deque
from crossy_env import CrossyEnv

# --- CONFIG ---
DATA_DIR = "data"
SESSIONS_TO_RECORD = 10  # How many runs (deaths) to record
IMG_SIZE = 128  # Downscale to 128x128 for VAE
STACK_SIZE = 4  # History length
SAVE_INTERVAL = 500  # Save to disk every N steps to save RAM

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def process_frame(frame):
    """
    1. Grayscale
    2. Resize to 128x128
    3. Normalize 0-1
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized / 255.0


def main():
    env = CrossyEnv()

    frame_buffer = deque(maxlen=STACK_SIZE + 1)  # Stores t-3, t-2, t-1, t, t+1
    data_cache = []

    total_frames_saved = 0
    session_count = 0

    print(f"[RECORDER] Starting Collection for {SESSIONS_TO_RECORD} sessions...")
    print(f"[RECORDER] Focus the game window. Starting in 3 seconds...")
    time.sleep(3)

    while session_count < SESSIONS_TO_RECORD:
        print(f"[SESSION] {session_count + 1}/{SESSIONS_TO_RECORD} Started")

        # Reset Env
        _ = env.reset()
        frame_buffer.clear()
        step = 0
        done = False

        while not done:
            # 1. Grab Raw Frame
            raw_frame = env.grab_frame()
            if raw_frame is None: continue

            # 2. Process & Store
            processed = process_frame(raw_frame)
            frame_buffer.append(processed)

            # 3. Create Training Pair if buffer full
            # Input: [t-3, t-2, t-1, t] -> Target: [t+1]
            if len(frame_buffer) == STACK_SIZE + 1:
                # Convert deque to numpy array
                seq = np.array(frame_buffer, dtype=np.float32)

                # Split: Input Stack vs Next Frame Target
                input_stack = seq[:-1]  # Shape (4, 128, 128)
                target_frame = seq[-1]  # Shape (128, 128)

                data_cache.append((input_stack, target_frame))

            # 4. Visualization (Safety Check)
            cv2.imshow("Recorder Input (Current)", raw_frame)
            cv2.imshow("Recorder Processed", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

            # 5. Semi-Random Action (Biased Forward)
            # 0:Idle, 1:Up, 2:Left, 3:Right
            if step % 2 == 0:  # Act every few frames to simulate reaction time
                r = random.random()
                if r < 0.65:
                    action = 1  # 65% Forward
                elif r < 0.8:
                    action = 0  # 15% Idle
                elif r < 0.9:
                    action = 2  # 10% Left
                else:
                    action = 3  # 10% Right
            else:
                action = 0  # Idle

            _, _, done = env.step(action)
            step += 1

            # 6. Periodic Save
            if len(data_cache) >= SAVE_INTERVAL:
                timestamp = int(time.time())
                save_path = os.path.join(DATA_DIR, f"chunk_{timestamp}.npy")
                np.save(save_path, np.array(data_cache, dtype=object))
                print(f"[SAVE] Saved {len(data_cache)} sequences to {save_path}")
                total_frames_saved += len(data_cache)
                data_cache = []

        session_count += 1
        print(f"[SESSION] Finished. Total Frames: {total_frames_saved}")
        time.sleep(1.0)  # Wait for death animation

    # Flush remaining data
    if data_cache:
        timestamp = int(time.time())
        np.save(os.path.join(DATA_DIR, f"chunk_{timestamp}.npy"), np.array(data_cache, dtype=object))

    cv2.destroyAllWindows()
    print("[RECORDER] Collection Complete.")


if __name__ == "__main__":
    main()
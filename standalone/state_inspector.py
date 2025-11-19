# --- state_inspector.py ---

import cv2
import queue
import time
from vision import VisionSystem


def main():
    WINDOW_TITLE = "Crossy Road"
    print("--- Game State Inspector (Integrated) ---")
    print("Uses the exact same VisionSystem as the Agent.")
    print("Press 'q' to quit.")

    # Initialize Vision System
    vision_queue = queue.Queue(maxsize=1)
    # We disable the internal preview of vision.py so we can draw our own custom debug info here
    vision = VisionSystem(WINDOW_TITLE, vision_queue, show_preview=False)
    vision.start()

    # Logic State
    current_state = "UNKNOWN"
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            try:
                # Block briefly to sync with vision
                data = vision_queue.get(timeout=2.0)
            except queue.Empty:
                print("Waiting for vision data...")
                continue

            frame = data['frame']
            display = frame.copy()

            # --- Extract Data ---
            pause_visible = data['pause_visible']
            retry_visible = data['retry_visible']  # Still available as backup
            score = data['score']

            # --- Determine State ---
            # The Agent logic relies purely on pause_visible
            new_state = "PLAYING" if pause_visible else "DEAD/MENU"

            if new_state != current_state:
                print(f"[{time.strftime('%H:%M:%S')}] State Change: {current_state} -> {new_state}")
                current_state = new_state

            # --- Draw Inspector HUD ---
            # Opaque background
            cv2.rectangle(display, (0, 0), (400, 200), (0, 0, 0), -1)

            # 1. State
            color = (0, 255, 0) if current_state == "PLAYING" else (0, 0, 255)
            cv2.putText(display, f"STATE: {current_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # 2. Pause Button Signal (Fast)
            p_color = (0, 255, 0) if pause_visible else (0, 0, 255)
            cv2.putText(display, f"Pause Visible: {pause_visible}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, p_color, 2)

            # 3. Retry Button Signal (Slow)
            r_color = (0, 255, 0) if retry_visible else (100, 100, 100)
            cv2.putText(display, f"Retry Visible: {retry_visible}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_color,
                        2)

            # 4. Score
            cv2.putText(display, f"Score Read: {score}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 5. FPS
            frame_count += 1
            if time.time() - start_time > 1.0:
                fps = frame_count / (time.time() - start_time)
                print(f"Inspector FPS: {fps:.1f}", end='\r')
                frame_count = 0
                start_time = time.time()

            cv2.imshow("State Inspector", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
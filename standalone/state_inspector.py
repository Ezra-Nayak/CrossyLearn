# --- state_inspector.py ---

import cv2
import queue
import time
from vision import VisionSystem
from tracker import RamTracker


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

    ram_tracker = RamTracker()

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

            raw_state = ram_tracker.get_game_state()

            # --- Determine State (Now prioritizing RAM!) ---
            if raw_state is not None:
                is_alive = (raw_state == 1)
                new_state = "PLAYING" if is_alive else "DEAD/MENU"
            else:
                new_state = "PLAYING" if pause_visible else "DEAD/MENU"

            if new_state != current_state:
                print(
                    f"[{time.strftime('%H:%M:%S')}] State Change: {current_state} -> {new_state} (RAM Flag: {raw_state})")
                current_state = new_state

            # --- Draw Inspector HUD ---
            # Opaque background (Expanded for new data)
            cv2.rectangle(display, (0, 0), (450, 250), (0, 0, 0), -1)

            # 1. Logic State
            color = (0, 255, 0) if current_state == "PLAYING" else (0, 0, 255)
            cv2.putText(display, f"STATE: {current_state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # 2. RAM Flag (Event Driven)
            ram_color = (0, 255, 0) if raw_state == 1 else (0, 0, 255)
            ram_txt = f"RAM Flag: {raw_state}" if raw_state is not None else "RAM Flag: NULL"
            cv2.putText(display, ram_txt, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ram_color, 2)

            # 3. Vision Backup Signals
            p_color = (0, 255, 0) if pause_visible else (0, 0, 255)
            cv2.putText(display, f"Vis Pause: {pause_visible}", (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, p_color, 2)

            r_color = (0, 255, 0) if retry_visible else (100, 100, 100)
            cv2.putText(display, f"Vis Retry: {retry_visible}", (220, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_color, 2)

            # 4. RAM Tracker Coordinates & Z-Score
            coords = ram_tracker.get_coords()
            if coords:
                raw_x, raw_y, raw_z = coords
                grid_x = round(raw_x)
                grid_z = round(raw_z)

                cv2.putText(display, f"RAM Pos: X:{grid_x} | Z:{grid_z}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 0), 2)
                cv2.putText(display, f"Raw Y (Jump): {raw_y:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)
            else:
                cv2.putText(display, "RAM POS: INJECTING/SEARCHING...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

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
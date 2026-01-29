# --- check_tracker.py ---
import cv2
import queue
import time
import numpy as np
from vision import VisionSystem


def main():
    print("--- TRACKER STRESS TEST ---")
    print("1. Move the chicken continuously.")
    print("2. Watch for RED flashes or 'LOST' signals.")
    print("3. Press 'q' to quit.")

    # Initialize Vision
    vision_q = queue.Queue(maxsize=1)
    # We disable the internal preview to draw our own debug info
    vis = VisionSystem("Crossy Road", vision_q, show_preview=False)
    vis.start()

    lost_streak = 0
    last_known_pos = None

    try:
        while True:
            try:
                data = vision_q.get(timeout=1.0)
            except queue.Empty:
                continue

            frame = data['frame']
            pos = data['chicken_pos']
            box = data['chicken_box']  # (x, y, w, h)

            display = frame.copy()

            if pos:
                # --- LOCKED STATE ---
                if lost_streak > 0:
                    print(f"[RECOVERED] Was lost for {lost_streak} frames.")
                lost_streak = 0
                last_known_pos = pos

                # Draw Green Box
                if box:
                    x, y, w, h = box
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display, f"LOCKED ({w}x{h})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw Center Point
                cv2.circle(display, pos, 5, (0, 0, 255), -1)

            else:
                # --- LOST STATE ---
                lost_streak += 1

                # Visual Warning
                cv2.putText(display, "LOST TRACKING", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # Draw Ghost (Last Known)
                if last_known_pos:
                    cv2.circle(display, last_known_pos, 5, (0, 255, 255), -1)
                    cv2.putText(display, "GHOST", (last_known_pos[0] + 10, last_known_pos[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Console Log for bad streaks
            # A hop usually takes 3-6 frames. If we lose track for >2 frames, it's an issue.
            if lost_streak > 2:
                print(f"\r[WARNING] Tracking Lost Streak: {lost_streak} frames", end="")

            cv2.imshow("Tracker Diagnostics", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        vis.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
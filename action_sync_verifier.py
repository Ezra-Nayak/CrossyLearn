import cv2
import time
import queue
import win32api
import numpy as np

# Import your existing vision system
from vision import VisionSystem

WINDOW_TITLE = "Crossy Road"
SLEEP_DURATION = 0.15  # Tweak this to see how it affects the frame the AI sees!

# Virtual Key Codes for Arrow Keys
VK_UP = 0x26
VK_LEFT = 0x25
VK_RIGHT = 0x27


def is_key_pressed(vk_code):
    # High bit is 1 if the key is currently being pressed down
    return (win32api.GetAsyncKeyState(vk_code) & 0x8000) != 0


def main():
    print(f"--- Action Sync Verifier ---")
    print(f"Testing Sleep Duration: {SLEEP_DURATION}s")
    print("1. Click into Crossy Road and play normally with Arrow Keys.")
    print("2. Watch the Agent View window. It will show you the exact frame the AI receives AFTER the jump.")
    print("3. Press 'q' on the Agent View window to quit.\n")

    # Start Vision Queue
    vision_q = queue.Queue(maxsize=1)
    vis = VisionSystem(WINDOW_TITLE, vision_q, show_preview=False)
    vis.start()

    # Track key states to prevent holding down from triggering multiple times
    last_state = {VK_UP: False, VK_LEFT: False, VK_RIGHT: False}

    try:
        while True:
            # 1. Grab normal passive frame (What the camera sees continuously)
            try:
                data = vision_q.get(timeout=1.0)
                frame = data['frame']
            except queue.Empty:
                continue

            # 2. Check if the human pressed an arrow key
            action_triggered = None
            for key, name in [(VK_UP, 'UP'), (VK_LEFT, 'LEFT'), (VK_RIGHT, 'RIGHT')]:
                pressed = is_key_pressed(key)
                if pressed and not last_state[key]:
                    action_triggered = name
                last_state[key] = pressed

            # 3. If a key was pressed, simulate the Agent's perspective!
            if action_triggered:
                print(f"[{action_triggered}] pressed! Simulating {SLEEP_DURATION}s jump animation...")

                # --- THIS IS EXACTLY WHAT THE AGENT DOES ---
                # A. Wait for the jump animation to finish
                time.sleep(SLEEP_DURATION)

                # B. Flush the queue of all the blurry mid-air frames
                flushed_count = 0
                while not vision_q.empty():
                    vision_q.get_nowait()
                    flushed_count += 1

                # C. Grab the pristine "landed" frame
                try:
                    data = vision_q.get(timeout=1.0)
                    agent_frame = data['frame']
                    print(f"-> Flushed {flushed_count} mid-air frames. AI receives this landed frame.")
                except queue.Empty:
                    agent_frame = frame  # Fallback

                # D. Render the Agent's perspective clearly for the human
                # Draw a green border and text so you know this is the post-jump frame
                h, w = agent_frame.shape[:2]
                cv2.rectangle(agent_frame, (0, 0), (w, h), (0, 255, 0), 8)
                cv2.putText(agent_frame, "AI SEES THIS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # Freeze the frame on screen for 0.75 seconds so your human eyes can study it
                cv2.imshow("Agent View", agent_frame)
                cv2.waitKey(750)

            else:
                # Normal passive viewing
                cv2.imshow("Agent View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        vis.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
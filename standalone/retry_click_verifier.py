# --- retry_click_verifier.py ---

import time
import win32gui
import pydirectinput
import queue
import cv2
from vision import VisionSystem

# --- CONFIGURATION ---
WINDOW_TITLE = "Crossy Road"
# The center point calculated from your ROI (703, 866, 128, 92)
RETRY_CLICK_COORDS = (767, 912)


def click_retry_button(hwnd):
    """
    Performs the click logic exactly as defined in the Environment.
    """
    if not hwnd or not win32gui.IsWindow(hwnd):
        return

    try:
        # 1. Get the Screen Coordinates of the Game's Top-Left Corner
        left, top, _, _ = win32gui.GetClientRect(hwnd)
        # ClientToScreen converts (0,0) of the client area to Global Screen Coords
        client_point = win32gui.ClientToScreen(hwnd, (left, top))

        # 2. Calculate Target
        # We add the Title Bar offset (+50) here because we included it in crossy_env.py
        # If the click is too low, remove the +50.
        target_x = client_point[0] + RETRY_CLICK_COORDS[0]
        target_y = client_point[1] + RETRY_CLICK_COORDS[1] + 50

        print(f"Clicking at Screen Coords: ({target_x}, {target_y})")

        # 3. Perform Click
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.1)
        pydirectinput.moveTo(target_x, target_y)
        pydirectinput.click()

    except Exception as e:
        print(f"Click failed: {e}")


def main():
    print("--- Retry Button Verifier ---")
    print("1. Play the game manually.")
    print("2. Die on purpose.")
    print("3. Wait for this script to click 'Retry'.")
    print("Press 'q' in the console to quit.")

    vision_queue = queue.Queue(maxsize=1)
    # Disable preview to keep it lightweight, we just need logic
    vision = VisionSystem(WINDOW_TITLE, vision_queue, show_preview=False)
    vision.start()

    last_state_alive = True
    death_timer = 0
    COOLDOWN = 2.0  # Wait 2 seconds after death before clicking

    try:
        while True:
            # Get Vision Data
            try:
                data = vision_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            is_alive = data['pause_visible']

            # State Machine
            if is_alive:
                if not last_state_alive:
                    print("STATUS: Playing (Alive)")
                last_state_alive = True
                death_timer = 0

            else:
                # We are DEAD or in MENU
                if last_state_alive:
                    print("STATUS: Death Detected! Waiting for button...")
                    death_timer = time.time()
                    last_state_alive = False

                # Check if enough time passed for animation to finish
                if death_timer > 0 and (time.time() - death_timer > COOLDOWN):
                    print("ACTION: Attempting Auto-Retry...")

                    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
                    click_retry_button(hwnd)

                    # Wait a bit for the click to register and menu to fade
                    time.sleep(0.5)
                    print("ACTION: Pressing Space...")
                    pydirectinput.press('space')

                    # Reset timer so we don't spam click while loading
                    death_timer = time.time() + 5.0

    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()


if __name__ == "__main__":
    main()
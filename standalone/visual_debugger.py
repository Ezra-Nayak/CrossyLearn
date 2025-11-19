# --- vision_debugger.py ---

import cv2
import numpy as np
import mss
import win32gui
import time
import pywintypes
import math
import random
import os

# --- CORE PARAMETERS TO DEBUG ---
# These are the exact values our agent uses. We are testing this configuration.
WINDOW_TITLE = "Crossy Road"
LOWER_BOUND = np.array([170, 125, 21])
UPPER_BOUND = np.array([179, 136, 37])
AREA_MIN = 0
AREA_MAX = 144
SEARCH_ZONE_Y_INTERCEPT = 310
LINE_ANGLE_DEG = 15


# ------------------------------------

def main():
    print("--- Ground-Truth Vision Debugger ---")
    print("This tool shows the raw output of the vision pipeline.")
    print("Every colored shape is a contour. Its area is printed next to it.")
    print("The console shows detailed data for every contour found.")
    print("\nACTION: Make the chicken hop and observe the console output.")
    print("Press 'q' to quit.")

    hwnd = None
    with mss.mss() as sct:
        while True:
            try:
                if not hwnd or not win32gui.IsWindow(hwnd):
                    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
                    if not hwnd: print("Waiting for game window...", end='\r'); time.sleep(0.5); continue

                # --- Screen Capture ---
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                TITLE_BAR_HEIGHT = 50
                monitor = {"top": client_top + TITLE_BAR_HEIGHT, "left": client_left, "width": right - left,
                           "height": bottom - top - TITLE_BAR_HEIGHT}
                if monitor['width'] <= 0 or monitor['height'] <= 0: time.sleep(0.1); continue

                frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
                display_frame = frame.copy()
                frame_h, frame_w, _ = display_frame.shape

                # --- SOTA Angled Mask Logic ---
                angle_rad = math.radians(LINE_ANGLE_DEG)
                slope = math.tan(angle_rad)
                search_y1 = SEARCH_ZONE_Y_INTERCEPT
                search_y2 = int(slope * frame_w + search_y1)
                search_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
                pts = np.array([[0, search_y1], [frame_w, search_y2], [frame_w, frame_h], [0, frame_h]], dtype=np.int32)
                cv2.fillPoly(search_mask, [pts], 255)

                # --- Raw Vision Pipeline ---
                search_area = cv2.bitwise_and(frame, frame, mask=search_mask)
                hsv_frame = cv2.cvtColor(search_area, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_frame, LOWER_BOUND, UPPER_BOUND)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # --- Ground-Truth Visualization and Logging ---
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear console for clean output
                print(f"--- Frame Analysis @ {time.time():.2f} ---")
                print(f"Found {len(contours)} raw contours.")

                if not contours:
                    print("No contours of the target color found in the search area.")

                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)

                    # Log to console
                    print(f"  Contour {i}: Area={area:.1f}, BBox=[{x},{y},{w},{h}]")

                    # Visualize on screen
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    cv2.drawContours(display_frame, [contour], -1, color, 1)
                    cv2.putText(display_frame, f"A:{int(area)}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                cv2.imshow('Vision Debugger', display_frame)
                cv2.imshow('Mask', mask)

            except (pywintypes.error, win32gui.error):
                hwnd = None;
                print("Game window lost! Waiting...", end='\r');
                time.sleep(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
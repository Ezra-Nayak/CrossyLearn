# --- pause_roi_debugger.py ---

import cv2
import numpy as np
import mss
import win32gui
import time

WINDOW_TITLE = "Crossy Road"
DEBUG_WINDOW = "Pause Button Debugger"
MASK_WINDOW = "Pause Threshold Mask"


def nothing(x):
    pass


def find_pause_button_roi():
    """
    Captures the game screen and uses geometric contour filtering to find
    the two vertical bars of the pause button in the top right corner.
    """
    print(f"Searching for '{WINDOW_TITLE}'...")

    cv2.namedWindow(DEBUG_WINDOW)
    cv2.namedWindow(MASK_WINDOW)

    # Trackbar to tune the "Whiteness" threshold.
    # Pause button is usually pure white (255), but compression/rendering might lower it.
    cv2.createTrackbar('White Thresh', DEBUG_WINDOW, 254, 255, nothing)

    # Trackbar to adjust the size of the search zone (percentage of screen width)
    cv2.createTrackbar('Search Width %', DEBUG_WINDOW, 8, 50, nothing)

    with mss.mss() as sct:
        while True:
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if not hwnd:
                time.sleep(1)
                continue

            try:
                # 1. Get Window Geometry
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                TITLE_BAR_HEIGHT = 50
                monitor = {
                    "top": client_top + TITLE_BAR_HEIGHT, "left": client_left,
                    "width": right - left, "height": bottom - top - TITLE_BAR_HEIGHT
                }

                if monitor['width'] <= 0 or monitor['height'] <= 0:
                    continue

                # 2. Grab Frame
                game_frame = np.array(sct.grab(monitor))
                game_frame_bgr = cv2.cvtColor(game_frame, cv2.COLOR_BGRA2BGR)
                h, w, _ = game_frame_bgr.shape
                display_frame = game_frame_bgr.copy()

                # 3. Define ROI (Top Right Corner) based on Trackbar
                # We look at the top 10% of height, and the right X% of width
                search_w_pct = cv2.getTrackbarPos('Search Width %', DEBUG_WINDOW) / 100.0
                thresh_val = cv2.getTrackbarPos('White Thresh', DEBUG_WINDOW)

                roi_w = int(w * search_w_pct)
                roi_h = int(h * 0.06)  # Top 10% vertical

                # ROI Coordinates: x1, y1, x2, y2
                roi_x1 = w - roi_w
                roi_y1 = 110  # Small buffer from top edge
                roi_x2 = w - 20  # Small buffer from right edge
                roi_y2 = roi_y1 + roi_h

                # Extract ROI
                if roi_x1 < 0: roi_x1 = 0
                roi_frame = game_frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

                # 4. Processing Logic (The Pause Detection)
                gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                valid_bars = 0

                # Draw the Search Zone Box (Blue)
                cv2.rectangle(display_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
                cv2.putText(display_frame, "Search Zone", (roi_x1, roi_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1)

                # Analyze Contours in the ROI
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < 10: continue  # Ignore tiny noise

                    x, y, cw, ch = cv2.boundingRect(c)

                    # Calculate Geometric Properties
                    aspect_ratio = float(ch) / cw if cw > 0 else 0

                    # Shift coordinates back to global frame for drawing
                    global_x = roi_x1 + x
                    global_y = roi_y1 + y

                    # Check: Is it a vertical bar?
                    # Aspect ratio > 1.5 (Tall) and Area > 15 (Not dust)
                    is_bar = (aspect_ratio > 1.5 and area > 15)

                    if is_bar:
                        valid_bars += 1
                        # Draw Green Box (Confirmed Bar)
                        cv2.rectangle(display_frame, (global_x, global_y), (global_x + cw, global_y + ch), (0, 255, 0),
                                      2)
                    else:
                        # Draw Red Box (Rejected Noise)
                        cv2.rectangle(display_frame, (global_x, global_y), (global_x + cw, global_y + ch), (0, 0, 255),
                                      1)

                # 5. Final Decision
                pause_detected = (valid_bars == 2)

                # Status Text
                status_color = (0, 255, 0) if pause_detected else (0, 0, 255)
                status_text = f"PAUSE BUTTON: {'DETECTED' if pause_detected else 'MISSING'}"
                cv2.putText(display_frame, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                # Show bar count
                cv2.putText(display_frame, f"Bars Found: {valid_bars}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)

                cv2.imshow(DEBUG_WINDOW, display_frame)
                cv2.imshow(MASK_WINDOW, mask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_pause_button_roi()
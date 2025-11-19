# --- terrain_tuner.py (v2 - Crash Resistant & Feature Rich) ---

import cv2
import numpy as np
import mss
import win32gui
import time
import pywintypes
from terrain_analyzer import Terrain, TERRAIN_HSV_RANGES

WINDOW_TITLE = "Crossy Road"
TUNER_WINDOW = "Terrain Tuner"
MASK_WINDOW = "Mask"
CONTROLS_WINDOW = "Controls"

# --- Globals for Mouse Callback ---
last_click_info = {}
CLICK_INFO_DURATION = 3.0  # seconds


def on_mouse_click(event, x, y, flags, param):
    """Callback to capture pixel info on click."""
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        if frame is not None and 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = hsv_frame[y, x]

            global last_click_info
            last_click_info = {
                "hsv": (h, s, v),
                "timestamp": time.time()
            }


def nothing(x):
    pass


def main():
    print("--- Advanced Terrain Tuner v2 ---")
    print("Instructions:")
    print(" - Adjust sliders to isolate a terrain type in the 'Mask' window.")
    print(" - Click on the 'Terrain Tuner' window to inspect a pixel's HSV value.")
    print(" - Press 'n' to cycle to the NEXT terrain type.")
    print(" - Press 's' to SAVE the current slider values for the selected terrain.")
    print(" - Press 'q' to quit and print the final dictionary.")
    print("----------------------------------------------------------------------")

    # Create windows
    cv2.namedWindow(TUNER_WINDOW)
    cv2.namedWindow(MASK_WINDOW)
    cv2.namedWindow(CONTROLS_WINDOW)
    cv2.resizeWindow(CONTROLS_WINDOW, 400, 300)

    # Create trackbars
    cv2.createTrackbar('H_min', CONTROLS_WINDOW, 0, 179, nothing)
    cv2.createTrackbar('H_max', CONTROLS_WINDOW, 179, 179, nothing)
    cv2.createTrackbar('S_min', CONTROLS_WINDOW, 0, 255, nothing)
    cv2.createTrackbar('S_max', CONTROLS_WINDOW, 255, 255, nothing)
    cv2.createTrackbar('V_min', CONTROLS_WINDOW, 0, 255, nothing)
    cv2.createTrackbar('V_max', CONTROLS_WINDOW, 255, 255, nothing)

    # --- Core Logic ---
    tuned_ranges = {k: [list(v[0]), list(v[1])] for k, v in TERRAIN_HSV_RANGES.items()}
    terrain_types = list(tuned_ranges.keys())
    current_terrain_idx = 0

    save_feedback_timer = 0
    SAVE_FEEDBACK_DURATION = 1.5

    def update_sliders_for_current_terrain():
        terrain_key = terrain_types[current_terrain_idx]
        lower, upper = tuned_ranges[terrain_key]
        cv2.setTrackbarPos('H_min', CONTROLS_WINDOW, lower[0])
        cv2.setTrackbarPos('S_min', CONTROLS_WINDOW, lower[1])
        cv2.setTrackbarPos('V_min', CONTROLS_WINDOW, lower[2])
        cv2.setTrackbarPos('H_max', CONTROLS_WINDOW, upper[0])
        cv2.setTrackbarPos('S_max', CONTROLS_WINDOW, upper[1])
        cv2.setTrackbarPos('V_max', CONTROLS_WINDOW, upper[2])
        print(f"Now tuning: {terrain_key.name}")

    update_sliders_for_current_terrain()

    hwnd = None
    with mss.mss() as sct:
        while True:
            try:
                if not hwnd or not win32gui.IsWindow(hwnd):
                    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
                    if not hwnd:
                        print("Waiting for game window...", end='\r')
                        time.sleep(0.5)
                        continue

                # Screen capture
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                TITLE_BAR_HEIGHT = 50
                monitor = {
                    "top": client_top + TITLE_BAR_HEIGHT, "left": client_left,
                    "width": right - left, "height": bottom - top - TITLE_BAR_HEIGHT
                }

                if monitor['width'] <= 0 or monitor['height'] <= 0:
                    time.sleep(0.1)
                    continue

                game_frame_bgra = np.array(sct.grab(monitor))
                display_frame = cv2.cvtColor(game_frame_bgra, cv2.COLOR_BGRA2BGR)

                # Set mouse callback with the latest frame
                cv2.setMouseCallback(TUNER_WINDOW, on_mouse_click, display_frame)

                # Get current slider values and create mask
                h_min, h_max, s_min, s_max, v_min, v_max = (
                    cv2.getTrackbarPos('H_min', CONTROLS_WINDOW), cv2.getTrackbarPos('H_max', CONTROLS_WINDOW),
                    cv2.getTrackbarPos('S_min', CONTROLS_WINDOW), cv2.getTrackbarPos('S_max', CONTROLS_WINDOW),
                    cv2.getTrackbarPos('V_min', CONTROLS_WINDOW), cv2.getTrackbarPos('V_max', CONTROLS_WINDOW)
                )
                hsv_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_frame, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
                cv2.imshow(MASK_WINDOW, mask)

                # --- Visualization ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(display_frame, (5, 5), (650, 80), (0, 0, 0), -1)  # Info background

                info_text = f"TUNING: {terrain_types[current_terrain_idx].name}"
                cv2.putText(display_frame, info_text, (10, 30), font, 0.8, (0, 255, 255), 2)
                hsv_text = f"H:[{h_min}-{h_max}] S:[{s_min}-{s_max}] V:[{v_min}-{v_max}]"
                cv2.putText(display_frame, hsv_text, (10, 60), font, 0.7, (255, 255, 255), 1)

                if time.time() < save_feedback_timer:
                    cv2.putText(display_frame, "SAVED!", (display_frame.shape[1] - 150, 40), font, 1.2, (0, 255, 0), 3)

                # Click info overlay
                if last_click_info and time.time() - last_click_info.get("timestamp", 0) < CLICK_INFO_DURATION:
                    h, s, v = last_click_info["hsv"]
                    click_text = f"Clicked HSV: ({h}, {s}, {v})"
                    text_size = cv2.getTextSize(click_text, font, 0.7, 2)[0]
                    cv2.rectangle(display_frame, (5, display_frame.shape[0] - 40),
                                  (15 + text_size[0], display_frame.shape[0] - 10), (0, 0, 0), -1)
                    cv2.putText(display_frame, click_text, (10, display_frame.shape[0] - 20), font, 0.7, (50, 150, 255),
                                2)

                cv2.imshow(TUNER_WINDOW, display_frame)

            except (pywintypes.error, win32gui.error) as e:
                # This catches the "Invalid window handle" error
                if e.winerror == 1400:
                    print("Game window lost! Waiting for it to reappear...", end='\r')
                    hwnd = None
                    # Also clear the tuner window to indicate a disconnect
                    cv2.imshow(TUNER_WINDOW, np.zeros((600, 800, 3), dtype=np.uint8))
                    time.sleep(1)
                else:
                    raise e  # Re-raise other errors

            # --- Key Handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_terrain_idx = (current_terrain_idx + 1) % len(terrain_types)
                update_sliders_for_current_terrain()
            elif key == ord('s'):
                current_terrain_key = terrain_types[current_terrain_idx]
                tuned_ranges[current_terrain_key] = [[h_min, s_min, v_min], [h_max, s_max, v_max]]
                save_feedback_timer = time.time() + SAVE_FEEDBACK_DURATION
                print(f"Saved values for {current_terrain_key.name}")

    cv2.destroyAllWindows()

    # --- Print Final Results ---
    print("\n\n--- Advanced Tuning Complete ---")
    print("Copy the following dictionary into your terrain_analyzer.py file:\n")
    print("TERRAIN_HSV_RANGES = {")
    for terrain_key, (lower, upper) in sorted(tuned_ranges.items(), key=lambda item: item[0].value):
        print(f"    Terrain.{terrain_key.name}: ({lower}, {upper}),")
    print("}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
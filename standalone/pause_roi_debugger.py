# --- pause_roi_debugger.py ---

import cv2
import numpy as np
import mss
import win32gui
import time

WINDOW_TITLE = "Crossy Road"
DEBUG_WINDOW = "Pause Patch Debugger"


def nothing(x):
    pass


def tune_pause_patches():
    """
    Captures the game screen and uses a dual-patch sampling method
    to detect the pause button. Use the trackbars to move the patches
    over the left and right bars of the pause button.
    """
    print(f"Searching for '{WINDOW_TITLE}'...")

    cv2.namedWindow(DEBUG_WINDOW)

    # Trackbars to tune offsets from top-right corner
    cv2.createTrackbar('White Thresh', DEBUG_WINDOW, 250, 255, nothing)
    cv2.createTrackbar('L-Bar X Dist', DEBUG_WINDOW, 70, 300, nothing)
    cv2.createTrackbar('L-Bar Y Dist', DEBUG_WINDOW, 180, 500, nothing)
    cv2.createTrackbar('R-Bar X Dist', DEBUG_WINDOW, 46, 300, nothing)
    cv2.createTrackbar('R-Bar Y Dist', DEBUG_WINDOW, 160, 500, nothing)
    cv2.createTrackbar('Patch Size', DEBUG_WINDOW, 8, 50, nothing)

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

                # 3. Get Trackbar Values
                thresh_val = cv2.getTrackbarPos('White Thresh', DEBUG_WINDOW)
                lx_dist = cv2.getTrackbarPos('L-Bar X Dist', DEBUG_WINDOW)
                ly_dist = cv2.getTrackbarPos('L-Bar Y Dist', DEBUG_WINDOW)
                rx_dist = cv2.getTrackbarPos('R-Bar X Dist', DEBUG_WINDOW)
                ry_dist = cv2.getTrackbarPos('R-Bar Y Dist', DEBUG_WINDOW)
                patch_size = cv2.getTrackbarPos('Patch Size', DEBUG_WINDOW)
                if patch_size < 1: patch_size = 1

                # Calculate absolute coordinates from Top-Right
                lx1, lx2 = w - lx_dist, w - lx_dist + patch_size
                ly1, ly2 = ly_dist, ly_dist + patch_size

                rx1, rx2 = w - rx_dist, w - rx_dist + patch_size
                ry1, ry2 = ry_dist, ry_dist + patch_size

                # Bounds check
                if lx1 < 0 or rx1 < 0 or ly2 > h or ry2 > h:
                    continue

                # 4. Processing Logic (Dual-Patch Detection)
                left_patch = game_frame_bgr[ly1:ly2, lx1:lx2]
                right_patch = game_frame_bgr[ry1:ry2, rx1:rx2]

                left_gray = cv2.cvtColor(left_patch, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_patch, cv2.COLOR_BGR2GRAY)

                _, left_thresh = cv2.threshold(left_gray, thresh_val, 255, cv2.THRESH_BINARY)
                _, right_thresh = cv2.threshold(right_gray, thresh_val, 255, cv2.THRESH_BINARY)

                left_score = np.count_nonzero(left_thresh) / (patch_size * patch_size)
                right_score = np.count_nonzero(right_thresh) / (patch_size * patch_size)

                # 5. Final Decision
                pause_detected = (left_score > 0.8) and (right_score > 0.8)

                # Draw Visuals
                color_left = (0, 255, 0) if left_score > 0.8 else (0, 0, 255)
                color_right = (0, 255, 0) if right_score > 0.8 else (0, 0, 255)

                # Draw patches
                cv2.rectangle(display_frame, (lx1, ly1), (lx2, ly2), color_left, 2)
                cv2.rectangle(display_frame, (rx1, ry1), (rx2, ry2), color_right, 2)

                # Draw labels
                cv2.putText(display_frame, f"L: {left_score * 100:.0f}%", (lx1 - 50, ly1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_left, 2)
                cv2.putText(display_frame, f"R: {right_score * 100:.0f}%", (rx1 - 10, ry1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_right, 2)

                # 6. Zoomed Inset for Visibility
                inset_w, inset_h = 200, 300
                inset_x1 = max(0, w - inset_w)
                # Crop from the top right
                inset_crop = display_frame[0:min(h, inset_h), inset_x1:w]

                # Scale it up 2x
                scale = 2
                inset_zoomed = cv2.resize(inset_crop, (inset_w * scale, min(h, inset_h) * scale),
                                          interpolation=cv2.INTER_NEAREST)
                zh, zw, _ = inset_zoomed.shape

                # Overlay safely on top left
                if h >= zh and w >= zw:
                    display_frame[0:zh, 0:zw] = inset_zoomed
                    cv2.rectangle(display_frame, (0, 0), (zw, zh), (255, 255, 255), 2)

                    # Text Drop Shadow
                    cv2.putText(display_frame, "ZOOMED VIEW (Top-Right)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 0), 3)
                    cv2.putText(display_frame, "ZOOMED VIEW (Top-Right)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                # Status Text
                status_color = (0, 255, 0) if pause_detected else (0, 0, 255)
                status_text = f"PAUSE BUTTON: {'DETECTED' if pause_detected else 'MISSING'}"
                cv2.putText(display_frame, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                cv2.imshow(DEBUG_WINDOW, display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    tune_pause_patches()
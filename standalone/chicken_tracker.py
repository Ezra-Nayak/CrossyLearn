import cv2
import numpy as np
import mss
import win32gui
import time
import math


def nothing(x):
    pass


def main():
    WINDOW_TITLE = "Crossy Road"
    TUNER_WINDOW = "Chicken Vision Tuner"
    MASK_WINDOW = "HSV Mask"
    CONTROLS_WINDOW = "Controls"

    cv2.namedWindow(TUNER_WINDOW)
    cv2.namedWindow(MASK_WINDOW)
    cv2.namedWindow(CONTROLS_WINDOW)
    cv2.resizeWindow(CONTROLS_WINDOW, 400, 600)

    # --- DEFAULTS (Current Config) ---
    # Hue: 170-179 (Red wraps around, so this is high red)
    cv2.createTrackbar('H_min', CONTROLS_WINDOW, 170, 179, nothing)
    cv2.createTrackbar('H_max', CONTROLS_WINDOW, 179, 179, nothing)
    cv2.createTrackbar('S_min', CONTROLS_WINDOW, 125, 255, nothing)
    cv2.createTrackbar('S_max', CONTROLS_WINDOW, 136, 255, nothing)
    cv2.createTrackbar('V_min', CONTROLS_WINDOW, 21, 255, nothing)
    cv2.createTrackbar('V_max', CONTROLS_WINDOW, 37, 255, nothing)

    # Area: Lower this if the "squished" chicken is being ignored
    cv2.createTrackbar('Area_min', CONTROLS_WINDOW, 0, 500, nothing)
    cv2.createTrackbar('Area_max', CONTROLS_WINDOW, 500, 1000, nothing)

    # Geometry
    cv2.createTrackbar('Y_Intercept', CONTROLS_WINDOW, 310, 1000, nothing)
    cv2.createTrackbar('Angle', CONTROLS_WINDOW, 15, 45, nothing)

    print("--- Chicken Tuner ---")
    print("1. Play the game.")
    print("2. Adjust sliders until the Chicken is detected even during the HOP (Squish).")
    print("3. Make sure 'Area_min' is low enough to catch the small squished sprite.")

    with mss.mss() as sct:
        while True:
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if not hwnd:
                time.sleep(0.5)
                continue

            # Capture
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            client_point = win32gui.ClientToScreen(hwnd, (left, top))
            TITLE_BAR_HEIGHT = 50
            monitor = {
                "top": client_point[1] + TITLE_BAR_HEIGHT,
                "left": client_point[0],
                "width": right - left,
                "height": bottom - top - TITLE_BAR_HEIGHT
            }
            if monitor['width'] <= 0: continue

            frame = np.array(sct.grab(monitor))
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            h, w, _ = display_frame.shape

            # Get Sliders
            h_min = cv2.getTrackbarPos('H_min', CONTROLS_WINDOW)
            h_max = cv2.getTrackbarPos('H_max', CONTROLS_WINDOW)
            s_min = cv2.getTrackbarPos('S_min', CONTROLS_WINDOW)
            s_max = cv2.getTrackbarPos('S_max', CONTROLS_WINDOW)
            v_min = cv2.getTrackbarPos('V_min', CONTROLS_WINDOW)
            v_max = cv2.getTrackbarPos('V_max', CONTROLS_WINDOW)
            area_min = cv2.getTrackbarPos('Area_min', CONTROLS_WINDOW)
            area_max = cv2.getTrackbarPos('Area_max', CONTROLS_WINDOW)
            y_intercept = cv2.getTrackbarPos('Y_Intercept', CONTROLS_WINDOW)
            angle_deg = cv2.getTrackbarPos('Angle', CONTROLS_WINDOW)

            # --- 1. Angled Mask (Same as Env) ---
            angle_rad = math.radians(angle_deg)
            slope = math.tan(angle_rad)
            search_y2 = int(slope * w + y_intercept)

            search_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
            pts = np.array([[0, y_intercept], [w, search_y2], [w, h], [0, h]], dtype=np.int32)
            cv2.fillPoly(search_mask, [pts], 255)

            masked_frame = cv2.bitwise_and(display_frame, display_frame, mask=search_mask)

            # --- 2. Color Threshold ---
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv, lower, upper)

            # --- 3. Contour Detection ---
            contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw Search Line
            cv2.line(display_frame, (0, y_intercept), (w, search_y2), (255, 255, 0), 2)

            found = False
            if contours:
                # Filter by Area
                valid = [c for c in contours if area_min < cv2.contourArea(c) < area_max]
                if valid:
                    c = max(valid, key=cv2.contourArea)
                    cx, cy, cw, ch = cv2.boundingRect(c)

                    # Draw Box
                    cv2.rectangle(display_frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Area: {int(cv2.contourArea(c))}", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    found = True

            if not found:
                cv2.putText(display_frame, "LOST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(TUNER_WINDOW, display_frame)
            cv2.imshow(MASK_WINDOW, color_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print("\n--- COPY THESE VALUES TO CROSSY_ENV.PY ---")
    print(f"LOWER_BOUND = np.array([{h_min}, {s_min}, {v_min}])")
    print(f"UPPER_BOUND = np.array([{h_max}, {s_max}, {v_max}])")
    print(f"AREA_MIN = {area_min}  # Make sure to update the logic in find_chicken")
    print(f"SEARCH_ZONE_Y_INTERCEPT = {y_intercept}")
    print("------------------------------------------")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
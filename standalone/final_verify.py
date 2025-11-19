import cv2
import numpy as np
import mss
import win32gui
import time
import math


def find_chicken(frame, color_lower, color_upper, area_min, area_max):
    """
    Finds the player character within a given frame using tuned parameters.
    Returns the bottom-center (x, y) coordinates and the bounding box.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, color_lower, color_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    valid_contours = [c for c in contours if area_min < cv2.contourArea(c) < area_max]
    if not valid_contours:
        return None, None

    chicken_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(chicken_contour)

    chicken_pos = (x + w // 2, y + h)
    return chicken_pos, (x, y, w, h)


def main():
    WINDOW_TITLE = "Crossy Road"
    VERIFIER_WINDOW = "Final Verifier"

    # --- YOUR FINAL TUNED VALUES ---
    LOWER_BOUND = np.array([170, 125, 21])
    UPPER_BOUND = np.array([179, 136, 37])
    AREA_MIN = 0
    AREA_MAX = 144
    SEARCH_ZONE_Y_INTERCEPT = 310
    PENALTY_LINE_Y_INTERCEPT = 850
    LINE_ANGLE_DEG = 15
    # -----------------------------

    print("Starting Final Verifier. Play the game and confirm all logic is correct.")
    print("Press 'q' to exit.")

    with mss.mss() as sct:
        while True:
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if not hwnd: continue

            # --- Screen Capture ---
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
            client_right, client_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
            TITLE_BAR_HEIGHT = 50
            monitor = {"top": client_top + TITLE_BAR_HEIGHT, "left": client_left, "width": client_right - client_left,
                       "height": client_bottom - client_top - TITLE_BAR_HEIGHT}
            if monitor['width'] <= 0 or monitor['height'] <= 0: continue

            game_frame = np.array(sct.grab(monitor))
            display_frame = cv2.cvtColor(game_frame, cv2.COLOR_BGRA2BGR)
            frame_h, frame_w, _ = display_frame.shape

            # --- Define Line Parameters ---
            angle_rad = math.radians(LINE_ANGLE_DEG)
            slope = math.tan(angle_rad)

            # --- Define Angled Search Zone ---
            search_x1, search_y1 = 0, SEARCH_ZONE_Y_INTERCEPT
            search_x2, search_y2 = frame_w, int(slope * frame_w + SEARCH_ZONE_Y_INTERCEPT)

            search_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
            pts = np.array([[search_x1, search_y1], [search_x2, search_y2], [frame_w, frame_h], [0, frame_h]],
                           dtype=np.int32)
            cv2.fillPoly(search_mask, [pts], 255)

            search_area_frame = cv2.bitwise_and(display_frame, display_frame, mask=search_mask)

            # --- Find the Chicken ---
            chicken_pos, chicken_box = find_chicken(search_area_frame, LOWER_BOUND, UPPER_BOUND, AREA_MIN, AREA_MAX)

            # --- Draw Visuals ---
            # Angled Search Line (Cyan)
            cv2.line(display_frame, (search_x1, search_y1), (search_x2, search_y2), (255, 255, 0), 2)
            # Angled Penalty Line (Red)
            penalty_x1, penalty_y1 = 0, PENALTY_LINE_Y_INTERCEPT
            penalty_x2, penalty_y2 = frame_w, int(slope * frame_w + PENALTY_LINE_Y_INTERCEPT)
            cv2.line(display_frame, (penalty_x1, penalty_y1), (penalty_x2, penalty_y2), (0, 0, 255), 2)

            # --- Check Penalty and Draw Info ---
            status_text = "STATUS: AWAITING CHICKEN"
            status_color = (255, 255, 0)

            if chicken_pos:
                cx, cy = chicken_pos
                x, y, w, h = chicken_box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Calculate the penalty line's y-value at the chicken's x-position
                penalty_line_y_at_chicken = int(slope * cx + PENALTY_LINE_Y_INTERCEPT)

                # If the chicken's feet are below the sloped line, it's a penalty
                if cy > penalty_line_y_at_chicken:
                    status_text = "STATUS: PENALTY ZONE"
                    status_color = (0, 0, 255)
                else:
                    status_text = "STATUS: SAFE"
                    status_color = (0, 255, 0)

            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.imshow(VERIFIER_WINDOW, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
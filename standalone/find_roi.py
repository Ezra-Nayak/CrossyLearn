import cv2
import numpy as np
import mss
import win32gui
import time


def find_static_element_roi():
    """
    Captures the game screen and searches for a static template to identify its ROI.
    """
    WINDOW_TITLE = "Crossy Road"
    TEMPLATE_PATH = 'templates/retry_button.png'

    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        print(f"Error: Could not load template at '{TEMPLATE_PATH}'")
        return

    t_h, t_w, _ = template.shape
    print(f"Template loaded. Dimensions (w, h): ({t_w}, {t_h})")
    print("Please get the game to the 'Game Over' screen to detect the button.")

    with mss.mss() as sct:
        while True:
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if not hwnd:
                print(f"'{WINDOW_TITLE}' window not found...")
                time.sleep(2)
                continue

            try:
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                client_right, client_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
                TITLE_BAR_HEIGHT = 50
                monitor = {
                    "top": client_top + TITLE_BAR_HEIGHT, "left": client_left,
                    "width": client_right - client_left, "height": (client_bottom - client_top) - TITLE_BAR_HEIGHT
                }

                if monitor['width'] <= 0 or monitor['height'] <= 0:
                    continue

                game_frame = np.array(sct.grab(monitor))
                game_frame_bgr = cv2.cvtColor(game_frame, cv2.COLOR_BGRA2BGR)

                # Search for the template in the entire frame
                res = cv2.matchTemplate(game_frame_bgr, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                display_frame = game_frame_bgr.copy()

                if max_val > 0.9:
                    top_left = max_loc
                    bottom_right = (top_left[0] + t_w, top_left[1] + t_h)

                    # Draw a rectangle around the found button
                    cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)

                    # Print the ROI tuple (x, y, w, h)
                    roi_tuple = (top_left[0], top_left[1], t_w, t_h)
                    print(f"SUCCESS: Button found! ROI (x, y, w, h): {roi_tuple}")

                    # Display the info on screen as well
                    text = f"ROI: {roi_tuple}"
                    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('ROI Finder', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"An error occurred: {e}")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_static_element_roi()
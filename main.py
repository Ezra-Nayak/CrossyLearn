import cv2
import numpy as np
import mss
import time
import win32gui
import win32ui
import os


def load_digit_templates():
    """Loads digit templates from the /templates directory."""
    templates = {}
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Template directory '{template_dir}' not found. Please run create_templates.py first.")

    for i in range(10):
        filepath = os.path.join(template_dir, f"{i}.png")
        if os.path.exists(filepath):
            templates[i] = cv2.imread(filepath, 0)  # Load in grayscale
    print(f"Loaded {len(templates)} digit templates.")
    return templates


def recognize_score(image, templates):
    """
    Recognizes the score using a dynamic, contour-based approach.
    """
    # Your confirmed, superior ROI
    SCORE_ROI = (10, 150, 10, 250)
    TEMPLATE_HEIGHT = 40  # The standardized height of our templates

    roi = image[SCORE_ROI[0]:SCORE_ROI[1], SCORE_ROI[2]:SCORE_ROI[3]]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)

    # cv2.imshow('Score ROI', thresh_roi) # Uncomment for debugging

    # 1. Find all potential digit contours in the ROI
    contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_digits = []

    # 2. For each contour, standardize it and then classify it
    for contour in contours:
        # Filter out very small contours that are likely noise
        if cv2.contourArea(contour) < 50:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Crop the detected digit from the thresholded ROI
        digit_crop = thresh_roi[y:y + h, x:x + w]

        # Standardize the cropped digit to match our template height
        aspect_ratio = w / h
        new_width = int(TEMPLATE_HEIGHT * aspect_ratio)
        # Ensure new_width is at least 1 to avoid cv2 error
        if new_width <= 0:
            continue
        standardized_digit = cv2.resize(digit_crop, (new_width, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA)

        # 3. Find the best match for this standardized digit among all templates
        best_match_score = -1
        best_match_digit = -1

        for digit_val, template in templates.items():
            # Pad the smaller image to match dimensions for template matching
            # This is a robust way to handle slight aspect ratio differences
            h_t, w_t = template.shape
            h_s, w_s = standardized_digit.shape

            # Create a black canvas of the larger of the two widths
            max_w = max(w_t, w_s)
            padded_template = np.zeros((TEMPLATE_HEIGHT, max_w), dtype=np.uint8)
            padded_digit = np.zeros((TEMPLATE_HEIGHT, max_w), dtype=np.uint8)

            # Place the images on the canvas
            padded_template[:, :w_t] = template
            padded_digit[:, :w_s] = standardized_digit

            res = cv2.matchTemplate(padded_digit, padded_template, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]

            if score > best_match_score:
                best_match_score = score
                best_match_digit = digit_val

        # 4. If the best match is confident enough, record it
        if best_match_score > 0.8:  # We can use a slightly lower threshold now
            detected_digits.append((best_match_digit, x))  # Use original x-coord for sorting

    if not detected_digits:
        return None

    # Sort detected digits by their x-coordinate
    detected_digits.sort(key=lambda d: d[1])

    # Build the final score string (no need for overlap filter with this method)
    score_str = "".join([str(d[0]) for d in detected_digits])

    return score_str if score_str else None


def main():
    """
    Main function to capture, display, and analyze the Crossy Road game window.
    """
    WINDOW_TITLE = "Crossy Road"
    TARGET_FPS = 60
    FRAME_DELAY = 1.0 / TARGET_FPS

    print("CrossyLearn Agent - Milestone 5: Corrected Score Recognition")
    print("-----------------------------------------------------------")

    templates = load_digit_templates()

    with mss.mss() as sct:
        while True:
            loop_start_time = time.time()

            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if not hwnd:
                print(f"'{WINDOW_TITLE}' window not found. Retrying...")
                time.sleep(2)
                continue

            try:
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                client_right, client_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))

                # Reverted to your confirmed, correct value
                TITLE_BAR_HEIGHT = 50

                monitor = {
                    "top": client_top + TITLE_BAR_HEIGHT,
                    "left": client_left,
                    "width": client_right - client_left,
                    "height": (client_bottom - client_top) - TITLE_BAR_HEIGHT
                }

                # Ensure monitor dimensions are valid
                if monitor['width'] <= 0 or monitor['height'] <= 0:
                    time.sleep(0.5)
                    continue

                img_bgra = sct.grab(monitor)
                game_frame = cv2.cvtColor(np.array(img_bgra), cv2.COLOR_BGRA2BGR)

                # --- Score Recognition ---
                score_val = recognize_score(game_frame.copy(), templates)
                display_score = score_val if score_val is not None else "N/A"

                # --- Display ---
                display_frame = game_frame.copy()

                # Dynamically center the score text
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"Score: {display_score}"
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                frame_width = display_frame.shape[1]
                text_x = (frame_width - text_size[0]) // 2

                cv2.putText(display_frame, text, (text_x, 40), font, 1, (0, 0, 255), 2)
                cv2.imshow('CrossyLearn Vision', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                elapsed_time = time.time() - loop_start_time
                sleep_duration = FRAME_DELAY - elapsed_time
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            except (win32ui.error, IndexError) as e:
                print(f"Window error: '{e}'. Retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

    print("Shutting down vision system.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
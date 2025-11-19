import cv2
import numpy as np
import mss
import win32gui
import time
import os


# --- UTILITY FUNCTIONS (Copied from main.py) ---

def load_digit_templates():
    """Loads digit templates from the 'templates' directory."""
    templates = {}
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        raise FileNotFoundError(
            f"Template directory '{template_dir}' not found. Make sure it's in the same folder as this script.")
    for i in range(10):
        filepath = os.path.join(template_dir, f"{i}.png")
        if os.path.exists(filepath):
            templates[i] = cv2.imread(filepath, 0)
    if not templates:
        raise FileNotFoundError("No digit templates (0.png, 1.png, etc.) found in the 'templates' directory.")
    print(f"Loaded {len(templates)} digit templates.")
    return templates


def recognize_score(image, templates, score_roi, threshold_val):
    """
    Recognizes a score within a specific ROI of an image.
    This version is modified to accept the ROI and threshold as parameters.
    """
    x, y, w, h = score_roi
    if w <= 0 or h <= 0 or y + h > image.shape[0] or x + w > image.shape[1]:
        return None, None  # Invalid ROI

    # --- Image Processing ---
    roi = image[y:y + h, x:x + w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, threshold_val, 255, cv2.THRESH_BINARY)

    # --- Contour Detection & Digit Recognition ---
    contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_digits = []
    TEMPLATE_HEIGHT = 40  # Standard height for matching

    for contour in contours:
        if cv2.contourArea(contour) < 50: continue  # Filter out small noise

        cx, cy, cw, ch = cv2.boundingRect(contour)
        digit_crop = thresh_roi[cy:cy + ch, cx:cx + cw]

        # Standardize the digit for template matching
        aspect_ratio = cw / ch
        new_width = int(TEMPLATE_HEIGHT * aspect_ratio)
        if new_width <= 0: continue
        standardized_digit = cv2.resize(digit_crop, (new_width, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA)

        # Find the best matching template
        best_match_score, best_match_digit = -1, -1
        for digit_val, template in templates.items():
            h_t, w_t = template.shape
            max_w = max(w_t, new_width)

            # Pad both template and digit to the same width for matching
            padded_template = np.zeros((TEMPLATE_HEIGHT, max_w), dtype=np.uint8)
            padded_digit = np.zeros((TEMPLATE_HEIGHT, max_w), dtype=np.uint8)
            padded_template[:, :w_t] = template
            padded_digit[:, :new_width] = standardized_digit

            res = cv2.matchTemplate(padded_digit, padded_template, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > best_match_score:
                best_match_score, best_match_digit = score, digit_val

        # If the match is good enough, add it to our list
        if best_match_score > 0.8:
            detected_digits.append((best_match_digit, cx))  # Store digit and its x-position

    if not detected_digits:
        return None, thresh_roi

    # Sort digits by their x-position (left to right) and form the score string
    detected_digits.sort(key=lambda d: d[1])
    score_str = "".join([str(d[0]) for d in detected_digits])

    return score_str, thresh_roi


def nothing(x):
    pass


def main():
    WINDOW_TITLE = "Crossy Road"
    TUNER_WINDOW = "Score Tuner"
    CONTROLS_WINDOW = "Controls"
    ROI_WINDOW = "Thresholded ROI"

    try:
        digit_templates = load_digit_templates()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # --- Setup OpenCV Windows and Trackbars ---
    cv2.namedWindow(TUNER_WINDOW)
    cv2.namedWindow(CONTROLS_WINDOW)
    cv2.namedWindow(ROI_WINDOW)

    # tuned values are 10,10,233,121
    cv2.createTrackbar('ROI_X', CONTROLS_WINDOW, 10, 2000, nothing)
    cv2.createTrackbar('ROI_Y', CONTROLS_WINDOW, 10, 2000, nothing)
    cv2.createTrackbar('ROI_W', CONTROLS_WINDOW, 233, 500, nothing)
    cv2.createTrackbar('ROI_H', CONTROLS_WINDOW, 121, 500, nothing)
    cv2.createTrackbar('Threshold', CONTROLS_WINDOW, 180, 255, nothing)

    print("Starting Score Tuner...")
    print("1. Adjust sliders to draw a green box around the score.")
    print("2. Adjust 'Threshold' until digits are clear in the 'Thresholded ROI' window.")
    print("3. Press 'q' to quit and print the final values.")

    with mss.mss() as sct:
        while True:
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if not hwnd:
                print("Waiting for game window...", end='\r')
                time.sleep(1)
                continue

            # --- Screen Capture ---
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
            client_right, client_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
            TITLE_BAR_HEIGHT = 50
            monitor = {
                "top": client_top + TITLE_BAR_HEIGHT,
                "left": client_left,
                "width": client_right - client_left,
                "height": client_bottom - client_top - TITLE_BAR_HEIGHT
            }

            if monitor['width'] <= 0 or monitor['height'] <= 0: continue

            game_frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
            display_frame = game_frame.copy()

            # --- Get Current Slider Values ---
            roi_x = cv2.getTrackbarPos('ROI_X', CONTROLS_WINDOW)
            roi_y = cv2.getTrackbarPos('ROI_Y', CONTROLS_WINDOW)
            roi_w = cv2.getTrackbarPos('ROI_W', CONTROLS_WINDOW)
            roi_h = cv2.getTrackbarPos('ROI_H', CONTROLS_WINDOW)
            threshold = cv2.getTrackbarPos('Threshold', CONTROLS_WINDOW)
            current_roi = (roi_x, roi_y, roi_w, roi_h)

            # --- Recognize and Visualize ---
            recognized_text, thresh_image = recognize_score(game_frame, digit_templates, current_roi, threshold)

            # Draw the ROI rectangle on the display frame
            cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

            # Display the recognized score
            score_display = f"Recognized: {recognized_text if recognized_text else 'N/A'}"
            cv2.putText(display_frame, score_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

            cv2.imshow(TUNER_WINDOW, display_frame)
            if thresh_image is not None:
                cv2.imshow(ROI_WINDOW, thresh_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # --- Print Final Values for easy copy-pasting ---
    print("\n--- Score Tuning Complete ---")
    print("Copy these values into your main.py script:")
    # Convert from (x, y, w, h) to the (y_start, y_end, x_start, x_end) format used in main.py
    y_start, y_end = roi_y, roi_y + roi_h
    x_start, x_end = roi_x, roi_x + roi_w
    print(f"SCORE_ROI = ({y_start}, {y_end}, {x_start}, {x_end})")
    print(f"And update the threshold value inside recognize_score to: {threshold}")
    print("--------------------------------")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
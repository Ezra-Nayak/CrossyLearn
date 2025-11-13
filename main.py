import cv2
import numpy as np
import mss
import time
import win32gui
import win32ui
import os
import pydirectinput
import threading


# --- All functions from before remain the same ---

def load_digit_templates():
    """Loads digit templates from the /templates directory."""
    templates = {}
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Template directory '{template_dir}' not found. Please run create_templates.py first.")
    for i in range(10):
        filepath = os.path.join(template_dir, f"{i}.png")
        if os.path.exists(filepath):
            templates[i] = cv2.imread(filepath, 0)
    print(f"Loaded {len(templates)} digit templates.")
    return templates


def recognize_score(image, templates):
    """Recognizes the score using a dynamic, contour-based approach."""
    SCORE_ROI = (10, 150, 10, 250)
    TEMPLATE_HEIGHT = 40
    roi = image[SCORE_ROI[0]:SCORE_ROI[1], SCORE_ROI[2]:SCORE_ROI[3]]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_digits = []
    for contour in contours:
        if cv2.contourArea(contour) < 50: continue
        x, y, w, h = cv2.boundingRect(contour)
        digit_crop = thresh_roi[y:y + h, x:x + w]
        aspect_ratio = w / h
        new_width = int(TEMPLATE_HEIGHT * aspect_ratio)
        if new_width <= 0: continue
        standardized_digit = cv2.resize(digit_crop, (new_width, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA)
        best_match_score, best_match_digit = -1, -1
        for digit_val, template in templates.items():
            h_t, w_t = template.shape
            h_s, w_s = standardized_digit.shape
            max_w = max(w_t, w_s)
            padded_template = np.zeros((TEMPLATE_HEIGHT, max_w), dtype=np.uint8)
            padded_digit = np.zeros((TEMPLATE_HEIGHT, max_w), dtype=np.uint8)
            padded_template[:, :w_t] = template
            padded_digit[:, :w_s] = standardized_digit
            res = cv2.matchTemplate(padded_digit, padded_template, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > best_match_score:
                best_match_score, best_match_digit = score, digit_val
        if best_match_score > 0.8:
            detected_digits.append((best_match_digit, x))
    if not detected_digits: return None
    detected_digits.sort(key=lambda d: d[1])
    score_str = "".join([str(d[0]) for d in detected_digits])
    return score_str if score_str else None


def send_key(hwnd, key):
    """Brings the specified window to the foreground and presses a key."""
    try:
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.1)
        pydirectinput.press(key)
        time.sleep(0.1)
    except win32ui.error:
        print("Error: Could not set foreground window. It might be closed.")
        pass


# --- New Threaded Vision Class ---

class VisionThread(threading.Thread):
    def __init__(self, window_title):
        super().__init__()
        self.daemon = True  # Allows main thread to exit even if this thread is running
        self.window_title = window_title
        self.templates = load_digit_templates()

        # Shared data between threads
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_score = None
        self.running = True

    def run(self):
        with mss.mss() as sct:
            while self.running:
                try:
                    hwnd = win32gui.FindWindow(None, self.window_title)
                    if not hwnd:
                        time.sleep(1)
                        continue

                    left, top, right, bottom = win32gui.GetClientRect(hwnd)
                    client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                    client_right, client_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
                    TITLE_BAR_HEIGHT = 50
                    monitor = {
                        "top": client_top + TITLE_BAR_HEIGHT, "left": client_left,
                        "width": client_right - client_left, "height": (client_bottom - client_top) - TITLE_BAR_HEIGHT
                    }
                    if monitor['width'] <= 0 or monitor['height'] <= 0:
                        time.sleep(0.1)
                        continue

                    img_bgra = sct.grab(monitor)
                    game_frame = cv2.cvtColor(np.array(img_bgra), cv2.COLOR_BGRA2BGR)

                    score_val = recognize_score(game_frame.copy(), self.templates)

                    # Use lock to update shared data safely
                    with self.lock:
                        self.latest_frame = game_frame
                        self.latest_score = score_val

                except Exception as e:
                    print(f"Error in Vision Thread: {e}")
                    time.sleep(1)

    def stop(self):
        self.running = False


def main():
    WINDOW_TITLE = "Crossy Road"
    TARGET_FPS = 60
    FRAME_DELAY = 1.0 / TARGET_FPS

    print("CrossyLearn Agent - Milestone 8: Multithreaded Architecture")
    print("----------------------------------------------------------")

    # Start the vision processing in a background thread
    vision_thread = VisionThread(WINDOW_TITLE)
    vision_thread.start()

    last_action_time = time.time()
    action_interval = 3.0

    while True:
        loop_start_time = time.time()

        # Get the latest data from the vision thread
        with vision_thread.lock:
            frame = vision_thread.latest_frame
            score = vision_thread.latest_score

        if frame is None:
            print("Waiting for first frame from vision thread...")
            time.sleep(0.5)
            continue

        display_score = score if score is not None else "N/A"
        display_frame = frame.copy()

        # --- Agent Action (Asynchronous) ---
        current_time = time.time()
        if current_time - last_action_time > action_interval:
            print(f"Dispatching 'up' command to game at {current_time:.2f}")
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if hwnd:
                # Run the blocking send_key function in a separate thread
                # to avoid blocking the main render loop.
                action_thread = threading.Thread(target=send_key, args=(hwnd, 'up'))
                action_thread.start()

            last_action_time = current_time

        # --- Display (Lightweight) ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Score: {display_score}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(display_frame, text, (text_x, 40), font, 1, (0, 0, 255), 2)
        cv2.imshow('CrossyLearn Vision', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # --- Precise Frame Rate Limiting ---
        elapsed_time = time.time() - loop_start_time
        sleep_duration = FRAME_DELAY - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    print("Shutting down...")
    vision_thread.stop()
    vision_thread.join()  # Wait for the thread to finish cleanly
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
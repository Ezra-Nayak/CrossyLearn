import cv2
import numpy as np
import mss
import time
import win32gui
import win32ui
import os
import pydirectinput
import threading
from enum import Enum

class GameState(Enum):
    MENU = 1
    PLAYING = 2


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


def detect_retry_button(image, template):
    """Detects the retry button on the screen to confirm player death."""
    if template is None:
        return False

    # Define an ROI for the bottom half of the screen where the button appears
    h, w, _ = image.shape
    RETRY_ROI = image[h // 2:, :]  # Search the bottom half

    res = cv2.matchTemplate(RETRY_ROI, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    # If the best match is very high, we've found the button
    return max_val > 0.9


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
        self.daemon = True
        self.window_title = window_title
        self.digit_templates = load_digit_templates()
        self.retry_template = cv2.imread('templates/retry_button.png')
        if self.retry_template is None:
            raise FileNotFoundError("Could not load 'templates/retry_button.png'")

        # Shared data
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_score = None
        self.is_dead = False
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

                    # --- Perform all vision analysis ---
                    score_val = recognize_score(game_frame.copy(), self.digit_templates)
                    death_detected = detect_retry_button(game_frame.copy(), self.retry_template)

                    # --- Safely update shared data ---
                    with self.lock:
                        self.latest_frame = game_frame
                        self.latest_score = score_val
                        self.is_dead = death_detected

                except Exception as e:
                    print(f"Error in Vision Thread: {e}")
                    time.sleep(1)

    def stop(self):
        self.running = False


def main():
    WINDOW_TITLE = "Crossy Road"
    TARGET_FPS = 60
    FRAME_DELAY = 1.0 / TARGET_FPS

    print("CrossyLearn Agent - Milestone 11: Vision-Based Death Detection")
    print("-------------------------------------------------------------")

    vision_thread = VisionThread(WINDOW_TITLE)
    vision_thread.start()

    game_state = GameState.MENU
    last_score = 0

    while True:
        loop_start_time = time.time()

        with vision_thread.lock:
            frame = vision_thread.latest_frame
            score_str = vision_thread.latest_score
            is_dead = vision_thread.is_dead

        if frame is None:
            print("Waiting for first frame from vision thread...")
            time.sleep(0.5)
            continue

        current_score = int(score_str) if score_str is not None and score_str.isdigit() else None

        # --- Rewritten Game State Machine ---
        if game_state == GameState.MENU:
            # Transition to PLAYING when score appears AND we are not on a death screen
            if current_score is not None and not is_dead:
                game_state = GameState.PLAYING
                last_score = current_score
                print("STATE CHANGE: MENU -> PLAYING")

        elif game_state == GameState.PLAYING:
            # INSTANT death detection via retry button
            if is_dead:
                print(f"EVENT: Player has died! Score was {last_score}. Reward: -100")
                game_state = GameState.MENU
                print("STATE CHANGE: PLAYING -> MENU")
            # Detect score increase
            elif current_score is not None and current_score > last_score:
                reward = (current_score - last_score) * 10
                print(f"EVENT: Score increased to {current_score}! Reward: +{reward}")
                last_score = current_score

        # --- Display ---
        display_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        display_score_text = str(last_score) if game_state == GameState.PLAYING and current_score is not None else "N/A"
        text = f"Score: {display_score_text}"
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(display_frame, text, (text_x, 40), font, 1, (0, 0, 255), 2)
        state_text = f"State: {game_state.name}"
        cv2.putText(display_frame, state_text, (10, 40), font, 1, (0, 0, 255), 2)
        cv2.imshow('CrossyLearn Vision', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - loop_start_time
        sleep_duration = FRAME_DELAY - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)

    print("Shutting down...")
    vision_thread.stop()
    vision_thread.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
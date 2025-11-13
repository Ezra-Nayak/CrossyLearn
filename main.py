import cv2
import numpy as np
import mss
import time
import win32gui
import win32ui
import os
import pydirectinput
import threading
import random
from enum import Enum


class GameState(Enum):
    MENU = 1
    PLAYING = 2


# --- AGENT CLASS ---

class Agent:
    def __init__(self):
        self.actions = ['up', 'left', 'right']
        self.last_action_time = 0
        self.action_interval = 0.5  # seconds between actions

    def choose_action(self):
        """For now, chooses a random action."""
        return random.choice(self.actions)

    def act(self, game_state, hwnd):
        """Decides whether to take an action based on game state and timing."""
        current_time = time.time()

        if game_state == GameState.MENU:
            # If in menu, wait a bit then press spacebar to start the game.
            if current_time - self.last_action_time > 2.0:  # 2-second delay before starting
                print("Agent is starting a new game...")
                self.dispatch_action(hwnd, 'space')
                self.last_action_time = current_time

        elif game_state == GameState.PLAYING:
            # If playing, take an action based on the interval.
            if current_time - self.last_action_time > self.action_interval:
                action = self.choose_action()
                print(f"Agent chose action: {action}")
                self.dispatch_action(hwnd, action)
                self.last_action_time = current_time

    def dispatch_action(self, hwnd, action):
        """Sends the keypress in a separate thread to not block the main loop."""
        if hwnd:
            action_thread = threading.Thread(target=send_key, args=(hwnd, action))
            action_thread.start()


# --- VISION & CONTROL FUNCTIONS ---

def load_digit_templates():
    # ... (rest of the functions are unchanged)
    templates = {}
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        raise FileNotFoundError(f"Template directory '{template_dir}' not found.")
    for i in range(10):
        filepath = os.path.join(template_dir, f"{i}.png")
        if os.path.exists(filepath):
            templates[i] = cv2.imread(filepath, 0)
    print(f"Loaded {len(templates)} digit templates.")
    return templates


def recognize_score(image, templates):
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
            h_t, w_t = template.shape;
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
    if template is None: return False
    ROI = (608, 1488, 252, 135)
    margin = 10
    x, y, w, h = ROI
    # Check if ROI is within image bounds
    if y - margin < 0 or y + h + margin > image.shape[0] or x - margin < 0 or x + w + margin > image.shape[1]:
        return False
    search_area = image[y - margin: y + h + margin, x - margin: x + w + margin]
    if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]:
        return False
    res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val > 0.8


def send_key(hwnd, key):
    try:
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.1)
        pydirectinput.press(key)
        time.sleep(0.1)
    except win32ui.error:
        print("Error: Could not set foreground window.")
        pass


class VisionThread(threading.Thread):
    def __init__(self, window_title):
        super().__init__()
        self.daemon = True
        self.window_title = window_title
        self.digit_templates = load_digit_templates()
        self.retry_template = cv2.imread('templates/retry_button.png')
        if self.retry_template is None:
            raise FileNotFoundError("Could not load 'templates/retry_button.png'")
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
                    if not hwnd: time.sleep(1); continue
                    left, top, right, bottom = win32gui.GetClientRect(hwnd)
                    client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                    client_right, client_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
                    TITLE_BAR_HEIGHT = 50
                    monitor = {
                        "top": client_top + TITLE_BAR_HEIGHT, "left": client_left,
                        "width": client_right - client_left, "height": (client_bottom - client_top) - TITLE_BAR_HEIGHT
                    }
                    if monitor['width'] <= 0 or monitor['height'] <= 0: time.sleep(0.1); continue
                    game_frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
                    score_val = recognize_score(game_frame.copy(), self.digit_templates)
                    death_detected = detect_retry_button(game_frame.copy(), self.retry_template)
                    with self.lock:
                        self.latest_frame = game_frame
                        self.latest_score = score_val
                        self.is_dead = death_detected
                except Exception as e:
                    print(f"Error in Vision Thread: {e}")
                    time.sleep(1)

    def stop(self):
        self.running = False


# --- MAIN APPLICATION LOOP ---

def main():
    WINDOW_TITLE = "Crossy Road"
    TARGET_FPS = 60
    FRAME_DELAY = 1.0 / TARGET_FPS

    print("CrossyLearn Agent - Milestone 13: Baseline Autonomous Agent")
    print("-----------------------------------------------------------")

    vision_thread = VisionThread(WINDOW_TITLE)
    vision_thread.start()

    agent = Agent()
    game_state = GameState.MENU
    last_score = 0
    high_score = 0

    while True:
        loop_start_time = time.time()
        with vision_thread.lock:
            frame = vision_thread.latest_frame
            score_str = vision_thread.latest_score
            is_dead = vision_thread.is_dead

        if frame is None:
            print("Waiting for first frame...");
            time.sleep(0.5);
            continue

        current_score = int(score_str) if score_str is not None and score_str.isdigit() else None

        # --- Agent takes an action ---
        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        agent.act(game_state, hwnd)

        # --- Game State Machine ---
        if game_state == GameState.MENU:
            if current_score is not None and not is_dead:
                game_state = GameState.PLAYING
                last_score = current_score if current_score is not None else 0
                print("\n--- NEW GAME ---")
                print(f"STATE CHANGE: MENU -> PLAYING (High Score: {high_score})")

        elif game_state == GameState.PLAYING:
            if is_dead:
                punishment = -100
                print(f"EVENT: Player has died! Score was {last_score}. Punishment: {punishment}")
                game_state = GameState.MENU
                print("STATE CHANGE: PLAYING -> MENU")
            elif current_score is not None and current_score > last_score:
                shaping_reward = 1
                print(f"EVENT: Score increased to {current_score}! Reward: +{shaping_reward}")
                if current_score > high_score:
                    jackpot_reward = 100
                    print(f"EVENT: New high score! Jackpot Reward: +{jackpot_reward}")
                    high_score = current_score
                last_score = current_score

        # --- Display ---
        display_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        score_text = f"Score: {last_score if game_state == GameState.PLAYING else 'N/A'} | High Score: {high_score}"
        text_size = cv2.getTextSize(score_text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(display_frame, score_text, (text_x, 40), font, 1, (0, 255, 0), 2)
        state_text = f"State: {game_state.name}"
        cv2.putText(display_frame, state_text, (10, 40), font, 1, (0, 255, 0), 2)
        cv2.imshow('CrossyLearn Vision', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        elapsed_time = time.time() - loop_start_time
        sleep_duration = FRAME_DELAY - elapsed_time
        if sleep_duration > 0: time.sleep(sleep_duration)

    print("Shutting down...")
    vision_thread.stop()
    vision_thread.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
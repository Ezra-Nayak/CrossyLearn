# --- CrossyLearn: Modern Architecture with Baseline Agent (Corrected) ---

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
import queue
import math


class GameState(Enum):
    MENU = 1
    PLAYING = 2


# --- AGENT CLASS (Corrected logic from before) ---

class Agent:
    def __init__(self):
        self.actions = ['up', 'left', 'right', 'down']
        self.last_action_time = 0
        self.action_interval = 0.5  # Normal seconds between actions
        self.new_game_cooldown = 2.5  # Special delay for the first action
        self.is_first_move = True

    def on_new_game(self):
        """Resets the agent's state for a new game."""
        print("Agent acknowledging new game. Cooldown will be applied on first move.")
        self.is_first_move = True
        # We set last_action_time to the current time.
        # The cooldown logic is now handled entirely within the act() method.
        self.last_action_time = time.time()

    def choose_action(self):
        """Chooses a random action, avoiding 'left' on the first move."""
        if self.is_first_move:
            return random.choice(['up', 'right', 'down'])
        else:
            return random.choice(self.actions)

    def act(self, game_state, hwnd, window_geo):
        """Decides whether to take an action based on game state and timing."""
        current_time = time.time()

        if game_state == GameState.MENU:
            # If in menu, wait a bit then click the retry button to start.
            if current_time - self.last_action_time > 2.0:
                print("Agent is starting a new game via mouse click...")
                if window_geo:
                    client_left, client_top, title_bar_height = window_geo
                    roi_x, roi_y, roi_w, roi_h = RETRY_BUTTON_ROI
                    click_x = client_left + roi_x + (roi_w // 2)
                    click_y = client_top + title_bar_height + roi_y + (roi_h // 2)
                    self.dispatch_click(hwnd, (click_x, click_y))
                    self.last_action_time = current_time

        elif game_state == GameState.PLAYING:
            # Determine the correct interval: a long one for the first move, a short one for all others.
            interval = self.new_game_cooldown if self.is_first_move else self.action_interval

            if current_time - self.last_action_time > interval:
                action = self.choose_action()
                print(f"Agent chose action: {action}")
                self.dispatch_action(hwnd, action)
                self.last_action_time = current_time
                if self.is_first_move:
                    self.is_first_move = False

    def dispatch_action(self, hwnd, action):
        if hwnd:
            action_thread = threading.Thread(target=send_key, args=(hwnd, action))
            action_thread.start()

    def dispatch_click(self, hwnd, click_pos):
        if hwnd:
            click_thread = threading.Thread(target=send_click, args=(hwnd, click_pos))
            click_thread.start()


# --- VISION & CONTROL FUNCTIONS (Unchanged) ---

def load_digit_templates():
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
    SCORE_ROI = (10, 121, 10, 233)
    TEMPLATE_HEIGHT = 40
    try:
        roi = image[SCORE_ROI[0]:SCORE_ROI[1], SCORE_ROI[2]:SCORE_ROI[3]]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
    except Exception:
        return None
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
            if score > best_match_score: best_match_score, best_match_digit = score, digit_val
        if best_match_score > 0.8: detected_digits.append((best_match_digit, x))
    if not detected_digits: return None
    detected_digits.sort(key=lambda d: d[1])
    score_str = "".join([str(d[0]) for d in detected_digits])
    return score_str if score_str else None


RETRY_BUTTON_ROI = (687, 864, 156, 98)


def detect_retry_button(image, template):
    if template is None: return False
    x, y, w, h = RETRY_BUTTON_ROI
    margin = 10
    if y - margin < 0 or y + h + margin > image.shape[0] or x - margin < 0 or x + w + margin > image.shape[
        1]: return False
    search_area = image[y - margin: y + h + margin, x - margin: x + w + margin]
    if search_area.shape[0] < template.shape[0] or search_area.shape[1] < template.shape[1]: return False
    res = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val > 0.8


def send_key(hwnd, key):
    try:
        win32gui.SetForegroundWindow(hwnd);
        time.sleep(0.05)
        pydirectinput.press(key);
        time.sleep(0.05)
    except win32ui.error:
        pass


def send_click(hwnd, click_pos):
    try:
        win32gui.SetForegroundWindow(hwnd);
        time.sleep(0.05)
        pydirectinput.moveTo(click_pos[0], click_pos[1])
        pydirectinput.click();
        time.sleep(0.05)
    except win32ui.error:
        pass


class VisionProducer(threading.Thread):
    def __init__(self, window_title, output_queue):
        super().__init__()
        self.daemon = True
        self.window_title = window_title
        self.output_queue = output_queue
        self.running = True
        self.digit_templates = load_digit_templates()
        self.retry_template = cv2.imread('templates/retry_button.png')
        if self.retry_template is None: raise FileNotFoundError("Could not load 'templates/retry_button.png'")
        self.LOWER_BOUND = np.array([118, 0, 0]);
        self.UPPER_BOUND = np.array([119, 255, 255])
        self.AREA_MIN = 1320;
        self.AREA_MAX = 2500
        self.SEARCH_ZONE_Y_INTERCEPT = 310;
        self.PENALTY_LINE_Y_INTERCEPT = 720
        self.LINE_ANGLE_DEG = 15

    def find_chicken(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.LOWER_BOUND, self.UPPER_BOUND)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None
        valid_contours = [c for c in contours if self.AREA_MIN < cv2.contourArea(c) < self.AREA_MAX]
        if not valid_contours: return None, None
        chicken_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(chicken_contour)
        return (x + w // 2, y + h), (x, y, w, h)

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
                    monitor = {"top": client_top + TITLE_BAR_HEIGHT, "left": client_left,
                               "width": client_right - client_left,
                               "height": client_bottom - client_top - TITLE_BAR_HEIGHT}
                    if monitor['width'] <= 0 or monitor['height'] <= 0: time.sleep(0.1); continue
                    game_frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
                    frame_h, frame_w, _ = game_frame.shape
                    score_val = recognize_score(game_frame.copy(), self.digit_templates)
                    death_detected = detect_retry_button(game_frame.copy(), self.retry_template)
                    angle_rad = math.radians(self.LINE_ANGLE_DEG);
                    slope = math.tan(angle_rad)
                    search_y1 = self.SEARCH_ZONE_Y_INTERCEPT;
                    search_y2 = int(slope * frame_w + search_y1)
                    search_mask = np.zeros(game_frame.shape[:2], dtype=np.uint8)
                    pts = np.array([[0, search_y1], [frame_w, search_y2], [frame_w, frame_h], [0, frame_h]],
                                   dtype=np.int32)
                    cv2.fillPoly(search_mask, [pts], 255)
                    search_area = cv2.bitwise_and(game_frame, game_frame, mask=search_mask)
                    chicken_pos, chicken_box = self.find_chicken(search_area)
                    penalty_detected = False
                    if chicken_pos:
                        cx, cy = chicken_pos
                        penalty_line_y_at_chicken = int(slope * cx + self.PENALTY_LINE_Y_INTERCEPT)
                        if cy > penalty_line_y_at_chicken: penalty_detected = True
                    p_x1, p_y1 = 0, self.PENALTY_LINE_Y_INTERCEPT;
                    p_x2, p_y2 = frame_w, int(slope * frame_w + self.PENALTY_LINE_Y_INTERCEPT)
                    result = {"frame": game_frame, "score": score_val, "is_dead": death_detected,
                              "in_penalty_zone": penalty_detected, "chicken_box": chicken_box,
                              "penalty_line_pts": ((p_x1, p_y1), (p_x2, p_y2))}
                    try:
                        self.output_queue.put_nowait(result)
                    except queue.Full:
                        pass
                except Exception as e:
                    print(f"Error in VisionProducer: {e}"); time.sleep(1)

    def stop(self):
        self.running = False


# --- MAIN APPLICATION LOOP ---

def main():
    WINDOW_TITLE = "Crossy Road"
    TARGET_FPS = 45
    FRAME_DELAY = 1.0 / TARGET_FPS
    print("CrossyLearn Agent - Modern Architecture w/ Baseline Agent (Corrected)")
    print("--------------------------------------------------------------------")

    vision_queue = queue.Queue(maxsize=1)
    vision_producer = VisionProducer(WINDOW_TITLE, vision_queue)
    vision_producer.start()

    agent = Agent()
    game_state = GameState.MENU
    last_score = 0
    high_score = 0
    episode_num = 0
    penalty_timer = 0
    PENALTY_INTERVAL = 0.5
    frame = None
    fps = 0
    frame_count = 0
    fps_time = time.time()

    # --- BUG FIX: State tracking variable ---
    was_dead = False

    while True:
        loop_start_time = time.time()
        try:
            vision_data = vision_queue.get_nowait()
        except queue.Empty:
            if frame is None:
                print("Waiting for first frame from VisionProducer...")
                time.sleep(0.5)
                continue
            pass

        frame = vision_data['frame']
        score_str = vision_data['score']
        is_dead = vision_data['is_dead']
        in_penalty_zone = vision_data['in_penalty_zone']
        chicken_box = vision_data['chicken_box']
        penalty_line_pts = vision_data['penalty_line_pts']
        current_score = int(score_str) if score_str is not None and score_str.isdigit() else None

        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        window_geo = None
        if hwnd:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
            window_geo = (client_left, client_top, 50)

        # Agent acts based on the CURRENT state. The agent now only clicks retry if is_dead is true.
        if is_dead:
            agent.act(game_state, hwnd, window_geo)

        # --- Game State Machine (CORRECTED LOGIC) ---
        if game_state == GameState.MENU:
            # This is the key fix. A new game starts the frame AFTER we were dead, but are no longer.
            if was_dead and not is_dead:
                game_state = GameState.PLAYING
                last_score = 0  # Reset score for the new run
                episode_num += 1
                print(f"\n--- NEW GAME (Episode {episode_num}) ---")
                print(f"STATE CHANGE: MENU -> PLAYING (High Score: {high_score})")
                agent.on_new_game()

        elif game_state == GameState.PLAYING:
            # Agent acts during the playing state
            agent.act(game_state, hwnd, window_geo)

            if is_dead:
                print(f"EVENT: Player has died! Final score was {last_score}.")
                game_state = GameState.MENU
                print("STATE CHANGE: PLAYING -> MENU")
                penalty_timer = 0
            else:
                if current_score is not None and current_score > last_score:
                    print(f"EVENT: Score increased to {current_score}!")
                    if current_score > high_score:
                        print(f"EVENT: New high score!")
                        high_score = current_score
                    last_score = current_score
                if in_penalty_zone:
                    if penalty_timer == 0:
                        penalty_timer = time.time()
                    elif time.time() - penalty_timer > PENALTY_INTERVAL:
                        print(f"EVENT: In penalty zone!")
                        penalty_timer = time.time()
                else:
                    penalty_timer = 0

        # --- BUG FIX: Update was_dead at the END of the loop ---
        was_dead = is_dead

        # --- Display ---
        display_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        state_text = f"State: {game_state.name} | Episode: {episode_num}"
        cv2.putText(display_frame, state_text, (10, 40), font, 1, (0, 255, 0), 2)
        score_text = f"Score: {last_score if game_state == GameState.PLAYING else 'N/A'} | High Score: {high_score}"
        text_size = cv2.getTextSize(score_text, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(display_frame, score_text, (text_x, 80), font, 1, (0, 255, 0), 2)
        if penalty_line_pts: cv2.line(display_frame, penalty_line_pts[0], penalty_line_pts[1], (0, 0, 255), 2)
        if chicken_box and game_state == GameState.PLAYING:
            x, y, w, h = chicken_box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame_count += 1
        current_time = time.time()
        if current_time - fps_time >= 1.0:
            fps = frame_count;
            frame_count = 0;
            fps_time = current_time
        cv2.putText(display_frame, f"FPS: {fps}", (10, 120), font, 0.7, (0, 255, 0), 2)
        cv2.imshow('CrossyLearn Vision', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

        elapsed_time = time.time() - loop_start_time
        sleep_duration = FRAME_DELAY - elapsed_time
        if sleep_duration > 0: time.sleep(sleep_duration)

    print("Shutting down...")
    vision_producer.stop()
    vision_producer.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
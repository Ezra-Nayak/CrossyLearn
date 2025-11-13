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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- DEEP LEARNING SETUP & MODEL ARCHITECTURE ---

def setup_device():
    """Checks for DirectML availability and sets the device accordingly."""
    try:
        import torch_directml
        dml_device = torch_directml.device(torch_directml.default_device())
        print(f"[INFO] PyTorch is using DirectML device: {dml_device}")
        return dml_device
    except (ImportError, Exception):
        print("[INFO] DirectML not found or failed. Falling back to CPU.")
        return torch.device("cpu")


DEVICE = setup_device()


def preprocess_frame(frame):
    """Preprocesses a game frame for the DQN."""
    if frame is None: return None
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    tensor_frame = torch.from_numpy(resized_frame).unsqueeze(0).unsqueeze(0).float().to(DEVICE) / 255.0
    return tensor_frame


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x;
        out = self.relu(self.conv1(x));
        out = self.conv1(out)
        out += identity;
        return self.relu(out)


class DuelingDQN(nn.Module):
    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.res1 = ResidualBlock(64);
        self.res2 = ResidualBlock(64)
        self.value_fc = nn.Linear(7 * 7 * 64, 512);
        self.value_output = nn.Linear(512, 1)
        self.advantage_fc = nn.Linear(7 * 7 * 64, 512);
        self.advantage_output = nn.Linear(512, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x));
        x = self.relu(self.conv2(x));
        x = self.relu(self.conv3(x))
        x = self.res1(x);
        x = self.res2(x);
        x = x.view(x.size(0), -1)
        value = self.relu(self.value_fc(x));
        value = self.value_output(value)
        advantage = self.relu(self.advantage_fc(x));
        advantage = self.advantage_output(advantage)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True));
        return q_values


# --- AGENT CLASS ---

class Agent:
    def __init__(self):
        self.actions = ['up', 'left', 'right', 'down'];
        self.num_actions = len(self.actions)
        self.policy_net = DuelingDQN(self.num_actions).to(DEVICE)
        self.target_net = DuelingDQN(self.num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict());
        self.target_net.eval()
        self.last_action_time = 0;
        self.action_interval = 0.5;
        self.is_first_move = True

    def on_new_game(self):
        print("Agent acknowledging new game. Cooldown initiated.");
        self.is_first_move = True
        self.last_action_time = time.time() + 2.5

    def choose_action(self, state_tensor, is_first_move):
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        if is_first_move:
            return random.choice(['up', 'right', 'down'])
        else:
            return random.choice(self.actions)

    def act(self, game_state, hwnd, current_frame):
        current_time = time.time()
        if game_state == GameState.MENU:
            if current_time - self.last_action_time > 2.0:
                print("Agent is starting a new game...");
                self.dispatch_action(hwnd, 'space')
                self.last_action_time = current_time
        elif game_state == GameState.PLAYING:
            if current_time - self.last_action_time > self.action_interval:
                processed_frame = preprocess_frame(current_frame)
                if processed_frame is not None:
                    action = self.choose_action(processed_frame, self.is_first_move)
                    print(f"Agent chose action: {action}");
                    self.dispatch_action(hwnd, action)
                    self.last_action_time = current_time
                    if self.is_first_move: self.is_first_move = False

    def dispatch_action(self, hwnd, action):
        if hwnd:
            action_thread = threading.Thread(target=send_key, args=(hwnd, action));
            action_thread.start()


# --- VISION & CONTROL FUNCTIONS ---

def load_digit_templates():
    templates = {};
    template_dir = 'templates'
    if not os.path.exists(template_dir): raise FileNotFoundError(f"Template directory '{template_dir}' not found.")
    for i in range(10):
        filepath = os.path.join(template_dir, f"{i}.png")
        if os.path.exists(filepath): templates[i] = cv2.imread(filepath, 0)
    print(f"Loaded {len(templates)} digit templates.");
    return templates


def recognize_score(image, templates):
    SCORE_ROI = (10, 150, 10, 250);
    TEMPLATE_HEIGHT = 40
    roi = image[SCORE_ROI[0]:SCORE_ROI[1], SCORE_ROI[2]:SCORE_ROI[3]]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_digits = []
    for contour in contours:
        if cv2.contourArea(contour) < 50: continue
        x, y, w, h = cv2.boundingRect(contour);
        digit_crop = thresh_roi[y:y + h, x:x + w]
        aspect_ratio = w / h;
        new_width = int(TEMPLATE_HEIGHT * aspect_ratio)
        if new_width <= 0: continue
        standardized_digit = cv2.resize(digit_crop, (new_width, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA)
        best_match_score, best_match_digit = -1, -1
        for digit_val, template in templates.items():
            h_t, w_t = template.shape;
            h_s, w_s = standardized_digit.shape;
            max_w = max(w_t, w_s)
            padded_template = np.zeros((TEMPLATE_HEIGHT, max_w), dtype=np.uint8)
            padded_digit = np.zeros((TEMPLATE_HEIGHT, max_w), dtype=np.uint8)
            padded_template[:, :w_t] = template;
            padded_digit[:, :w_s] = standardized_digit
            res = cv2.matchTemplate(padded_digit, padded_template, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > best_match_score: best_match_score, best_match_digit = score, digit_val
        if best_match_score > 0.8: detected_digits.append((best_match_digit, x))
    if not detected_digits: return None
    detected_digits.sort(key=lambda d: d[1])
    score_str = "".join([str(d[0]) for d in detected_digits])
    return score_str if score_str else None


def detect_retry_button(image, template):
    if template is None: return False
    ROI = (608, 1488, 252, 135);
    margin = 10;
    x, y, w, h = ROI
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
        time.sleep(0.1)
        pydirectinput.press(key);
        time.sleep(0.1)
    except win32ui.error:
        pass


# --- BACKGROUND VISION THREAD (UPGRADED) ---

class VisionThread(threading.Thread):
    def __init__(self, window_title):
        super().__init__();
        self.daemon = True;
        self.window_title = window_title
        # Load templates
        self.digit_templates = load_digit_templates()
        self.retry_template = cv2.imread('templates/retry_button.png')
        if self.retry_template is None: raise FileNotFoundError("Could not load 'templates/retry_button.png'")

        # --- Final Tuned Parameters ---
        self.LOWER_BOUND = np.array([118, 0, 0])
        self.UPPER_BOUND = np.array([119, 255, 255])
        self.AREA_MIN = 1481
        self.AREA_MAX = 5000
        self.SEARCH_ZONE_Y_INTERCEPT = 700
        self.PENALTY_LINE_Y_INTERCEPT = 1285
        self.LINE_ANGLE_DEG = 15

        # Shared data
        self.lock = threading.Lock()
        self.latest_frame = None;
        self.latest_score = None
        self.is_dead = False;
        self.in_penalty_zone = False;
        self.running = True

    def find_chicken(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.LOWER_BOUND, self.UPPER_BOUND)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        valid_contours = [c for c in contours if self.AREA_MIN < cv2.contourArea(c) < self.AREA_MAX]
        if not valid_contours: return None
        chicken_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(chicken_contour)
        return (x + w // 2, y + h)

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

                    # --- Full Perception Pipeline ---
                    score_val = recognize_score(game_frame.copy(), self.digit_templates)
                    death_detected = detect_retry_button(game_frame.copy(), self.retry_template)

                    # Angled Search Zone Mask
                    angle_rad = math.radians(self.LINE_ANGLE_DEG);
                    slope = math.tan(angle_rad)
                    search_y1 = self.SEARCH_ZONE_Y_INTERCEPT
                    search_y2 = int(slope * frame_w + search_y1)
                    search_mask = np.zeros(game_frame.shape[:2], dtype=np.uint8)
                    pts = np.array([[0, search_y1], [frame_w, search_y2], [frame_w, frame_h], [0, frame_h]],
                                   dtype=np.int32)
                    cv2.fillPoly(search_mask, [pts], 255)
                    search_area = cv2.bitwise_and(game_frame, game_frame, mask=search_mask)

                    chicken_pos = self.find_chicken(search_area)

                    penalty_detected = False
                    if chicken_pos:
                        cx, cy = chicken_pos
                        # Adjust cy back to full frame coordinates
                        cy_full_frame = cy + self.SEARCH_ZONE_Y_INTERCEPT
                        penalty_line_y = int(slope * cx + self.PENALTY_LINE_Y_INTERCEPT)
                        if cy_full_frame > penalty_line_y:
                            penalty_detected = True

                    with self.lock:
                        self.latest_frame = game_frame;
                        self.latest_score = score_val
                        self.is_dead = death_detected;
                        self.in_penalty_zone = penalty_detected
                except Exception as e:
                    print(f"Error in Vision Thread: {e}");
                    time.sleep(1)

    def stop(self):
        self.running = False


# --- MAIN APPLICATION LOOP ---
class GameState(Enum):
    MENU = 1;
    PLAYING = 2


def main():
    WINDOW_TITLE = "Crossy Road";
    TARGET_FPS = 60;
    FRAME_DELAY = 1.0 / TARGET_FPS
    print("CrossyLearn Agent - Milestone 19: Final Perception System")
    print("---------------------------------------------------------")
    vision_thread = VisionThread(WINDOW_TITLE);
    vision_thread.start()
    agent = Agent();
    game_state = GameState.MENU;
    last_score = 0;
    high_score = 0
    penalty_timer = 0;
    PENALTY_INTERVAL = 1.0  # seconds

    while True:
        loop_start_time = time.time()
        with vision_thread.lock:
            frame = vision_thread.latest_frame;
            score_str = vision_thread.latest_score
            is_dead = vision_thread.is_dead;
            in_penalty_zone = vision_thread.in_penalty_zone

        if frame is None: print("Waiting for first frame..."); time.sleep(0.5); continue

        current_score = int(score_str) if score_str is not None and score_str.isdigit() else None
        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        agent.act(game_state, hwnd, frame)

        if game_state == GameState.MENU:
            if current_score is not None and not is_dead:
                game_state = GameState.PLAYING;
                last_score = current_score if current_score is not None else 0
                print("\n--- NEW GAME ---");
                print(f"STATE CHANGE: MENU -> PLAYING (High Score: {high_score})")
                agent.on_new_game()
        elif game_state == GameState.PLAYING:
            if is_dead:
                print(f"EVENT: Player has died! Score was {last_score}. Punishment: -100")
                game_state = GameState.MENU;
                print("STATE CHANGE: PLAYING -> MENU")
                penalty_timer = 0  # Reset timer on death
            elif current_score is not None and current_score > last_score:
                print(f"EVENT: Score increased to {current_score}! Reward: +1")
                if current_score > high_score:
                    print(f"EVENT: New high score! Jackpot Reward: +100");
                    high_score = current_score
                last_score = current_score

            # Penalty Zone Logic
            if in_penalty_zone:
                if penalty_timer == 0:
                    penalty_timer = time.time()
                elif time.time() - penalty_timer > PENALTY_INTERVAL:
                    print(f"EVENT: In penalty zone! Punishment: -10");
                    penalty_timer = time.time()
            else:
                penalty_timer = 0  # Reset timer if safe

        display_frame = frame.copy();
        font = cv2.FONT_HERSHEY_SIMPLEX
        score_text = f"Score: {last_score if game_state == GameState.PLAYING else 'N/A'} | High Score: {high_score}"
        text_size = cv2.getTextSize(score_text, font, 1, 2)[0];
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(display_frame, score_text, (text_x, 40), font, 1, (0, 255, 0), 2)
        state_text = f"State: {game_state.name}";
        cv2.putText(display_frame, state_text, (10, 40), font, 1, (0, 255, 0), 2)
        cv2.imshow('CrossyLearn Vision', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        elapsed_time = time.time() - loop_start_time
        sleep_duration = FRAME_DELAY - elapsed_time
        if sleep_duration > 0: time.sleep(sleep_duration)

    print("Shutting down...");
    vision_thread.stop();
    vision_thread.join();
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
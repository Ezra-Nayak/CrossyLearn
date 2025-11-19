# --- crossy_env.py ---
import cv2
import numpy as np
import mss
import win32gui
import pydirectinput
import time
import math
from standalone.terrain_analyzer import Terrain, get_terrain_type

# --- GAME CONSTANTS ---
WINDOW_TITLE = "Crossy Road"
# Vision Tuning (From your final_verify.py)
LOWER_BOUND = np.array([170, 125, 21])
UPPER_BOUND = np.array([179, 136, 37])
SEARCH_ZONE_Y_INTERCEPT = 310
LINE_ANGLE_DEG = 15
PENALTY_Y_LIMIT = 850  # If chicken drops below this, assume Eagle Death

# Isometric Vectors (From your feeler_visualiser.py)
FORWARD_VEC = np.array([28, -70])
RIGHT_VEC = np.array([80, 20])
PATCH_SIZE = 4

# Retry Button (From retry_click_verifier.py)
RETRY_CLICK_COORDS = (767, 912)


class CrossyEnv:
    def __init__(self):
        self.sct = mss.mss()
        self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if not self.hwnd:
            raise Exception("Game window not found!")

        # Action Space: 0: Idle, 1: Forward, 2: Left, 3: Right
        # Note: Backward is rarely useful and dangerous, excluded for faster training V1
        self.action_space = 4

        # State Space:
        # Grid 5x5 around chicken (25 ints) + Normalized Y Pos (1 float)
        self.grid_radius = 3  # Look 3 tiles ahead, 3 tiles wide
        self.state_dim = ((self.grid_radius * 2 + 1) ** 2) + 1

    def get_window_geometry(self):
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        client_point = win32gui.ClientToScreen(self.hwnd, (left, top))
        return {
            "top": client_point[1] + 50,  # Title bar
            "left": client_point[0],
            "width": right - left,
            "height": bottom - top - 50
        }

    def grab_frame(self):
        monitor = self.get_window_geometry()
        if monitor['width'] <= 0: return None
        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def find_chicken(self, frame):
        # Angled Mask Logic
        h, w, _ = frame.shape
        angle_rad = math.radians(LINE_ANGLE_DEG)
        slope = math.tan(angle_rad)

        search_y2 = int(slope * w + SEARCH_ZONE_Y_INTERCEPT)
        search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array([[0, SEARCH_ZONE_Y_INTERCEPT], [w, search_y2], [w, h], [0, h]], dtype=np.int32)
        cv2.fillPoly(search_mask, [pts], 255)

        frame_masked = cv2.bitwise_and(frame, frame, mask=search_mask)
        hsv = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BOUND, UPPER_BOUND)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        # Filter tiny noise
        valid = [c for c in contours if cv2.contourArea(c) > 10]  # Threshold from your logs
        if not valid: return None

        c = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h)  # Feet position

    def get_state(self, frame, chicken_pos):
        if not chicken_pos:
            # If lost, return zero grid
            return np.zeros(self.state_dim)

        cx, cy = chicken_pos
        grid_data = []

        # Scan a grid around the chicken
        # Range: -3 to +3
        r = self.grid_radius
        for dy_grid in range(-r, r + 1):  # Front to Back
            for dx_grid in range(-r, r + 1):  # Left to Right

                # Calculate pixel offset using Isometric Vectors
                # Forward is -Y in grid, Right is +X in grid
                # P = Chicken + (RightVec * dx) + (ForwardVec * dy)
                # Note: In grid, usually +y is up (forward).

                offset = (RIGHT_VEC * dx_grid) + (FORWARD_VEC * -dy_grid)
                px, py = int(cx + offset[0]), int(cy + offset[1])

                t_type = Terrain.UNKNOWN

                # Boundary check
                if 0 <= py < frame.shape[0] and 0 <= px < frame.shape[1]:
                    # Extract patch
                    y1, y2 = max(0, py - PATCH_SIZE), min(frame.shape[0], py + PATCH_SIZE)
                    x1, x2 = max(0, px - PATCH_SIZE), min(frame.shape[1], px + PATCH_SIZE)
                    patch = frame[y1:y2, x1:x2]
                    t_type = get_terrain_type(patch)

                grid_data.append(int(t_type))

        # Append Normalized Y position (Danger sensing)
        norm_y = cy / frame.shape[0]
        grid_data.append(norm_y)

        return np.array(grid_data)

    def is_pause_visible(self, frame):
        # Logic from pause_roi_debugger.py
        h, w, _ = frame.shape
        roi_w = int(w * 0.08)
        roi_h = int(h * 0.06)
        roi = frame[110:110 + roi_h, w - roi_w:w - 20]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_bars = 0
        for c in contours:
            if cv2.contourArea(c) > 15:
                _, _, cw, ch = cv2.boundingRect(c)
                if ch / cw > 1.5: valid_bars += 1
        return valid_bars == 2

    def reset(self):
        print("ENV: Resetting...")

        # 1. Check if we are effectively dead
        frame = self.grab_frame()
        if self.is_pause_visible(frame):
            # If we are alive, force death (for training consistency) or restart logic
            # For now, assume we call reset() only when dead.
            pass

        # 2. Click Retry
        monitor = self.get_window_geometry()
        tx = monitor['left'] + RETRY_CLICK_COORDS[0]
        ty = monitor['top'] + RETRY_CLICK_COORDS[1]

        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(0.1)
        pydirectinput.moveTo(tx, ty)
        pydirectinput.click()
        time.sleep(0.5)
        pydirectinput.press('space')
        time.sleep(1.5)  # Wait for start animation

        # 3. Get initial state
        frame = self.grab_frame()
        pos = self.find_chicken(frame)
        if not pos:
            # Fallback if vision fails on start
            pos = (frame.shape[1] // 2, frame.shape[0] // 2)

        return self.get_state(frame, pos)

    def step(self, action):
        # Actions: 0:Idle, 1:Fwd, 2:Left, 3:Right

        if action == 1:
            pydirectinput.press('up')
        elif action == 2:
            pydirectinput.press('left')
        elif action == 3:
            pydirectinput.press('right')

        # Small delay for animation to start
        time.sleep(0.15)

        frame = self.grab_frame()
        pos = self.find_chicken(frame)

        # --- REWARD LOGIC ---
        reward = 0
        done = False

        # 1. Death Check (Pause missing OR Y too low)
        is_alive = self.is_pause_visible(frame)
        y_limit_breached = False

        if pos:
            if pos[1] > PENALTY_Y_LIMIT:
                y_limit_breached = True
        else:
            # If pos is None, we might be "squished" or dead
            # If pause button is ALSO gone, definitely dead
            if not is_alive:
                done = True

        if not is_alive or y_limit_breached:
            done = True
            reward = -10
        else:
            # Alive Reward
            if action == 1:
                reward = 1.0  # Incentive to move forward
            elif action == 0:
                reward = -0.1  # Penalty for idling
            else:
                reward = -0.01  # Small cost for side movement

            # Positional Reward (Higher on screen = better)
            if pos:
                reward += (1.0 - (pos[1] / frame.shape[0])) * 0.5

        next_state = self.get_state(frame, pos)
        return next_state, reward, done
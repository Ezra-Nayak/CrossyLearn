# --- crossy_env.py ---
import cv2
import numpy as np
import mss
import win32gui
import pydirectinput
import time
import math
import subprocess
import os
from standalone.terrain_analyzer import Terrain, TERRAIN_HSV_RANGES

# --- CONFIGURATION ---
WINDOW_TITLE = "Crossy Road"
EXECUTABLE_PATH = r"C:\Program Files\WindowsApps\Yodo1Ltd.CrossyRoad_1.3.4.0_x86__s3s3f300emkze\Crossy Road.exe"

# Vision Constants
LOWER_BOUND = np.array([170, 125, 21])
UPPER_BOUND = np.array([179, 136, 37])
SEARCH_ZONE_Y_INTERCEPT = 310
LINE_ANGLE_DEG = 15

# Eagle Death Line (Slanted)
EAGLE_Y_INTERCEPT = 850

# Isometric Vectors
FORWARD_VEC = np.array([28, -70])
RIGHT_VEC = np.array([80, 20])
PATCH_SIZE = 4

RETRY_CLICK_COORDS = (767, 912)


class GameCrashedError(Exception):
    pass


class CrossyEnv:
    def __init__(self):
        self.sct = mss.mss()
        self.action_space = 4  # 0:Idle, 1:Up, 2:Left, 3:Right
        self.grid_radius = 3
        self.state_dim = ((self.grid_radius * 2 + 1) ** 2) + 1
        self.steps_in_episode = 0

        # Pre-calculate slopes
        angle_rad = math.radians(LINE_ANGLE_DEG)
        self.slope = math.tan(angle_rad)

        # Ensure game is open on init
        self._ensure_game_running()

    def _ensure_game_running(self):
        self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if not self.hwnd:
            print("[ENV] Game window not found. Launching application...")
            try:
                # Attempt to launch. Note: launching into WindowsApps folder often requires
                # special handling or 'explorer.exe shell:AppsFolder\...' but we try direct first.
                if os.path.exists(EXECUTABLE_PATH):
                    subprocess.Popen(EXECUTABLE_PATH)
                else:
                    # Fallback: Try generic start (sometimes works for registered Appx)
                    os.system(f'start "" "{EXECUTABLE_PATH}"')

                # Wait for load
                print("[ENV] Waiting 10 seconds for startup...")
                time.sleep(10)

                self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
                if not self.hwnd:
                    raise Exception("Failed to launch game or find window after launch.")

                # Get it to the main menu
                print("[ENV] Pressing Space to pass splash screen...")
                time.sleep(2)
                self._focus_window()
                pydirectinput.press('space')
                time.sleep(2)

            except Exception as e:
                print(f"[ENV] Critical Error launching game: {e}")
                raise e

    def _focus_window(self):
        if self.hwnd:
            try:
                win32gui.SetForegroundWindow(self.hwnd)
            except:
                pass

    def get_window_geometry(self):
        try:
            left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
            client_point = win32gui.ClientToScreen(self.hwnd, (left, top))
            return {
                "top": client_point[1] + 50,
                "left": client_point[0],
                "width": right - left,
                "height": bottom - top - 50
            }
        except Exception:
            raise GameCrashedError("Window handle invalid")

    def grab_frame(self):
        monitor = self.get_window_geometry()
        if monitor['width'] <= 0: return None
        img = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def find_chicken(self, frame):
        h, w, _ = frame.shape

        # Angled Search Mask
        search_y2 = int(self.slope * w + SEARCH_ZONE_Y_INTERCEPT)
        search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array([[0, SEARCH_ZONE_Y_INTERCEPT], [w, search_y2], [w, h], [0, h]], dtype=np.int32)
        cv2.fillPoly(search_mask, [pts], 255)

        frame_masked = cv2.bitwise_and(frame, frame, mask=search_mask)
        hsv = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BOUND, UPPER_BOUND)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > 10]

        if not valid: return None
        c = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return (x + w // 2, y + h)

    def check_eagle_death(self, chicken_pos, frame_w):
        """
        Returns True if chicken is below the angled eagle line.
        Line Eq: y = slope * x + 850
        """
        cx, cy = chicken_pos
        eagle_line_y = int(self.slope * cx + EAGLE_Y_INTERCEPT)

        # In CV2, Y increases downwards.
        # If cy > eagle_line_y, we are BELOW the line (visually lower/closer to eagle).
        return cy > eagle_line_y

    def get_state(self, frame, chicken_pos):
        if not chicken_pos:
            return np.zeros(self.state_dim), []

        cx, cy = chicken_pos
        grid_data = []
        debug_patches = []

        # OPTIMIZATION: Convert full frame to HSV once, instead of 49 times
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        r = self.grid_radius
        for dy_grid in range(-r, r + 1):
            for dx_grid in range(-r, r + 1):
                offset = (RIGHT_VEC * dx_grid) + (FORWARD_VEC * -dy_grid)
                px, py = int(cx + offset[0]), int(cy + offset[1])

                t_type = Terrain.UNKNOWN

                # Boundary check
                if 0 <= py < frame.shape[0] and 0 <= px < frame.shape[1]:
                    # Extract HSV Patch directly
                    y1, y2 = max(0, py - PATCH_SIZE), min(frame.shape[0], py + PATCH_SIZE)
                    x1, x2 = max(0, px - PATCH_SIZE), min(frame.shape[1], px + PATCH_SIZE)

                    if y2 > y1 and x2 > x1:
                        patch_hsv = hsv_frame[y1:y2, x1:x2]

                        # Fast Terrain Check (Inline)
                        avg_hsv = np.mean(patch_hsv, axis=(0, 1))
                        h_val, s_val, v_val = avg_hsv

                        for terrain, (lower, upper) in TERRAIN_HSV_RANGES.items():
                            if (lower[0] <= h_val <= upper[0] and
                                    lower[1] <= s_val <= upper[1] and
                                    lower[2] <= v_val <= upper[2]):
                                t_type = terrain
                                break

                grid_data.append(int(t_type))
                debug_patches.append((px, py, t_type))

        norm_y = cy / frame.shape[0]
        grid_data.append(norm_y)
        return np.array(grid_data), debug_patches

    def is_pause_visible(self, frame):
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
        self.steps_in_episode = 0
        self._ensure_game_running()

        monitor = self.get_window_geometry()
        tx = monitor['left'] + RETRY_CLICK_COORDS[0]
        ty = monitor['top'] + RETRY_CLICK_COORDS[1]

        self._focus_window()
        time.sleep(0.1)
        pydirectinput.moveTo(tx, ty)
        pydirectinput.click()
        time.sleep(0.5)
        pydirectinput.press('space')
        time.sleep(1.5)

        frame = self.grab_frame()
        pos = self.find_chicken(frame)
        if not pos: pos = (frame.shape[1] // 2, frame.shape[0] // 2)

        state, self.last_debug_grid = self.get_state(frame, pos)
        return state

    def step(self, action):
        # 1. Crash Check
        if not win32gui.IsWindow(self.hwnd):
            raise GameCrashedError("Window lost during step")

        # 2. Get current position BEFORE moving (for eagle check)
        frame_initial = self.grab_frame()
        pos_initial = self.find_chicken(frame_initial)

        # 3. Eagle Check / Input Cutoff
        if pos_initial:
            if self.check_eagle_death(pos_initial, frame_initial.shape[1]):
                # CUT INPUTS. Accept Death.
                print("[ENV] Eagle Line Crossed! Cutting inputs.")
                return np.zeros(self.state_dim), -10, True

        # 4. Execute Action
        # Masking logic happens in trainer, but we double check here:
        if self.steps_in_episode == 0 and action == 2:
            print("[ENV] Warning: 'Left' attempted on first move. Ignoring.")
            action = 0  # Force Idle

        if action == 1:
            pydirectinput.press('up')
        elif action == 2:
            pydirectinput.press('left')
        elif action == 3:
            pydirectinput.press('right')

        self.steps_in_episode += 1

        # 5. Sense Result
        frame = self.grab_frame()
        pos = self.find_chicken(frame)

        done = False
        reward = 0

        is_alive = self.is_pause_visible(frame)

        # Post-move Eagle check
        eagle_death = False
        if pos and self.check_eagle_death(pos, frame.shape[1]):
            eagle_death = True

        if not is_alive or eagle_death:
            done = True
            reward = -10
        else:
            if action == 1:
                reward = 1.0
            elif action == 0:
                reward = -0.1
            else:
                reward = -0.01
            if pos: reward += (1.0 - (pos[1] / frame.shape[0])) * 0.5

        state, self.last_debug_grid = self.get_state(frame, pos)

        # 6. Render Debug Window
        self.render_debug(frame, pos, eagle_death)

        return state, reward, done

    def render_debug(self, frame, chicken_pos, eagle_death):
        display = frame.copy()
        h, w, _ = display.shape

        # Draw Eagle Line
        y1 = EAGLE_Y_INTERCEPT
        y2 = int(self.slope * w + EAGLE_Y_INTERCEPT)
        cv2.line(display, (0, y1), (w, y2), (0, 0, 255), 2)

        # Draw Grid
        if self.last_debug_grid:
            for (px, py, t_type) in self.last_debug_grid:
                color = (200, 200, 200)
                if t_type == Terrain.GRASS:
                    color = (0, 255, 0)
                elif t_type == Terrain.ROAD:
                    color = (50, 50, 50)
                elif t_type == Terrain.WATER:
                    color = (255, 100, 0)
                elif t_type == Terrain.RAIL:
                    color = (0, 0, 100)
                cv2.circle(display, (px, py), 4, color, -1)

        # Draw Stats
        cv2.putText(display, f"Step: {self.steps_in_episode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                    2)

        status = "ALIVE"
        if eagle_death:
            status = "EAGLE DEATH"
        elif not self.is_pause_visible(frame):
            status = "DEAD"

        cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if "DEAD" in status else (0, 255, 0), 2)

        # Resize for desktop viewing if needed
        cv2.imshow("CrossyLearn Debug", display)
        cv2.waitKey(1)
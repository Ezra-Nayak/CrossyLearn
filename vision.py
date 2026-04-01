# --- vision.py ---

import cv2
import numpy as np
import mss
import win32gui
import threading
import queue
import time
import os
import math

# Import our modules
from tracker import ChickenTracker
from terrain_analyzer import get_terrain_type, Terrain


class VisionSystem(threading.Thread):
    def __init__(self, window_title, output_queue, show_preview=True):
        super().__init__()
        self.daemon = True
        self.window_title = window_title
        self.output_queue = output_queue
        self.running = True
        self.show_preview = show_preview

        # --- CONFIGURATION ---
        self.VISION_PARAMS = {
            'LOWER_BOUND': np.array([170, 125, 21]),
            'UPPER_BOUND': np.array([179, 136, 37]),
            'SEARCH_ZONE_Y_INTERCEPT': 310,
            'LINE_ANGLE_DEG': 15,
        }
        self.PENALTY_LINE_Y_INTERCEPT = 720

        # Feeler Vectors (From your Hop Calibrator)
        self.FORWARD_VEC = np.array([28, -70])
        self.RIGHT_VEC = np.array([80, 20])

        # Initialize Components
        self.tracker = ChickenTracker(self.VISION_PARAMS)
        self.digit_templates = self._load_digit_templates()

        self.retry_template = cv2.imread('templates/retry_button.png')
        if self.retry_template is None:
            print("WARNING: 'templates/retry_button.png' not found. Legacy death detection will fail.")

    def _load_digit_templates(self):
        templates = {}
        template_dir = 'templates'
        if not os.path.exists(template_dir):
            print(f"WARNING: Template directory '{template_dir}' not found. OCR will fail.")
            return {}

        for i in range(10):
            filepath = os.path.join(template_dir, f"{i}.png")
            if os.path.exists(filepath):
                templates[i] = cv2.imread(filepath, 0)
        print(f"Vision: Loaded {len(templates)} digit templates.")
        return templates

    def recognize_score(self, image):
        # Hardcoded ROI based on your original main.py
        try:
            roi = image[10:121, 10:233]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
        except Exception:
            return None

        TEMPLATE_HEIGHT = 40
        contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_digits = []

        for contour in contours:
            if cv2.contourArea(contour) < 50: continue
            x, y, w, h = cv2.boundingRect(contour)

            if h == 0: continue
            aspect_ratio = w / h
            new_width = int(TEMPLATE_HEIGHT * aspect_ratio)
            if new_width <= 0: continue

            digit_crop = thresh_roi[y:y + h, x:x + w]
            standardized_digit = cv2.resize(digit_crop, (new_width, TEMPLATE_HEIGHT), interpolation=cv2.INTER_AREA)

            best_match_score, best_match_digit = -1, -1
            for digit_val, template in self.digit_templates.items():
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

    def detect_retry_button(self, image):
        # Legacy slow detection (Backup)
        if self.retry_template is None: return False
        x, y, w, h = 687, 864, 156, 98
        margin = 10

        if y - margin < 0 or y + h + margin > image.shape[0] or x - margin < 0 or x + w + margin > image.shape[1]:
            return False

        search_area = image[y - margin: y + h + margin, x - margin: x + w + margin]
        if search_area.shape[0] < self.retry_template.shape[0] or search_area.shape[1] < self.retry_template.shape[1]:
            return False

        res = cv2.matchTemplate(search_area, self.retry_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val > 0.8

    def detect_pause_button(self, frame):
        """
        Detects the two white vertical bars in the top right corner.
        Uses a dual-patch sampling method to avoid contour merging on chaotic backgrounds.
        Scans a small section of the left pause bar and a small section of the right pause bar.
        Returns True if visible (Game Active), False if missing (Dead/Menu).
        """
        h, w, _ = frame.shape

        # --- TUNED PARAMETERS ---
        THRESH_VAL = 220
        PATCH_SIZE = 8

        # Offsets from the Top-Right corner (w, 0)
        L_X_DIST = 73  # Left bar patch X distance from right edge
        L_Y_DIST = 152  # Left bar patch Y distance from top edge

        R_X_DIST = 45  # Right bar patch X distance from right edge
        R_Y_DIST = 120  # Right bar patch Y distance from top edge
        # ------------------------

        lx1, lx2 = w - L_X_DIST, w - L_X_DIST + PATCH_SIZE
        ly1, ly2 = L_Y_DIST, L_Y_DIST + PATCH_SIZE

        rx1, rx2 = w - R_X_DIST, w - R_X_DIST + PATCH_SIZE
        ry1, ry2 = R_Y_DIST, R_Y_DIST + PATCH_SIZE

        # Bounds check
        if lx1 < 0 or rx1 < 0 or ly2 > h or ry2 > h:
            return False

        left_patch = frame[ly1:ly2, lx1:lx2]
        right_patch = frame[ry1:ry2, rx1:rx2]

        left_gray = cv2.cvtColor(left_patch, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_patch, cv2.COLOR_BGR2GRAY)

        _, left_thresh = cv2.threshold(left_gray, THRESH_VAL, 255, cv2.THRESH_BINARY)
        _, right_thresh = cv2.threshold(right_gray, THRESH_VAL, 255, cv2.THRESH_BINARY)

        # Require 80% of the pixels in both patches to be pure white
        left_score = np.count_nonzero(left_thresh) / (PATCH_SIZE * PATCH_SIZE)
        right_score = np.count_nonzero(right_thresh) / (PATCH_SIZE * PATCH_SIZE)

        return (left_score > 0.8) and (right_score > 0.8)

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
                    TITLE_BAR_HEIGHT = 50
                    monitor = {
                        "top": client_top + TITLE_BAR_HEIGHT, "left": client_left,
                        "width": right - left, "height": bottom - top - TITLE_BAR_HEIGHT
                    }

                    if monitor['width'] <= 0 or monitor['height'] <= 0:
                        time.sleep(0.1)
                        continue

                    # Capture and Color Convert
                    game_frame_bgra = np.array(sct.grab(monitor))
                    game_frame = cv2.cvtColor(game_frame_bgra, cv2.COLOR_BGRA2BGR)
                    frame_h, frame_w, _ = game_frame.shape

                    # --- 1. TRACKING (w/ Anti-Squish) ---
                    chicken_pos, chicken_box, track_status = self.tracker.track(game_frame)

                    # --- 2. SCORE & STATUS ---
                    score_val = self.recognize_score(game_frame)
                    retry_visible = self.detect_retry_button(game_frame)
                    pause_visible = self.detect_pause_button(game_frame)

                    # --- 3. TERRAIN FEELERS ---
                    terrain_feelers = {
                        "forward": Terrain.UNKNOWN,
                        "left": Terrain.UNKNOWN,
                        "right": Terrain.UNKNOWN,
                        "forward_left": Terrain.UNKNOWN,
                        "forward_right": Terrain.UNKNOWN
                    }

                    if chicken_pos:
                        c_vec = np.array(chicken_pos)

                        # Calculate vector positions
                        vecs = {
                            "forward": c_vec + self.FORWARD_VEC,
                            "left": c_vec - self.RIGHT_VEC,
                            "right": c_vec + self.RIGHT_VEC,
                            "forward_left": c_vec + self.FORWARD_VEC - self.RIGHT_VEC,
                            "forward_right": c_vec + self.FORWARD_VEC + self.RIGHT_VEC
                        }

                        for name, vec in vecs.items():
                            px, py = int(vec[0]), int(vec[1])
                            PATCH_SIZE = 2

                            # Boundary Checks
                            if 0 < px < frame_w and 0 < py < frame_h:
                                y1, y2 = py - PATCH_SIZE // 2, py + PATCH_SIZE // 2
                                x1, x2 = px - PATCH_SIZE // 2, px + PATCH_SIZE // 2
                                patch = game_frame[y1:y2, x1:x2]
                                if patch.size > 0:
                                    terrain_feelers[name] = get_terrain_type(patch)

                    # --- 4. PENALTY LINE LOGIC ---
                    angle_rad = math.radians(self.VISION_PARAMS['LINE_ANGLE_DEG'])
                    slope = math.tan(angle_rad)
                    penalty_detected = False

                    if chicken_pos:
                        cx, cy = chicken_pos
                        penalty_line_y = int(slope * cx + self.PENALTY_LINE_Y_INTERCEPT)
                        if cy > penalty_line_y:
                            penalty_detected = True

                    p_x1, p_y1 = 0, self.PENALTY_LINE_Y_INTERCEPT
                    p_x2, p_y2 = frame_w, int(slope * frame_w + self.PENALTY_LINE_Y_INTERCEPT)

                    # --- 5. PACK DATA ---
                    result = {
                        "frame": game_frame,
                        "score": score_val,
                        "retry_visible": retry_visible,
                        "pause_visible": pause_visible,
                        "in_penalty_zone": penalty_detected,
                        "chicken_pos": chicken_pos,
                        "chicken_box": chicken_box,
                        "track_status": track_status,
                        "penalty_line_pts": ((p_x1, p_y1), (p_x2, p_y2)),
                        "terrain": terrain_feelers
                    }

                    # --- 6. OPTIONAL PREVIEW (DEBUGGING) ---
                    if self.show_preview:
                        display = game_frame.copy()
                        # Draw Opaque HUD
                        cv2.rectangle(display, (0, 0), (350, 150), (0, 0, 0), -1)

                        state_color = (0, 255, 0) if pause_visible else (0, 0, 255)
                        state_text = "ALIVE" if pause_visible else "DEAD/MENU"
                        cv2.putText(display, f"State: {state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    state_color, 2)
                        cv2.putText(display, f"Score: {score_val}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2)

                        # Draw Box
                        if chicken_box:
                            bx, by, bw, bh = chicken_box
                            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                            cv2.putText(display, track_status, (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                        1)

                        # Draw Feelers
                        y_off = 90
                        for fname, fval in terrain_feelers.items():
                            cv2.putText(display, f"{fname}: {fval.name}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (200, 200, 200), 1)
                            y_off += 20

                        cv2.imshow("Vision Preview", display)
                        cv2.waitKey(1)

                    try:
                        # Standardize to 10 FPS (100ms cycle)
                        # We subtract a small amount (15ms) to account for capture overhead
                        # and ensure the PPO loop stays synced at 10.0 FPS.
                        time.sleep(0.085)
                        self.output_queue.put_nowait(result)
                    except queue.Full:
                        pass

                except Exception as e:
                    print(f"VisionSystem Error: {e}")
                    time.sleep(1)

    def stop(self):
        self.running = False
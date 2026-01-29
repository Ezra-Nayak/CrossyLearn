# --- tracker.py ---
import cv2
import numpy as np
import math


class ChickenTracker:
    def __init__(self, vision_params):
        self.params = vision_params

        # COASTING STATE
        self.is_locked = False
        self.last_valid_pos = None
        self.last_valid_box = None

        self.frames_since_valid_lock = 0
        # Tuned to 10 based on your logs (covers sideways hops + brief occlusions)
        self.MAX_COAST_FRAMES = 10

        self.MIN_SOLID_AREA = 20.0

    def track(self, frame):
        h, w, _ = frame.shape

        # 1. Masking
        angle_rad = math.radians(self.params['LINE_ANGLE_DEG'])
        slope = math.tan(angle_rad)
        search_y1 = self.params['SEARCH_ZONE_Y_INTERCEPT']
        search_y2 = int(slope * w + search_y1)

        search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array([[0, search_y1], [w, search_y2], [w, h], [0, h]], dtype=np.int32)
        cv2.fillPoly(search_mask, [pts], 255)

        masked = cv2.bitwise_and(frame, frame, mask=search_mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.params['LOWER_BOUND'], self.params['UPPER_BOUND'])

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > self.MIN_SOLID_AREA]

        if valid:
            # --- LOCKED ---
            c = max(valid, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            self.is_locked = True
            self.frames_since_valid_lock = 0
            self.last_valid_pos = (x + w // 2, y + h)
            self.last_valid_box = (x, y, w, h)

            return self.last_valid_pos, self.last_valid_box, "LOCKED"

        else:
            # --- COASTING OR LOST ---
            if self.is_locked:
                self.frames_since_valid_lock += 1

                if self.frames_since_valid_lock <= self.MAX_COAST_FRAMES:
                    # Freeze at last known spot (The Anti-Flicker)
                    return self.last_valid_pos, self.last_valid_box, "COASTING"
                else:
                    # Truly gone (Trucks/Trees/Eagles)
                    self.is_locked = False
                    return None, None, "LOST"
            else:
                return None, None, "SEARCHING"
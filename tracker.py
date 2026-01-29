# --- tracker.py ---

import cv2
import numpy as np
import time
import math


class ChickenTracker:
    def __init__(self, vision_params):
        """
        A robust tracker that handles the 'Squish' animation where the chicken
        disappears or shatters into small contours for 2-4 frames.
        """
        self.params = vision_params

        # State Memory
        self.is_locked = False
        self.last_valid_pos = None  # (cx, cy)
        self.last_valid_box = None  # (x, y, w, h)

        # Coasting Logic (Option B: Freezing)
        self.frames_since_valid_lock = 0
        # INCREASED: To handle fast hop sequences better
        self.MAX_COAST_FRAMES = 10

        # Landing Detection
        # LOWERED: Rails reduce the effective area of the chicken.
        # 75 was too high. 20 allows tracking on rails while still ignoring single-pixel noise.
        self.MIN_SOLID_AREA = 20.0

    def track(self, frame):
        """
        Returns:
            pos (tuple): (cx, cy) center bottom of chicken
            box (tuple): (x, y, w, h) bounding box
            status (str): 'LOCKED', 'COASTING', or 'LOST'
        """
        frame_h, frame_w, _ = frame.shape

        # 1. Create Search Mask (The Angled Line)
        angle_rad = math.radians(self.params['LINE_ANGLE_DEG'])
        slope = math.tan(angle_rad)
        search_y1 = self.params['SEARCH_ZONE_Y_INTERCEPT']
        search_y2 = int(slope * frame_w + search_y1)
        search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array([[0, search_y1], [frame_w, search_y2], [frame_w, frame_h], [0, frame_h]], dtype=np.int32)
        cv2.fillPoly(search_mask, [pts], 255)

        # 2. Color Thresholding
        search_area = cv2.bitwise_and(frame, frame, mask=search_mask)
        hsv_frame = cv2.cvtColor(search_area, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.params['LOWER_BOUND'], self.params['UPPER_BOUND'])
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 3. Filter for "Solid" Chicken
        solid_contours = [c for c in contours if cv2.contourArea(c) > self.MIN_SOLID_AREA]

        if solid_contours:
            # --- STATE: LOCKED ---
            # We found a big, solid object. Update everything.
            best_contour = max(solid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(best_contour)

            self.is_locked = True
            self.frames_since_valid_lock = 0

            self.last_valid_box = (x, y, w, h)
            # Center-Bottom of the bounding box is the most reliable anchor
            self.last_valid_pos = (x + w // 2, y + h)

            return self.last_valid_pos, self.last_valid_box, "LOCKED"

        else:
            # --- STATE: COASTING or LOST ---
            if self.is_locked:
                # We *were* locked, but lost it. Assume this is the "Squish".
                self.frames_since_valid_lock += 1

                if self.frames_since_valid_lock <= self.MAX_COAST_FRAMES:
                    # Freeze! Return the old data as if it's current.
                    # This keeps the feelers working on the previous tile.
                    return self.last_valid_pos, self.last_valid_box, "COASTING"
                else:
                    # It's been too long. The chicken is gone (Eagle? Drowned? Glitch?)
                    self.is_locked = False
                    return None, None, "LOST"
            else:
                return None, None, "SEARCHING"
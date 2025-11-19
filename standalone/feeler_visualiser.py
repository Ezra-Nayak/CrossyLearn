# --- feeler_visualiser.py ---
import cv2
import numpy as np
import mss
import win32gui
import time
import math
from standalone.terrain_analyzer import Terrain, TERRAIN_HSV_RANGES

# --- CONFIGURATION MATCHING CROSSY_ENV ---
WINDOW_TITLE = "Crossy Road"
LOWER_BOUND = np.array([175, 150, 230])
UPPER_BOUND = np.array([178, 150, 230])
AREA_MIN = 35
AREA_MAX = 185
SEARCH_ZONE_Y_INTERCEPT = 310
LINE_ANGLE_DEG = 15

# Vectors
FORWARD_VEC = np.array([28, -70])
RIGHT_VEC = np.array([80, 20])
PATCH_SIZE = 4
GRID_RADIUS = 3  # 3 means 7x7 grid

# TWEAK THIS TO ALIGN GRID WITH FEET
GRID_Y_OFFSET = 0


def get_terrain_type_fast(hsv_patch):
    if hsv_patch.shape[0] == 0 or hsv_patch.shape[1] == 0: return Terrain.UNKNOWN
    avg_hsv = np.mean(hsv_patch, axis=(0, 1))
    h, s, v = avg_hsv
    for terrain, (lower, upper) in TERRAIN_HSV_RANGES.items():
        if (lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]):
            return terrain
    return Terrain.UNKNOWN


def main():
    print("--- 7x7 Grid Visualizer ---")
    print("Press 'q' to quit.")

    angle_rad = math.radians(LINE_ANGLE_DEG)
    slope = math.tan(angle_rad)

    with mss.mss() as sct:
        while True:
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if not hwnd: time.sleep(0.5); continue

            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            client_point = win32gui.ClientToScreen(hwnd, (left, top))
            monitor = {
                "top": client_point[1] + 50, "left": client_point[0],
                "width": right - left, "height": bottom - top - 50
            }
            if monitor['width'] <= 0: continue

            frame = np.array(sct.grab(monitor))
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame_h, frame_w, _ = display_frame.shape

            # --- 1. Find Chicken ---
            search_y2 = int(slope * frame_w + SEARCH_ZONE_Y_INTERCEPT)
            search_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
            pts = np.array([[0, SEARCH_ZONE_Y_INTERCEPT], [frame_w, search_y2], [frame_w, frame_h], [0, frame_h]],
                           dtype=np.int32)
            cv2.fillPoly(search_mask, [pts], 255)

            frame_masked = cv2.bitwise_and(display_frame, display_frame, mask=search_mask)
            hsv_frame_full = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2HSV)  # Use masked for detection
            mask = cv2.inRange(hsv_frame_full, LOWER_BOUND, UPPER_BOUND)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            chicken_pos = None
            valid = [c for c in contours if AREA_MIN < cv2.contourArea(c) < AREA_MAX]
            if valid:
                c = max(valid, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                chicken_pos = (x + w // 2, y + h)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- 2. Draw Grid ---
            if chicken_pos:
                cx, cy = chicken_pos
                cy += GRID_Y_OFFSET  # Apply manual offset

                # Convert FULL frame to HSV for terrain analysis (unmasked)
                hsv_frame_analysis = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)

                for dy_grid in range(-GRID_RADIUS, GRID_RADIUS + 1):
                    for dx_grid in range(-GRID_RADIUS, GRID_RADIUS + 1):
                        offset = (RIGHT_VEC * dx_grid) + (FORWARD_VEC * -dy_grid)
                        px, py = int(cx + offset[0]), int(cy + offset[1])

                        color = (100, 100, 100)  # Gray = Unknown

                        if 0 <= py < frame_h and 0 <= px < frame_w:
                            y1, y2 = max(0, py - PATCH_SIZE), min(frame_h, py + PATCH_SIZE)
                            x1, x2 = max(0, px - PATCH_SIZE), min(frame_w, px + PATCH_SIZE)

                            patch = hsv_frame_analysis[y1:y2, x1:x2]
                            t_type = get_terrain_type_fast(patch)

                            if t_type == Terrain.GRASS:
                                color = (0, 255, 0)
                            elif t_type == Terrain.ROAD:
                                color = (50, 50, 50)  # Dark Grey
                            elif t_type == Terrain.WATER:
                                color = (255, 100, 0)  # Blue/Orange
                            elif t_type == Terrain.RAIL:
                                color = (0, 0, 100)  # Redish
                            elif t_type == Terrain.LOG:
                                color = (0, 255, 255)  # Yellow
                            elif t_type == Terrain.LILLYPAD:
                                color = (0, 100, 0)

                            # Draw dot
                        cv2.circle(display_frame, (px, py), 3, color, -1)
                        # Mark the "Chicken" center
                        if dy_grid == 0 and dx_grid == 0:
                            cv2.circle(display_frame, (px, py), 5, (255, 255, 255), 1)

            cv2.imshow('7x7 Grid Visualizer', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
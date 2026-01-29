# --- feeler_visualiser.py ---
import cv2
import numpy as np
import mss
import win32gui
import time
import math
from standalone.terrain_analyzer import Terrain, TERRAIN_HSV_RANGES

# --- CONFIGURATION ---
WINDOW_TITLE = "Crossy Road"

# YOUR TUNED VALUES
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
GRID_RADIUS = 3  # 7x7 grid


def nothing(x):
    pass


def get_terrain_type_fast(hsv_patch):
    if hsv_patch.shape[0] == 0 or hsv_patch.shape[1] == 0: return Terrain.UNKNOWN
    avg_hsv = np.mean(hsv_patch, axis=(0, 1))
    h, s, v = avg_hsv
    for terrain, (lower, upper) in TERRAIN_HSV_RANGES.items():
        if (lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]):
            return terrain
    return Terrain.UNKNOWN


def main():
    print("--- Grid Orientation Tuner ---")
    print("1. Use 'Offset Y' to move the grid DOWN until the WHITE ring is at the FEET.")
    print("2. The BLUE ring represents FORWARD (the next hop).")
    print("3. If the White Ring is on the head, the Agent thinks the air is the ground (UNKNOWN).")

    TUNER_WINDOW = "Grid Visualizer"
    TEXT_WINDOW = "Terrain Text"

    cv2.namedWindow(TUNER_WINDOW)
    cv2.namedWindow(TEXT_WINDOW)

    cv2.createTrackbar('Offset X', TUNER_WINDOW, 100, 200, nothing)
    cv2.createTrackbar('Offset Y', TUNER_WINDOW, 100, 200, nothing)

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

            text_canvas = np.zeros((400, 600, 3), dtype=np.uint8)

            # 1. Get Sliders
            val_x = cv2.getTrackbarPos('Offset X', TUNER_WINDOW)
            val_y = cv2.getTrackbarPos('Offset Y', TUNER_WINDOW)
            off_x = val_x - 100
            off_y = val_y - 100

            # 2. Find Chicken
            search_y2 = int(slope * frame_w + SEARCH_ZONE_Y_INTERCEPT)
            search_mask = np.zeros(display_frame.shape[:2], dtype=np.uint8)
            pts = np.array([[0, SEARCH_ZONE_Y_INTERCEPT], [frame_w, search_y2], [frame_w, frame_h], [0, frame_h]],
                           dtype=np.int32)
            cv2.fillPoly(search_mask, [pts], 255)

            frame_masked = cv2.bitwise_and(display_frame, display_frame, mask=search_mask)
            hsv_frame_full = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame_full, LOWER_BOUND, UPPER_BOUND)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            chicken_pos = None
            valid = [c for c in contours if AREA_MIN < cv2.contourArea(c) < AREA_MAX]
            if valid:
                c = max(valid, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                chicken_pos = (x + w // 2, y + h)
                # Draw Box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 3. Grid Analysis
            if chicken_pos:
                cx, cy = chicken_pos
                cx += off_x
                cy += off_y

                hsv_frame_analysis = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)

                for dy_grid in range(-GRID_RADIUS, GRID_RADIUS + 1):
                    for dx_grid in range(-GRID_RADIUS, GRID_RADIUS + 1):

                        offset = (RIGHT_VEC * dx_grid) + (FORWARD_VEC * -dy_grid)
                        px, py = int(cx + offset[0]), int(cy + offset[1])

                        t_type = Terrain.UNKNOWN

                        if 0 <= py < frame_h and 0 <= px < frame_w:
                            y1, y2 = max(0, py - PATCH_SIZE), min(frame_h, py + PATCH_SIZE)
                            x1, x2 = max(0, px - PATCH_SIZE), min(frame_w, px + PATCH_SIZE)

                            if y2 > y1 and x2 > x1:
                                patch = hsv_frame_analysis[y1:y2, x1:x2]
                                t_type = get_terrain_type_fast(patch)

                        dot_color = (100, 100, 100)
                        if t_type == Terrain.GRASS:
                            dot_color = (0, 255, 0)
                        elif t_type == Terrain.ROAD:
                            dot_color = (50, 50, 50)
                        elif t_type == Terrain.WATER:
                            dot_color = (255, 100, 0)
                        elif t_type == Terrain.RAIL:
                            dot_color = (0, 0, 100)

                        cv2.circle(display_frame, (px, py), 3, dot_color, -1)

                        # HIGHLIGHTS
                        if dy_grid == 0 and dx_grid == 0:
                            # CENTER (Feet) - White Ring
                            cv2.circle(display_frame, (px, py), 6, (255, 255, 255), 2)
                        elif dy_grid == 1 and dx_grid == 0:
                            # FORWARD (Next Step) - Blue Ring (dy=1 because -y is up in array logic, but dy loop is -R to +R)
                            # Wait, loop is range(-3, 4).
                            # Forward Vec is -70 Y.
                            # logic: offset = ... + FORWARD_VEC * -dy_grid
                            # To go FORWARD, we need positive FORWARD_VEC component.
                            # So -dy_grid must be positive. dy_grid must be negative.
                            # Forward is dy_grid = -1.
                            pass

                        if dy_grid == -1 and dx_grid == 0:
                            # FORWARD - Blue Ring
                            cv2.circle(display_frame, (px, py), 6, (255, 0, 0), 2)

                        # Text Window Logic
                        row_idx = dy_grid + GRID_RADIUS
                        col_idx = dx_grid + GRID_RADIUS

                        text_x = 10 + col_idx * 80
                        text_y = 40 + row_idx * 50

                        name = t_type.name
                        font_color = (200, 200, 200)

                        if dy_grid == 0 and dx_grid == 0:
                            name = f"[{name}]"
                            font_color = (0, 255, 255)
                        elif dy_grid == -1 and dx_grid == 0:
                            name = f"^{name}^"  # Mark forward
                            font_color = (255, 100, 100)

                        if name == "UNKNOWN": name = "."
                        cv2.putText(text_canvas, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, font_color, 1)

            cv2.putText(display_frame, f"X: {off_x} | Y: {off_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                        2)

            cv2.imshow(TUNER_WINDOW, display_frame)
            cv2.imshow(TEXT_WINDOW, text_canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("\n--- OFFSET TUNING COMPLETE ---")
    print(f"GRID_OFFSET_X = {off_x}")
    print(f"GRID_OFFSET_Y = {off_y}")
    print("------------------------------")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# --- feeler_visualiser.py ---

import cv2
import numpy as np
import mss
import win32gui
import time
import pywintypes
from tracker import ChickenTracker
from terrain_analyzer import Terrain, get_terrain_type

# --- CONFIGURATION ---
WINDOW_TITLE = "Crossy Road"

# Match these exactly to vision.py
VISION_PARAMS = {
    'LOWER_BOUND': np.array([170, 125, 21]),
    'UPPER_BOUND': np.array([179, 136, 37]),
    'SEARCH_ZONE_Y_INTERCEPT': 310,
    'LINE_ANGLE_DEG': 15,
}

# Calibrated Vectors
FORWARD_VEC = np.array([28, -70])
RIGHT_VEC = np.array([80, 20])
PATCH_SIZE = 2

TERRAIN_COLORS = {
    Terrain.UNKNOWN: (200, 200, 200),
    Terrain.GRASS: (0, 255, 0),
    Terrain.ROAD: (100, 100, 100),
    Terrain.WATER: (255, 100, 0),
    Terrain.RAIL: (80, 80, 120),
    Terrain.LOG: (0, 100, 150),
    Terrain.LILLYPAD: (100, 200, 0),
    Terrain.TREE: (50, 150, 100),
    Terrain.ROCK: (150, 150, 150),
}


def main():
    print("--- Feeler Visualizer (Updated) ---")
    print("Press 'q' on the window to quit.")

    hwnd = None
    tracker = ChickenTracker(VISION_PARAMS)

    with mss.mss() as sct:
        while True:
            try:
                if not hwnd or not win32gui.IsWindow(hwnd):
                    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
                    if not hwnd:
                        time.sleep(0.5)
                        continue

                # Screen capture
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                TITLE_BAR_HEIGHT = 50
                monitor = {
                    "top": client_top + TITLE_BAR_HEIGHT, "left": client_left,
                    "width": right - left, "height": bottom - top - TITLE_BAR_HEIGHT
                }

                if monitor['width'] <= 0: continue

                game_frame_bgra = np.array(sct.grab(monitor))
                display_frame = cv2.cvtColor(game_frame_bgra, cv2.COLOR_BGRA2BGR)
                frame_h, frame_w, _ = display_frame.shape

                # --- Core Logic ---
                # Fix: Unpack 3 values (pos, box, status)
                chicken_pos, chicken_box, status = tracker.track(display_frame)

                detected_terrains = {}
                feeler_positions = {}

                if chicken_pos:
                    # Draw chicken box & status
                    x, y, w, h = chicken_box
                    color = (0, 255, 0) if status == "LOCKED" else (0, 255, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # --- Calculate Feelers ---
                    c_vec = np.array(chicken_pos)

                    # Vector math
                    pos_f = c_vec + FORWARD_VEC
                    pos_l = c_vec - RIGHT_VEC
                    pos_r = c_vec + RIGHT_VEC
                    pos_fl = c_vec + FORWARD_VEC - RIGHT_VEC
                    pos_fr = c_vec + FORWARD_VEC + RIGHT_VEC

                    feeler_positions = {
                        "F": tuple(pos_f.astype(int)),
                        "L": tuple(pos_l.astype(int)),
                        "R": tuple(pos_r.astype(int)),
                        "FL": tuple(pos_fl.astype(int)),
                        "FR": tuple(pos_fr.astype(int)),
                    }

                    # --- Analyze Feelers ---
                    for name, (px, py) in feeler_positions.items():
                        y_start, y_end = py - PATCH_SIZE // 2, py + PATCH_SIZE // 2
                        x_start, x_end = px - PATCH_SIZE // 2, px + PATCH_SIZE // 2

                        terrain_type = Terrain.UNKNOWN
                        if 0 < y_start and y_end < frame_h and 0 < x_start and x_end < frame_w:
                            patch = display_frame[y_start:y_end, x_start:x_end]
                            terrain_type = get_terrain_type(patch)

                        detected_terrains[name] = terrain_type

                        # Draw the circle
                        color = TERRAIN_COLORS.get(terrain_type, (255, 255, 255))
                        cv2.circle(display_frame, (px, py), 8, color, -1)
                        cv2.circle(display_frame, (px, py), 9, (0, 0, 0), 1)  # Outline
                        cv2.putText(display_frame, name, (px - 9, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                        cv2.putText(display_frame, name, (px - 9, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1)

                # --- Draw UI Panel ---
                cv2.rectangle(display_frame, (0, 0), (250, 180), (0, 0, 0), -1)

                if chicken_pos:
                    cv2.putText(display_frame, f"Tracker: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                    y_off = 60
                    ordered = ["F", "L", "R", "FL", "FR"]
                    for name in ordered:
                        terr = detected_terrains.get(name, Terrain.UNKNOWN)
                        t_color = TERRAIN_COLORS.get(terr, (255, 255, 255))
                        cv2.putText(display_frame, f"{name}: {terr.name}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    t_color, 2)
                        y_off += 25
                else:
                    cv2.putText(display_frame, "SEARCHING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Feeler Visualizer', display_frame)

            except (pywintypes.error, win32gui.error):
                hwnd = None
                time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
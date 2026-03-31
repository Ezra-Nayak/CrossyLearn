import cv2
import numpy as np
import time
import os
import win32gui
from collections import deque
from record_runs_ram_expert import AlgorithmicExpert, ACTION_UP, ACTION_LEFT, ACTION_RIGHT
from tracker import RamTracker

EXECUTABLE_PATH = r"C:\Program Files\WindowsApps\Yodo1Ltd.CrossyRoad_1.3.4.0_x86__s3s3f300emkze\Crossy Road.exe"
WINDOW_TITLE = "Crossy Road"


def nothing(x):
    pass


class VisualKinematicTuner:
    def __init__(self):
        print("--- JUMP KINEMATICS TUNER ---")
        self.ensure_game_running()

        print("Connecting RAM Hooks...")
        self.bot = AlgorithmicExpert()
        self.ram_tracker = RamTracker()

        # Screen dimensions
        self.W, self.H = 600, 800

        # Setup OpenCV UI
        self.window_name = 'Kinematics Radar (Press Q to quit)'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.W, self.H)

        # Tunable Sliders (scaled * 100 for integer trackbars)
        cv2.createTrackbar('Wait Time (ms)', self.window_name, 200, 500, nothing)  # 0.20s
        cv2.createTrackbar('Air Time (ms)', self.window_name, 200, 500, nothing)  # DT=0.20s
        cv2.createTrackbar('Drift Momentum (%)', self.window_name, 100, 200, nothing)  # 100% inheritance

        self.player_history = deque(maxlen=30)
        self.jump_records = deque(maxlen=5)  # Stores (launch_pos, actual_land, predicted_land)

        self.is_jumping = False
        self.jump_start_pos = None
        self.jump_start_time = 0
        self.predicted_landing = None

    def ensure_game_running(self):
        hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if not hwnd:
            print("[RECOVERY] Game window not found. Launching Crossy Road...")
            os.system(f'start "" "{EXECUTABLE_PATH}"')
            time.sleep(7)
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if hwnd:
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(1)
                print("[RECOVERY] Game launched and focused.")
            else:
                print("[ERROR] Failed to find window after launch. Please start manually.")

    def world_to_screen(self, x, z, player_z):
        """Maps Game Engine Coordinates to 2D Radar Screen Coordinates."""
        # X range: [-6, 6] -> [0, W]
        sx = int(((x + 6) / 12.0) * self.W)
        # Z range: [player_z - 3, player_z + 12] -> [H, 0] (Inverted Y)
        sz = int(self.H - (((z - player_z + 3) / 15.0) * self.H))
        return sx, sz

    def get_current_log_velocity(self, px, pz, terrain_map, entities):
        """Finds the velocity of the log the player is currently standing on."""
        rz = round(pz)
        if terrain_map.get(rz, "GRASS") == "RIVER":
            for e in entities:
                if e['type'] in ['LOG', 'LILYPAD'] and abs(e['z'] - pz) < 0.5:
                    # Account for slight drifting in alignment
                    if abs(e['x'] - px) < (e['w'] / 2.0 + 0.3):
                        return e['vx']
        return 0.0

    def calculate_predicted_landing(self, px, pz, current_vx, action):
        """Calculates the predicted jump using the live slider values."""
        wait_t = cv2.getTrackbarPos('Wait Time (ms)', self.window_name) / 1000.0
        air_t = cv2.getTrackbarPos('Air Time (ms)', self.window_name) / 1000.0
        momentum = cv2.getTrackbarPos('Drift Momentum (%)', self.window_name) / 100.0

        # Drift while sitting on the log waiting to jump
        x_launch = px + (current_vx * wait_t)

        # Drift carried into the air during the jump animation
        x_land = x_launch + (current_vx * air_t * momentum)
        z_land = pz

        if action == ACTION_LEFT:
            x_land -= 1.0
        elif action == ACTION_RIGHT:
            x_land += 1.0
        elif action == ACTION_UP:
            z_land += 1.0

        return x_land, z_land

    def run(self):
        print("\n[READY] Play the game manually using your keyboard! The bot is NOT controlling.")

        last_frame_time = time.time()

        while True:
            # Sync to ~60 FPS to prevent flickering and vault data thrashing
            elapsed = time.time() - last_frame_time
            if elapsed < 0.016:
                time.sleep(0.001)
                continue
            last_frame_time = time.time()

            frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)

            # Poll state natively via RAM Tracker
            coords = self.ram_tracker.get_coords()
            if coords:
                px, py, pz = coords
            else:
                px, py, pz = 0.0, 0.0, 0.0

            # Only poll the destructive vault if we are on a fresh logic tick
            terrain_map, entities = self.bot.poll_world_state(px, pz)

            self.player_history.append((px, pz, time.time()))

            # Draw Grid & Terrain
            for z in range(int(pz - 3), int(pz + 13)):
                terrain = terrain_map.get(z, "GRASS")
                color = (40, 40, 40) if terrain == "ROAD" else (100, 50, 0) if terrain == "RIVER" else (20, 100, 20)
                y1 = self.world_to_screen(0, z - 0.5, pz)[1]
                y2 = self.world_to_screen(0, z + 0.5, pz)[1]
                cv2.rectangle(frame, (0, y2), (self.W, y1), color, -1)

            # Draw Entities
            for e in entities:
                sx1, sz1 = self.world_to_screen(e['x'] - e['w'] / 2, e['z'] - e['h'] / 2, pz)
                sx2, sz2 = self.world_to_screen(e['x'] + e['w'] / 2, e['z'] + e['h'] / 2, pz)

                color = (255, 255, 255)
                if e['type'] == 'LOG':
                    color = (40, 70, 130)  # Brownish
                elif e['type'] == 'CAR':
                    color = (0, 0, 255)  # Red
                elif e['type'] == 'LILYPAD':
                    color = (0, 200, 0)  # Green

                cv2.rectangle(frame, (sx1, sz2), (sx2, sz1), color, -1)

                # Draw Velocity Vector
                if abs(e['vx']) > 0:
                    evx_s, evz_s = self.world_to_screen(e['x'] + e['vx'], e['z'], pz)
                    cv2.arrowedLine(frame, (sx1 + (sx2 - sx1) // 2, sz1 + (sz2 - sz1) // 2), (evx_s, evz_s),
                                    (0, 255, 255), 1)

            # Jump Tracking Logic
            current_vx = self.get_current_log_velocity(px, pz, terrain_map, entities)

            # 1. Update Ground Baseline (py drops when landing, stays stable on logs)
            if not hasattr(self, 'ground_y'): self.ground_y = py
            if not self.is_jumping and py < self.ground_y + 0.1:
                self.ground_y = (self.ground_y * 0.9) + (py * 0.1)

            # 2. Detect Jump Start via Y-Height (Immune to log drift!)
            if not self.is_jumping and py > self.ground_y + 0.15:
                self.is_jumping = True

                # Grab position from ~4 frames ago to get the exact launch pad coordinates
                hist_len = len(self.player_history)
                idx = max(0, hist_len - 4)
                self.jump_start_pos = self.player_history[idx][:2]
                self.jump_start_time = time.time()
                self.jump_start_vx = current_vx  # Lock in the log velocity at the moment of launch

            # 3. Detect Jump Land (Y-Height settles back down, or timeout)
            elif self.is_jumping:
                time_in_air = time.time() - self.jump_start_time
                if (py <= self.ground_y + 0.05 and time_in_air > 0.1) or time_in_air > 0.5:
                    self.is_jumping = False

                    # Figure out actual intended action by analyzing the jump macro-movement
                    dx = px - self.jump_start_pos[0]
                    dz = pz - self.jump_start_pos[1]

                    # Subtract log drift to isolate the player's intentional input
                    drift_dx = self.jump_start_vx * time_in_air
                    net_dx = dx - drift_dx

                    action = ACTION_UP
                    if abs(dz) > 0.4:
                        action = ACTION_UP
                    elif net_dx > 0.4:
                        action = ACTION_RIGHT
                    elif net_dx < -0.4:
                        action = ACTION_LEFT

                    # Calculate the mathematical prediction based on SLIDER values
                    pred_x, pred_z = self.calculate_predicted_landing(
                        self.jump_start_pos[0], self.jump_start_pos[1], self.jump_start_vx, action
                    )

                    # Save record: (Start, Actual End, Predicted End)
                    self.jump_records.append((self.jump_start_pos, (px, pz), (pred_x, pred_z)))

            # Draw Jump History Ghosts
            for start, actual, pred in self.jump_records:
                s_actual = self.world_to_screen(actual[0], actual[1], pz)
                s_pred = self.world_to_screen(pred[0], pred[1], pz)
                s_start = self.world_to_screen(start[0], start[1], pz)

                # Draw lines
                cv2.line(frame, s_start, s_actual, (0, 255, 0), 2)  # Actual path = GREEN
                cv2.line(frame, s_start, s_pred, (0, 0, 255), 2)  # Predicted path = RED

                # Draw Endpoints
                cv2.circle(frame, s_actual, 8, (0, 255, 0), -1)  # Actual Land
                cv2.drawMarker(frame, s_pred, (0, 0, 255), cv2.MARKER_CROSS, 15, 2)  # Predicted Land

            # Draw Live Predictions (If standing still on a log)
            if not self.is_jumping and abs(current_vx) > 0.1:
                p_up = self.calculate_predicted_landing(px, pz, current_vx, ACTION_UP)
                s_up = self.world_to_screen(p_up[0], p_up[1], pz)
                cv2.drawMarker(frame, s_up, (255, 0, 255), cv2.MARKER_SQUARE, 10, 2)
                cv2.putText(frame, "Predicted UP", (s_up[0] + 10, s_up[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 0, 255), 1)

            # Draw Player
            sx, sz = self.world_to_screen(px, pz, pz)
            cv2.circle(frame, (sx, sz), 10, (255, 0, 0), -1)

            # HUD
            cv2.putText(frame, "GREEN = Actual Path", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "RED X = Predicted Path", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Log Vel: {current_vx:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                        2)

            # Show Window
            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    tuner = VisualKinematicTuner()
    tuner.run()
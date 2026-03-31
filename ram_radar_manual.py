import cv2
import numpy as np
import time
import struct
import pymem
import ctypes
import heapq
import pydirectinput

pydirectinput.PAUSE = 0

from tracker import RamTracker


class RamRadar:
    def __init__(self):
        try:
            self.pm = pymem.Pymem("Crossy Road.exe")
            print("[RADAR] Attached to Crossy Road memory.")
        except Exception as e:
            print(f"[RADAR ERROR] Could not attach: {e}")
            self.pm = None

        self.entity_vault_addr = None
        self.entity_history = {}
        if self.pm:
            self.inject_entity_tracker()

    def inject_entity_tracker(self):
        try:
            aob_sig = b"\xF3\x0F\x10\x6E\x08\xF3\x0F\x10\x81"
            module = pymem.process.module_from_name(self.pm.process_handle, "UnityPlayer.dll")
            if not module: return

            module_data = self.pm.read_bytes(module.lpBaseOfDll, module.SizeOfImage)
            sig_offset = module_data.find(aob_sig)
            if sig_offset == -1: return

            target_addr = module.lpBaseOfDll + sig_offset

            alloc_addr = self.pm.allocate(17408)
            self.entity_vault_addr = alloc_addr
            code_addr = alloc_addr + 16384

            payload = bytearray()
            payload.extend(b"\x50\x52\x89\xF2\xC1\xEA\x03\x81\xE2\xFF\x03\x00\x00\xC1\xE2\x04")
            payload.extend(b"\x81\xC2" + struct.pack("<I", self.entity_vault_addr))
            payload.extend(b"\x89\x32\x8B\x06\x89\x42\x04\x8B\x46\x04\x89\x42\x08\x8B\x46\x08\x89\x42\x0C")
            payload.extend(b"\x5A\x58\xF3\x0F\x10\x6E\x08")

            return_addr = target_addr + 5
            jmp_rel = return_addr - (code_addr + len(payload) + 5)
            payload.extend(b"\xE9" + struct.pack("<i", jmp_rel))
            self.pm.write_bytes(code_addr, bytes(payload), len(payload))

            hook = bytearray(b"\xE9" + struct.pack("<i", code_addr - (target_addr + 5)))

            kernel32 = ctypes.windll.kernel32
            old_protect = ctypes.c_ulong()
            kernel32.VirtualProtectEx(self.pm.process_handle, target_addr, 5, 0x40, ctypes.byref(old_protect))
            self.pm.write_bytes(target_addr, bytes(hook), 5)
            kernel32.VirtualProtectEx(self.pm.process_handle, target_addr, 5, old_protect, ctypes.byref(old_protect))

            print(f"[RADAR] 3D Map Injected! Vault: {hex(self.entity_vault_addr)}")
        except Exception as e:
            pass

    def poll_world_state(self, player_x, player_z):
        if not self.pm or not self.entity_vault_addr:
            return None, None

        try:
            # 1. Read the vault
            vault_data = self.pm.read_bytes(self.entity_vault_addr, 16384)

            # 2. INSTANT GARBAGE COLLECTION!
            # Wipe the vault clean so "Ghosts" (pooled objects) are erased.
            # Only actively rendering objects will be repopulated by the AOB before our next read.
            self.pm.write_bytes(self.entity_vault_addr, b'\x00' * 16384, 16384)

            current_time = time.time()
            entities = []
            terrain_votes = {}

            for i in range(1024):
                offset = i * 16
                ptr, x, y, z = struct.unpack_from("<Ifff", vault_data, offset)

                if ptr != 0:
                    # Ignore the player
                    distance_to_player = ((x - player_x) ** 2 + (z - player_z) ** 2) ** 0.5
                    if distance_to_player < 0.5:
                        continue

                    speed = 0.0
                    if ptr in self.entity_history:
                        last_x, last_time, last_speed = self.entity_history[ptr]
                        dt = current_time - last_time
                        if dt > 0.01:
                            instant_speed = (x - last_x) / dt
                            # Low-pass filter to smooth out memory jitter
                            speed = (last_speed * 0.7) + (instant_speed * 0.3)
                        else:
                            speed = last_speed

                    self.entity_history[ptr] = (x, current_time, speed)
                    is_moving = abs(speed) > 0.1

                    # GROUND TRUTH CLASSIFICATIONw
                    ent_type = "UNKNOWN"
                    if is_moving:
                        if y < 0.5:
                            ent_type = "LOG"  # Logs are ~0.14
                        else:
                            ent_type = "CAR"  # Cars ~0.73 to 0.91
                    else:
                        if y < 0.3:
                            ent_type = "LILYPAD"  # Lilypads are ~0.19
                        else:
                            # Railroad pole check: height ~1.19, X between -1.0 and 0.0
                            if abs(y - 1.19) < 0.05 and -1.0 <= x <= 0.0:
                                ent_type = "POLE"
                            else:
                                ent_type = "OBSTACLE"  # Trees/Rocks are 0.69 to 1.38

                    # Vote for the terrain type of this Z-row
                    rz = round(z)
                    if rz not in terrain_votes:
                        terrain_votes[rz] = {'ROAD': 0, 'GRASS': 0, 'RIVER': 0, 'RAILROAD': 0, 'LILYPADS': 0,
                                             'OBSTACLES': 0}

                    if ent_type == "CAR":
                        terrain_votes[rz]['ROAD'] += 1
                    elif ent_type == "LOG":
                        terrain_votes[rz]['RIVER'] += 1
                    elif ent_type == "LILYPAD":
                        terrain_votes[rz]['LILYPADS'] += 1
                        terrain_votes[rz]['RIVER'] += 1
                    elif ent_type == "OBSTACLE":
                        terrain_votes[rz]['OBSTACLES'] += 1
                        terrain_votes[rz]['GRASS'] += 1
                    elif ent_type == "POLE":
                        terrain_votes[rz]['OBSTACLES'] += 1
                        terrain_votes[rz]['GRASS'] += 1
                        if (rz + 1) not in terrain_votes:
                            terrain_votes[rz + 1] = {'ROAD': 0, 'GRASS': 0, 'RIVER': 0, 'RAILROAD': 0, 'LILYPADS': 0,
                                                     'OBSTACLES': 0}
                        terrain_votes[rz + 1]['RAILROAD'] += 10

                    entities.append({
                        'x': x, 'y': y, 'z': z, 'type': ent_type, 'speed': speed
                    })

            # Cleanup old pointers
            stale_keys = [k for k, v in self.entity_history.items() if current_time - v[1] > 0.5]
            for k in stale_keys: del self.entity_history[k]

            # Resolve Terrain rows based on Votes
            terrain_map = {}
            for tz in range(int(player_z) - 5, int(player_z) + 30):
                votes = terrain_votes.get(tz, {'ROAD': 0, 'GRASS': 0, 'RIVER': 0, 'RAILROAD': 0, 'LILYPADS': 0,
                                               'OBSTACLES': 0})

                # If lilypad AND some other non-water obstacle in the lane, cancel the lilypad's RIVER votes
                if votes['LILYPADS'] > 0 and votes['OBSTACLES'] > 0:
                    votes['RIVER'] -= votes['LILYPADS']

                if votes['RAILROAD'] > 0:
                    terrain_map[tz] = "RAILROAD"
                elif votes['RIVER'] > 0:
                    terrain_map[tz] = "RIVER"
                elif votes['ROAD'] > 0:
                    terrain_map[tz] = "ROAD"
                elif votes['GRASS'] > 0:
                    terrain_map[tz] = "GRASS"
                else:
                    # An empty row. Usually roads or grass. Let's default to GRASS visually,
                    # but the bot will treat it as a safe empty space regardless.
                    terrain_map[tz] = "EMPTY"

            return terrain_map, entities

        except Exception as e:
            return None, None

    def calculate_sota_path(self, player_x, player_z, terrain_map, entities):
        if terrain_map is None or entities is None:
            return []

        start_x = round(player_x)
        start_z = round(player_z)
        target_z = start_z + 5
        TICK_TIME = 0.15
        MAX_T = 25

        z_entities = {}
        for ent in entities:
            ez = round(ent['z'])
            if ez not in z_entities: z_entities[ez] = []
            z_entities[ez].append(ent)

        def is_safe_detailed(x, z, t):
            if x < -10 or x > 10: return False, False
            terrain = terrain_map.get(z, "EMPTY")
            ents = z_entities.get(z, [])

            landing_t = t
            hit_car = False
            on_platform = False
            is_lilypad_node = False
            hit_obstacle = False

            for ent in ents:
                pred_x = ent['x'] + (ent['speed'] * landing_t * TICK_TIME)

                if ent['type'] == "CAR":
                    if abs(pred_x - x) < 1.5: hit_car = True
                elif ent['type'] in ["OBSTACLE", "POLE"]:
                    if abs(pred_x - x) < 0.8: hit_obstacle = True
                elif ent['type'] == "LOG":
                    if abs(pred_x - x) < 1.3: on_platform = True
                elif ent['type'] == "LILYPAD":
                    if abs(ent['x'] - x) < 0.6:
                        on_platform = True
                        is_lilypad_node = True

            if hit_car or hit_obstacle: return False, False
            if terrain == "RIVER" and not on_platform: return False, False

            return True, is_lilypad_nodew

        # open_set: (f_score, time_step, x, z, path_history)
        open_set = []
        heapq.heappush(open_set, (0, 0, start_x, start_z, [(start_x, start_z)]))
        visited = set()
        visited.add((start_x, start_z, 0))

        best_path = []
        best_z = -999

        while open_set:
            f, t, cx, cz, path = heapq.heappop(open_set)

            if cz > best_z:
                best_z = cz
                best_path = path

            if cz >= target_z:
                return path

            if t >= MAX_T:
                continue

            actions = [(0, 0), (0, 1), (-1, 0), (1, 0)]
            for dx, dz in actions:
                nx = cx + dx
                nz = cz + dz
                nt = t + 1

                if (nx, nz, nt) in visited:
                    continue

                safe_status, is_lilypad = is_safe_detailed(nx, nz, nt)
                if safe_status:
                    visited.add((nx, nz, nt))

                    # Base cost g(n)
                    g = nt + (abs(nx - start_x) * 0.1)

                    # LILYPAD REWARD LOGIC:
                    # If this node is a lilypad, and we haven't already visited a lilypad in this Z-row
                    # in our current path history, apply a significant reward.
                    if is_lilypad:
                        already_hit_lily_in_row = any(
                            (p[1] == nz and any(e['type'] == "LILYPAD" and abs(e['x'] - p[0]) < 0.6
                                                for e in z_entities.get(nz, [])))
                            for p in path
                        )
                        if not already_hit_lily_in_row:
                            g -= 15.0  # Strong bias to "pick up" the lilypad

                    h = (target_z - nz) * 2.0
                    f_new = g + h

                    heapq.heappush(open_set, (f_new, nt, nx, nz, path + [(nx, nz)]))

        return best_path


def main():
    print("--- GHOST-BUSTING RADAR ---")

    chicken = RamTracker()
    radar = RamRadar()

    WIDTH = 600
    HEIGHT = 800
    SCALE = 35

    last_action_time = 0
    COOLDOWN = 0.165

    cv2.namedWindow("RAM Radar", cv2.WINDOW_AUTOSIZE)

    while True:
        coords = chicken.get_coords()
        if coords:
            player_x, _, player_z = coords
        else:
            player_x, player_z = 0.0, 0.0

        terrain_map, entities = radar.poll_world_state(player_x, player_z)
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        if terrain_map is not None and entities is not None:
            base_z = player_z - 4

            # --- 1. DRAW TERRAIN ---
            for tz, t_type in terrain_map.items():
                screen_y = HEIGHT - int((tz - base_z) * SCALE) - SCALE

                color = (20, 20, 20)  # Default dark grey for Empty Space
                if t_type == "ROAD":
                    color = (60, 60, 60)
                elif t_type == "GRASS":
                    color = (50, 150, 50)
                elif t_type == "RIVER":
                    color = (150, 50, 50)  # Blue (BGR)
                elif t_type == "RAILROAD":
                    color = (50, 50, 150)  # Dark Red/Purple for Railroad

                cv2.rectangle(frame, (0, screen_y), (WIDTH, screen_y + SCALE), color, -1)

            # --- 2. DRAW ENTITIES ---
            for ent in entities:
                screen_x = int((ent['x'] * SCALE) + (WIDTH / 2))
                screen_y = HEIGHT - int((ent['z'] - base_z) * SCALE) - int(SCALE / 2)

                if ent['type'] == "CAR":
                    pt1 = (screen_x - int(1.2 * SCALE), screen_y - int(0.4 * SCALE))
                    pt2 = (screen_x + int(1.2 * SCALE), screen_y + int(0.4 * SCALE))
                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), -1)  # RED

                elif ent['type'] == "LOG":
                    pt1 = (screen_x - int(1.5 * SCALE), screen_y - int(0.3 * SCALE))
                    pt2 = (screen_x + int(1.5 * SCALE), screen_y + int(0.3 * SCALE))
                    cv2.rectangle(frame, pt1, pt2, (30, 75, 150), -1)  # BROWN

                elif ent['type'] == "LILYPAD":
                    cv2.circle(frame, (screen_x, screen_y), int(0.4 * SCALE), (100, 200, 100), -1)  # LIGHT GREEN


                elif ent['type'] in ["OBSTACLE", "POLE"]:
                    cv2.circle(frame, (screen_x, screen_y), int(0.3 * SCALE), (30, 80, 30), -1)  # DARK GREEN

            # --- 3. DRAW CHICKEN ---
            px = int((player_x * SCALE) + (WIDTH / 2))
            pz = HEIGHT - int((player_z - base_z) * SCALE) - int(SCALE / 2)
            cv2.circle(frame, (px, pz), int(0.35 * SCALE), (255, 255, 255), -1)
            cv2.circle(frame, (px, pz - 8), int(0.15 * SCALE), (0, 0, 255), -1)

            # --- 4. SOTA PATHFINDING OVERLAY ---
            optimal_path = radar.calculate_sota_path(player_x, player_z, terrain_map, entities)
            if optimal_path and len(optimal_path) > 1:
                # Draw dynamic path steps using OpenCV
                for i in range(len(optimal_path) - 1):
                    p1_x = int((optimal_path[i][0] * SCALE) + (WIDTH / 2))
                    p1_y = HEIGHT - int((optimal_path[i][1] - base_z) * SCALE) - int(SCALE / 2)
                    p2_x = int((optimal_path[i + 1][0] * SCALE) + (WIDTH / 2))
                    p2_y = HEIGHT - int((optimal_path[i + 1][1] - base_z) * SCALE) - int(SCALE / 2)

                    # Highlight path connections and nodes
                    cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 255, 255), 3)  # Yellow line
                    cv2.circle(frame, (p2_x, p2_y), 5, (0, 200, 255), -1)  # Orange nodes

            # HUD
            cv2.putText(frame, f"Z: {player_z:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            cv2.putText(frame, f"Entities: {len(entities)}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200),
                        1)

            # --- 5. EXECUTE ADAPTIVE MOVE (NON-BLOCKING) ---
            current_time = time.time()
            # Only take over when Z-score (player_z) is above 5
            if player_z > 5 and len(optimal_path) > 1 and (current_time - last_action_time) >= COOLDOWN:
                curr_node = optimal_path[0]
                next_node = optimal_path[1]

                dx = next_node[0] - curr_node[0]
                dz = next_node[1] - curr_node[1]

                # Only move if the path suggests a hop (prevents spamming 'Wait' actions)
                if dx != 0 or dz != 0:
                    if dz > 0:
                        pydirectinput.press('w')
                    elif dx > 0:
                        pydirectinput.press('d')
                    elif dx < 0:
                        pydirectinput.press('a')
                    # 's' removed per instruction

                    last_action_time = current_time

        else:
            cv2.putText(frame, "Waiting for Unity 3D Engine...", (150, HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

        cv2.imshow("RAM Radar", frame)
        if cv2.waitKey(33) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
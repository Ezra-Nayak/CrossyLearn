import time
import os
import torch
import pymem
import struct
from train_ppo import CrossyGameEnv

# Action Map: 0:Up, 1:Left, 2:Right, 3:Idle
ACTION_UP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_IDLE = 3


class AlgorithmicExpert:
    def __init__(self):
        self.TERRAIN_MANAGER_BASE = 0x00000000
        self.entity_vault_addr = None
        self.entity_history = {}
        self.known_terrain = {}  # Persistent terrain classification
        self.lilypad_memory = {}  # Persistent grid memory for lilypads: {rounded_z: (x, width)}
        self.pm = None
        self.attach_and_inject()

    def attach_and_inject(self):
        """Attaches to the game process and injects the entity tracker."""
        try:
            self.pm = pymem.Pymem("Crossy Road.exe")
            print("[RAM BOT] Attached to Crossy Road memory.")
        except Exception:
            self.pm = None
            self.entity_vault_addr = None
            return False

        try:
            aob_sig = b"\xF3\x0F\x10\x6E\x08\xF3\x0F\x10\x81"
            module = pymem.process.module_from_name(self.pm.process_handle, "UnityPlayer.dll")
            if not module:
                print("[RAM ERROR] Could not find UnityPlayer.dll")
                self.pm = None
                return False

            module_data = self.pm.read_bytes(module.lpBaseOfDll, module.SizeOfImage)
            sig_offset = module_data.find(aob_sig)

            if sig_offset == -1:
                print("[RAM ERROR] Could not find 3D Engine AOB signature. (Already injected?)")
                self.pm = None
                return False

            target_addr = module.lpBaseOfDll + sig_offset
            alloc_addr = self.pm.allocate(17408)
            self.entity_vault_addr = alloc_addr
            code_addr = alloc_addr + 16384

            payload = bytearray()
            payload.extend(b"\x50\x52")
            payload.extend(b"\x89\xF2")
            payload.extend(b"\xC1\xEA\x03")
            payload.extend(b"\x81\xE2\xFF\x03\x00\x00")
            payload.extend(b"\xC1\xE2\x04")
            payload.extend(b"\x81\xC2" + struct.pack("<I", self.entity_vault_addr))
            payload.extend(b"\x89\x32")
            payload.extend(b"\x8B\x06\x89\x42\x04")
            payload.extend(b"\x8B\x46\x04\x89\x42\x08")
            payload.extend(b"\x8B\x46\x08\x89\x42\x0C")
            payload.extend(b"\x5A\x58")
            payload.extend(b"\xF3\x0F\x10\x6E\x08")

            return_addr = target_addr + 5
            jmp_rel = return_addr - (code_addr + len(payload) + 5)
            payload.extend(b"\xE9" + struct.pack("<i", jmp_rel))

            self.pm.write_bytes(code_addr, bytes(payload), len(payload))

            hook = bytearray()
            hook.extend(b"\xE9" + struct.pack("<i", code_addr - (target_addr + 5)))

            import ctypes
            kernel32 = ctypes.windll.kernel32
            old_protect = ctypes.c_ulong()
            kernel32.VirtualProtectEx(self.pm.process_handle, target_addr, 5, 0x40, ctypes.byref(old_protect))
            self.pm.write_bytes(target_addr, bytes(hook), 5)
            kernel32.VirtualProtectEx(self.pm.process_handle, target_addr, 5, old_protect, ctypes.byref(old_protect))

            print(f"[RAM] 3D Geometry Hash Map injected! Vault at {hex(self.entity_vault_addr)}")
            return True

        except Exception as e:
            print(f"[RAM ERROR] Injection Failed: {e}")
            self.entity_vault_addr = None
            self.pm = None
            return False

    def poll_world_state(self, player_x, player_z):
        if not self.pm or not self.entity_vault_addr:
            if not self.attach_and_inject():
                return {},[]

        try:
            vault_data = self.pm.read_bytes(self.entity_vault_addr, 16384)
            self.pm.write_bytes(self.entity_vault_addr, b'\x00' * 16384, 16384)

            current_time = time.time()
            entities = []
            terrain_map = {}
            solid_lanes = set()

            # 1. Scan the vault for all current entities
            for i in range(1024):
                offset = i * 16
                ptr, x, y, z = struct.unpack_from("<Ifff", vault_data, offset)

                if ptr != 0:
                    # Update Velocity History
                    hist = self.entity_history.get(ptr, [])
                    if isinstance(hist, tuple): hist = []  # Defensive type check
                    hist.append((x, z, current_time))
                    if len(hist) > 10:
                        hist.pop(0)
                    self.entity_history[ptr] = hist

                    vx, vz = 0.0, 0.0
                    if len(hist) >= 3:
                        oldest_x, oldest_z, oldest_t = hist[0]
                        dt = current_time - oldest_t
                        if dt > 0.05:
                            vx = (x - oldest_x) / dt
                            vz = (z - oldest_z) / dt
                            # Raised cap from 30.0 to 50.0 to allow ultra-fast Train tracking
                            if abs(vx) > 50.0: vx, vz = 0.0, 0.0

                    try:
                        bounds_data = self.pm.read_bytes(ptr + 0x0C, 12)
                        w, h, d = struct.unpack("<fff", bounds_data)
                    except Exception:
                        continue

                    rz = round(z)
                    is_moving = abs(vx) > 0.5 or abs(vz) > 0.5

                    # ENGINE PHYSICS TERRAIN DETECTOR:
                    # Solid ground lanes have static physical colliders spanning the map.
                    if w >= 12.0 and not is_moving:
                        if h >= 0.1 and y > -0.5:
                            solid_lanes.add(rz)
                        continue

                    # Ignore the Player
                    if abs(x - player_x) < 0.5 and abs(z - player_z) < 0.5 and h < 0.5:
                        continue

                        # Ignore invisible or under-map pooled objects (Stricter Y check for fake lilypads)
                    if y < -0.4 or w < 0.05 or h < 0.01:
                        continue

                        # Detect Railway Signal Poles (High Y, thin width, static)
                    if y > 1.0 and w < 0.5 and h > 0.8 and not is_moving:
                        terrain_map[rz + 1] = "RAIL"
                        continue  # Do not add pole as a dodging obstacle to keep pathing clean

                    ent_type = "UNKNOWN"
                    if w > 10.0:
                        continue
                    if w < 0.25 and h < 0.05:
                        continue
                    elif h < 0.12 and w < 1.0:
                        ent_type = "LILYPAD"
                        terrain_map[rz] = "RIVER"
                        self.lilypad_memory[rz] = (x, w)
                    elif h < 0.4 and is_moving:
                        ent_type = "LOG"
                        terrain_map[rz] = "RIVER"
                    elif h >= 0.4 and is_moving:
                        if abs(vx) > 15.0:
                            ent_type = "TRAIN"
                            terrain_map[rz] = "RAIL"
                        else:
                            ent_type = "CAR"
                            terrain_map[rz] = "ROAD"
                    elif h >= 0.15 and not is_moving and w < 2.0:
                        ent_type = "OBSTACLE"
                        terrain_map[rz] = "GRASS"
                    else:
                        continue

                    entities.append({
                        'ptr': ptr, 'x': x, 'y': y, 'z': z,
                        'vx': vx, 'vz': vz, 'w': w, 'h': h, 'd': d,
                        'type': ent_type
                    })

            # 2. POST-SCAN PROCESSING (Outside the 1024 loop)
            # Fix Terrain Map: If a lane is physically solid, it CANNOT be a River.
            for tz in solid_lanes:
                if terrain_map.get(tz) == "RIVER":
                    terrain_map[tz] = "GRASS"
                    if tz in self.lilypad_memory:
                        del self.lilypad_memory[tz]

            # Purge fake lilypads from the entities list
            entities = [e for e in entities if not (e['type'] == 'LILYPAD' and round(e['z']) in solid_lanes)]

            # 3. Cleanup Stale Data
            stale_keys = [k for k, v in self.entity_history.items() if (current_time - v[-1][2]) > 0.5]
            for k in stale_keys:
                del self.entity_history[k]

            stale_lilypads = [k for k in self.lilypad_memory.keys() if k < (player_z - 5) or k > (player_z + 35)]
            for k in stale_lilypads:
                del self.lilypad_memory[k]

            # 4. Fill Grid Gaps
            min_z, max_z = int(player_z) - 5, int(player_z) + 30
            for tz in range(min_z, max_z):
                if tz not in terrain_map:
                    terrain_map[tz] = "GRASS"

            return terrain_map, entities

        except Exception as e:
            print(f"[POLL ERROR] {e}")
            self.pm = None
            self.entity_vault_addr = None
            return {}, []

    @staticmethod
    def _check_temporal_overlap(car_x, car_v, car_w, player_x, player_w, t_start, t_end):
        """
        100% Mathematically exact Continuous Collision Detection in 1D Time Space.
        Calculates the exact time interval a moving object occupies the player's X space.
        """
        eff_w = (car_w / 2.0) + (player_w / 2.0)
        dist = player_x - car_x

        if abs(car_v) < 0.01:
            return abs(dist) < eff_w

        if car_v > 0:
            t_in = (dist - eff_w) / car_v
            t_out = (dist + eff_w) / car_v
        else:
            t_in = (dist + eff_w) / car_v
            t_out = (dist - eff_w) / car_v

        # Check if the danger interval [t_in, t_out] overlaps with presence[t_start, t_end]
        return t_in < t_end and t_out > t_start

    def calculate_perfect_action(self, player_x, player_z, action_mask, terrain_map, entities):
        """
        100% ENGINE-ACCURATE Pathfinding using 2D Continuous Collision Detection (CCD).
        Upgraded to Time-based Uniform Cost Search (Dijkstra) for optimal fast pathing.
        Applies mathematical drift for log riding.
        """
        import heapq

        MAX_DEPTH = 12
        DT = 0.2

        BOUND_LEFT = -4.5
        BOUND_RIGHT = 4.5
        ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_RIGHT, ACTION_IDLE]

        # Priority Queue: (time_t, -score, x, z, path)
        queue = []
        heapq.heappush(queue, (0.0, 0.0, player_x, player_z, []))
        visited = set()

        best_score = -float('inf')
        best_first_action = ACTION_IDLE
        longest_survival_depth = 0

        while queue:
            t, neg_score, x, z, path = heapq.heappop(queue)
            score = -neg_score
            depth = len(path)

            # Prioritize paths that reach the deepest depth safely, breaking ties with the highest score
            if depth > longest_survival_depth:
                longest_survival_depth = depth
                best_score = score
                best_first_action = path[0] if path else ACTION_IDLE
            elif depth == longest_survival_depth and depth > 0:
                if score > best_score:
                    best_score = score
                    best_first_action = path[0]

            if depth >= MAX_DEPTH:
                continue

            is_standing_on_lilypad = False
            rz_current = round(z)
            if rz_current in self.lilypad_memory:
                lx, lw = self.lilypad_memory[rz_current]
                if abs(x - lx) < (lw / 2.0 + 0.1):
                    is_standing_on_lilypad = True

            # Calculate if we are standing on a moving platform (Log/Lilypad) to accurately calculate drift
            current_vx = 0.0
            if terrain_map.get(rz_current, "GRASS") == "RIVER":
                for e in entities:
                    if e['type'] in ['LOG', 'LILYPAD'] and abs(e['z'] - z) < 0.5:
                        log_x_at_t = e['x'] + (e['vx'] * t)
                        if abs(log_x_at_t - x) < (e['w'] / 2.0 + 0.1):
                            current_vx = e['vx']
                            break

            for action in ACTIONS:
                if is_standing_on_lilypad and action in [ACTION_LEFT, ACTION_RIGHT]:
                    continue
                if depth == 0 and action_mask[action] < -1.0:
                    continue

                nx, nz = x, z
                if action == ACTION_UP:
                    nz += 1
                elif action == ACTION_LEFT:
                    nx -= 1
                elif action == ACTION_RIGHT:
                    nx += 1

                if action == ACTION_IDLE:
                    nt = t + 0.20
                    # Apply log drift during idle state!
                    x_land = x + (current_vx * 0.20)
                else:
                    nt = t + DT
                    # Apply log drift momentum during the jump air-time!
                    x_land = x + (current_vx * DT)
                    if action == ACTION_LEFT:
                        x_land -= 1.0
                    elif action == ACTION_RIGHT:
                        x_land += 1.0

                if x_land < BOUND_LEFT or x_land > BOUND_RIGHT:
                    continue

                tz = round(nz)
                terrain = terrain_map.get(tz, "GRASS")
                local_ents = [e for e in entities if abs(e['z'] - z) < 0.5 or abs(e['z'] - nz) < 0.5]

                is_safe = True
                on_platform = False

                # 1. mathematically perfect 1D Temporal Sweeping (Shadow Casting)
                if action == ACTION_IDLE:
                    t_start = t
                    t_end = t + 0.25

                    for e in local_ents:
                        if e['type'] in ['CAR', 'TRAIN', 'OBSTACLE'] and abs(e['z'] - z) < 0.5:
                            speed_pad = abs(e['vx']) * 0.05
                            base_pad = 0.8 if e['type'] in ['CAR', 'TRAIN'] else 0.2
                            eff_w = e['w'] + base_pad + speed_pad
                            if AlgorithmicExpert._check_temporal_overlap(e['x'], e['vx'], eff_w, x_land, 0.8, t_start,
                                                                         t_end):
                                is_safe = False
                                break
                else:
                    t_wait_start = t
                    t_air_mid = t + (DT / 2.0)
                    t_grace_end = nt + 0.15

                    for e in local_ents:
                        if e['type'] not in ['CAR', 'TRAIN', 'OBSTACLE']: continue

                        speed_pad = abs(e['vx']) * 0.05
                        base_pad = 0.8 if e['type'] in ['CAR', 'TRAIN'] else 0.2
                        eff_w = e['w'] + base_pad + speed_pad

                        if abs(e['z'] - z) < 0.5:
                            if AlgorithmicExpert._check_temporal_overlap(e['x'], e['vx'], eff_w, x, 0.8, t_wait_start,
                                                                         t_air_mid):
                                is_safe = False
                                break

                        if abs(e['z'] - nz) < 0.5 and is_safe:
                            if AlgorithmicExpert._check_temporal_overlap(e['x'], e['vx'], eff_w, x_land, 0.8, t_air_mid,
                                                                         t_grace_end):
                                is_safe = False
                                break

                # 2. End of Jump Destination Check (River Log Landing)
                if is_safe and terrain == "RIVER":
                    for e in local_ents:
                        if e['type'] in ['LOG', 'LILYPAD'] and abs(e['z'] - nz) < 0.5:
                            log_x_at_nt = e['x'] + (e['vx'] * nt)
                            # CURES LOG SUICIDE: Compare final landing position vs final log position
                            if abs(log_x_at_nt - x_land) < (e['w'] / 2.0 - 0.15):
                                on_platform = True
                                break
                    if not on_platform:
                        is_safe = False

                if is_safe:
                    # CURES STALE PATH PRUNING: Quantize Time in state key so identical locations at different times are verified!
                    state_key = (round(x_land * 2) / 2.0, round(nz), round(nt * 5))
                    if state_key not in visited:
                        visited.add(state_key)

                        # Heavy reward for moving forward, penalty for time elapsed and moving away from center
                        new_score = (nz - player_z) * 10.0 - abs(x_land) * 2.5 - (nt * 2.0)

                        # Push to Priority Queue (t dictates expansion order, -new_score tracks best paths)
                        heapq.heappush(queue, (nt, -new_score, x_land, nz, path + [action]))

        return best_first_action


def main():
    os.makedirs("expert_data", exist_ok=True)
    print("--- AUTOMATED RAM-EXPERT RECORDER ---")

    env = CrossyGameEnv(ui=None)
    bot = AlgorithmicExpert()

    if not bot.pm:
        print("[WARNING] RAM hook failed. Script will run, but bot will always IDLE.")

    print("\n[READY] Focus Crossy Road. Bot will play automatically.")
    print("Press CTRL+C in this terminal to stop recording.\n")

    episodes_recorded = 0

    try:
        while True:
            # FIX: Adopt the standard RL Loop to prevent frame skipping
            latents, scalars, action_mask = env.reset()
            trajectory = []

            # The env.reset() loop waits for the UI to clear,
            # now we are guaranteed to start cleanly on frame 1.
            print("\n[REC] Automated run started. Recording...")

            while True:
                player_x = env.last_known_coords[0]
                player_z = env.last_known_coords[1]

                action = ACTION_IDLE
                terrain_ahead = "UNKNOWN"
                hazards_ahead = []

                terrain_map, entities = bot.poll_world_state(player_x, player_z)
                if terrain_map or entities:
                    terrain_ahead = terrain_map.get(round(player_z + 1), "GRASS")
                    action = bot.calculate_perfect_action(player_x, player_z, action_mask, terrain_map, entities)
                    hazards_ahead = [e for e in entities if abs(e['z'] - (player_z + 1)) <= 0.8]

                # Record BEFORE stepping (Matches vision state to intended action)
                trajectory.append({
                    'latents': latents,
                    'scalars': scalars,
                    'mask': action_mask,
                    'action': action
                })

                action_str = ['UP  ', 'LEFT', 'RGHT', 'IDLE'][action]

                print(
                    f"\r[RADAR] Ter: {terrain_ahead[:5]:<5} | Haz: {len(hazards_ahead):02d} | Z: {player_z:.0f} | Act: {action_str}    ",
                    end="")

                # Step physically advances the game and grabs the NEXT frame
                latents, scalars, reward, done, action_mask = env.step(action)

                if done:
                    # FIX: Wait a beat before saving/resetting to ensure game state settles
                    time.sleep(1)
                    if len(trajectory) > 60:
                        episodes_recorded += 1
                        filename = f"expert_data/ram_bot_run_{int(time.time())}.pt"
                        torch.save(trajectory, filename)
                        print(
                            f"\n[SAVE] Bot died. Saved {len(trajectory)} perfect frames to {filename} (Total Runs: {episodes_recorded})")
                    else:
                        print("\n[SKIP] Run too short, discarded.")

                    break  # Break out of inner loop to trigger env.reset()

    except KeyboardInterrupt:
        print(f"\n\n[SHUTDOWN] Exiting. Successfully recorded {episodes_recorded} perfect runs.")


if __name__ == "__main__":
    main()
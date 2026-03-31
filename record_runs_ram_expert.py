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
        try:
            self.pm = pymem.Pymem("Crossy Road.exe")
            print("[RAM BOT] Attached to Crossy Road memory.")
        except Exception as e:
            print(f"[RAM BOT ERROR] Could not attach to game: {e}")
            self.pm = None

        self.TERRAIN_MANAGER_BASE = 0x00000000
        self.entity_vault_addr = None
        self.entity_history = {}
        self.known_terrain = {}  # Persistent terrain classification
        self.lilypad_memory = {}  # Persistent grid memory for lilypads: {rounded_z: (x, width)}
        if self.pm:
            self.inject_entity_tracker()

    def inject_entity_tracker(self):
        """Injects an Assembly Hash Map into the 3D Engine Transform Loop."""
        try:
            aob_sig = b"\xF3\x0F\x10\x6E\x08\xF3\x0F\x10\x81"
            module = pymem.process.module_from_name(self.pm.process_handle, "UnityPlayer.dll")
            if not module:
                print("[RAM ERROR] Could not find UnityPlayer.dll")
                return

            module_data = self.pm.read_bytes(module.lpBaseOfDll, module.SizeOfImage)
            sig_offset = module_data.find(aob_sig)

            if sig_offset == -1:
                print("[RAM ERROR] Could not find 3D Engine AOB signature. (Already injected?)")
                # Assuming already injected if we can't find the original AOB
                return

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

        except Exception as e:
            print(f"[RAM ERROR] Injection Failed: {e}")
            self.entity_vault_addr = None

    def poll_world_state(self, player_x, player_z):
        if not self.pm or not self.entity_vault_addr:
            return {}, []

        try:
            vault_data = self.pm.read_bytes(self.entity_vault_addr, 16384)
            self.pm.write_bytes(self.entity_vault_addr, b'\x00' * 16384, 16384)

            current_time = time.time()
            entities = []
            terrain_map = {}

            for i in range(1024):
                offset = i * 16
                ptr, x, y, z = struct.unpack_from("<Ifff", vault_data, offset)

                if ptr != 0:
                    vx, vz = 0.0, 0.0

                    hist = self.entity_history.get(ptr, None)
                    # Length 6: (lx, ly, lz, lt, last_vx, last_vz)
                    if hist and len(hist) == 6:
                        lx, ly, lz, lt, prev_vx, prev_vz = hist
                        dt = current_time - lt

                        # Wait at least 19ms (Engine Tick 16.6ms + margin) to calculate derivative
                        # This PREVENTS the "Velocity Jitter" caused by fast Python loops!
                        if dt >= 0.019:
                            raw_vx = (x - lx) / dt
                            raw_vz = (z - lz) / dt

                            # Ignore teleportations (map looping)
                            if abs(raw_vx) > 30.0: raw_vx = prev_vx

                            # Smooth the velocity using a Moving Average
                            vx = (prev_vx * 0.5) + (raw_vx * 0.5)
                            vz = (prev_vz * 0.5) + (raw_vz * 0.5)

                            self.entity_history[ptr] = (x, y, z, current_time, vx, vz)
                        else:
                            # Not enough time passed. Keep accumulating dt, keep old velocity.
                            vx, vz = prev_vx, prev_vz
                    else:
                        # First time seeing entity
                        self.entity_history[ptr] = (x, y, z, current_time, 0.0, 0.0)

                    try:
                        bounds_data = self.pm.read_bytes(ptr + 0x0C, 12)
                        w, h, d = struct.unpack("<fff", bounds_data)
                    except Exception:
                        continue

                    # Ignore the Player
                    if abs(x - player_x) < 0.5 and abs(z - player_z) < 0.5 and h < 0.5:
                        continue

                    ent_type = "UNKNOWN"
                    is_moving = abs(vx) > 0.5 or abs(vz) > 0.5
                    rz = round(z)

                    if w > 10.0:
                        continue
                    elif h < 0.12 and w < 1.0:
                        # ACTUAL LILYPAD: Height ~0.062.
                        ent_type = "LILYPAD"
                        terrain_map[rz] = "RIVER"
                        self.lilypad_memory[rz] = (x, w)
                    elif h < 0.4 and is_moving:
                        ent_type = "LOG"
                        terrain_map[rz] = "RIVER"
                    elif h >= 0.4 and is_moving:
                        ent_type = "CAR"
                        terrain_map[rz] = "ROAD"
                    elif h >= 0.15 and not is_moving and w < 2.0:
                        # OBSTACLE: Heights 0.188 to 0.250+.
                        ent_type = "OBSTACLE"
                        terrain_map[rz] = "GRASS"
                    else:
                        continue  # Exclude non-physical particles

                    entities.append({
                        'ptr': ptr, 'x': x, 'y': y, 'z': z,
                        'vx': vx, 'vz': vz, 'w': w, 'h': h, 'd': d,
                        'type': ent_type
                    })

            stale_keys = [k for k, v in self.entity_history.items() if current_time - v[3] > 0.5]
            for k in stale_keys: del self.entity_history[k]

            # Cleanup lilypad memory for stale rows to prevent memory bloat
            stale_lilypads = [k for k in self.lilypad_memory.keys() if k < (player_z - 5) or k > (player_z + 35)]
            for k in stale_lilypads: del self.lilypad_memory[k]

            min_z = int(player_z) - 5
            max_z = int(player_z) + 30
            for tz in range(min_z, max_z):
                if tz not in terrain_map:
                    terrain_map[tz] = "GRASS"

            return terrain_map, entities

        except Exception:
            return {}, []

    def calculate_perfect_action(self, player_x, player_z, action_mask, terrain_map, entities):
        """
        100% ENGINE-ACCURATE Pathfinding using 2D Continuous Collision Detection (CCD).
        Calculates interpolations during jump air-time to prevent tail clipping.
        """
        import collections

        MAX_DEPTH = 8
        INPUT_LATENCY = 0.3  # Time for keystroke + game engine to initiate jump
        DT = 0.18  # Air-time of the jump

        # Hitboxes based on analyzed Engine Ground Truth
        PLAYER_WIDTH = 1
        PLAYER_DEPTH = 0.6
        COLLISION_BUFFER_X = 0.032  # Displacement per tick buffer from Ground Truth report
        BOUND_LEFT = -4.5
        BOUND_RIGHT = 4.5

        ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_RIGHT, ACTION_IDLE]
        queue = collections.deque([(player_x, player_z, 0.0, [])])
        visited = set()

        best_score = -float('inf')
        best_first_action = ACTION_IDLE
        longest_survival_depth = -1

        while queue:
            x, z, t, path = queue.popleft()
            depth = len(path)

            score = (z - player_z) * 10 - abs(x) * 0.5

            if depth > longest_survival_depth:
                longest_survival_depth = depth
                best_score = score
                if path: best_first_action = path[0]
            elif depth == longest_survival_depth:
                if score > best_score:
                    best_score = score
                    if path: best_first_action = path[0]

            if depth == MAX_DEPTH:
                continue

            is_standing_on_lilypad = False
            rz_current = round(z)
            if rz_current in self.lilypad_memory:
                lx, lw = self.lilypad_memory[rz_current]
                if abs(x - lx) < (lw / 2.0 + 0.1):
                    is_standing_on_lilypad = True

            for action in ACTIONS:
                # User Tweak: Only allow UP (or IDLE) when on a lilypad
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

                if nx < BOUND_LEFT or nx > BOUND_RIGHT:
                    continue

                nt = t + DT
                tz = round(nz)
                terrain = terrain_map.get(tz, "GRASS")

                # SOTA: Grab entities in the CURRENT lane and TARGET lane to check for jump overlaps
                local_ents = [e for e in entities if abs(e['z'] - z) < 1.0 or abs(e['z'] - nz) < 1.0]

                is_safe = True
                drifted_x = nx
                on_platform = False

                # 1. Continuous Swept Collision (Checking mid-air overlaps)
                steps = 4
                for i in range(1, steps + 1):
                    fraction = i / float(steps)

                    # SOTA FIX: Shift the timeline forward by the input latency!
                    # The jump doesn't start at `t`. It starts at `t + INPUT_LATENCY`.
                    sub_t = t + INPUT_LATENCY + (DT * fraction)

                    # Interpolated chicken position mid-jump
                    cur_x = x + (nx - x) * fraction
                    cur_z = z + (nz - z) * fraction

                    for e in local_ents:
                        if e['type'] in ['CAR', 'OBSTACLE']:
                            ext_x = e['x'] + (e['vx'] * sub_t)
                            ext_z = e['z']

                            dx = abs(cur_x - ext_x)
                            dz = abs(cur_z - ext_z)

                            # Engine-verified safety buffer (accounts for 16.6ms update aliasing)
                            safety_buffer_x = COLLISION_BUFFER_X if e['type'] == 'CAR' else 0.05

                            # True AABB Overlap check using Ground Truth constants
                            if dx < (e['w'] / 2.0 + PLAYER_WIDTH / 2.0 + safety_buffer_x) and \
                                    dz < (e['d'] / 2.0 + PLAYER_DEPTH / 2.0):
                                is_safe = False
                                break
                    if not is_safe:
                        break

                # 2. End of Jump Destination Check (River Log Landing)
                if is_safe and terrain == "RIVER":
                    # For rivers, we must verify we land on a platform at the END of the jump
                    time_of_landing = t + INPUT_LATENCY + DT
                    for e in local_ents:
                        if e['type'] in ['LOG', 'LILYPAD'] and abs(e['z'] - nz) < 0.5:
                            ext_x_end = e['x'] + (e['vx'] * time_of_landing)

                            # Platform landing grace
                            if abs(ext_x_end - nx) < (e['w'] / 2.0 - 0.1):
                                on_platform = True
                                drifted_x += e['vx'] * DT  # Drift while mid-air
                                break
                    if not on_platform:
                        is_safe = False

                if is_safe and BOUND_LEFT <= drifted_x <= BOUND_RIGHT:
                    state_key = (round(drifted_x * 2) / 2.0, round(nz), depth + 1)
                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append((drifted_x, nz, nt, path + [action]))

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

                if bot.pm:
                    terrain_map, entities = bot.poll_world_state(player_x, player_z)
                    terrain_ahead = terrain_map.get(round(player_z + 1), "GRASS")
                    action = bot.calculate_perfect_action(player_x, player_z, action_mask, terrain_map, entities)

                # Record BEFORE stepping (Matches vision state to intended action)
                trajectory.append({
                    'latents': latents,
                    'scalars': scalars,
                    'mask': action_mask,
                    'action': action
                })

                if bot.pm:
                    hazards_ahead = [e for e in entities if abs(e['z'] - (player_z + 1)) <= 0.8]
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
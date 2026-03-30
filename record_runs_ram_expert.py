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
        self.known_terrain = {}  # SOTA: Persistent memory prevents "Empty River" traps
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
            return {},[]

        try:
            vault_data = self.pm.read_bytes(self.entity_vault_addr, 16384)
            self.pm.write_bytes(self.entity_vault_addr, b'\x00' * 16384, 16384)

            current_time = time.time()
            entities =[]
            terrain_votes = {}

            for i in range(1024):
                offset = i * 16
                ptr, x, y, z = struct.unpack_from("<Ifff", vault_data, offset)

                if ptr != 0:
                    distance_to_player = ((x - player_x) ** 2 + (z - player_z) ** 2) ** 0.5
                    if distance_to_player < 0.5:
                        continue

                    speed = 0.0
                    delta_x = 0.0
                    if ptr in self.entity_history:
                        last_x, last_time = self.entity_history[ptr]
                        dt = current_time - last_time
                        if dt > 0.001:
                            raw_speed = (x - last_x) / dt
                            # SOTA: Increased to 50.0 so high-speed Trains aren't ignored
                            if abs(raw_speed) < 50.0:
                                speed = raw_speed
                                delta_x = abs(x - last_x)

                    self.entity_history[ptr] = (x, current_time)
                    is_moving = delta_x > 0.01

                    ent_type = "UNKNOWN"
                    if is_moving:
                        if y < 0.5:
                            ent_type = "LOG"
                        else:
                            ent_type = "CAR"
                    else:
                        if y < 0.3:
                            ent_type = "LILYPAD"
                        else:
                            if abs(y - 1.19) < 0.1:
                                continue
                            ent_type = "OBSTACLE"

                    rz = round(z)
                    if rz not in terrain_votes:
                        terrain_votes[rz] = {'ROAD': 0, 'GRASS': 0, 'RIVER': 0, 'LILYPADS': 0, 'OBSTACLES': 0}

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

                    width = 3.5 if abs(speed) > 8.0 else 1.8
                    if ent_type == "LILYPAD": width = 1.2
                    if ent_type == "OBSTACLE": width = 0.8

                    entities.append({
                        'x': x, 'y': y, 'z': z, 'type': ent_type, 'speed': speed, 'width': width
                    })

            stale_keys =[k for k, v in self.entity_history.items() if current_time - v[1] > 0.5]
            for k in stale_keys: del self.entity_history[k]

            terrain_map = {}
            for tz in range(int(player_z) - 5, int(player_z) + 30):
                votes = terrain_votes.get(tz, {'ROAD': 0, 'GRASS': 0, 'RIVER': 0, 'LILYPADS': 0, 'OBSTACLES': 0})

                if votes['LILYPADS'] > 0 and votes['OBSTACLES'] > 0:
                    votes['RIVER'] -= votes['LILYPADS']

                current_vote = "EMPTY"
                if votes['RIVER'] > 0: current_vote = "RIVER"
                elif votes['ROAD'] > 0: current_vote = "ROAD"
                elif votes['GRASS'] > 0: current_vote = "GRASS"

                # SOTA: Persistent Mapping. Once a river, always a river.
                if tz not in self.known_terrain:
                    self.known_terrain[tz] = current_vote
                else:
                    if current_vote in ["RIVER", "ROAD"]:
                        self.known_terrain[tz] = current_vote
                    elif current_vote == "GRASS" and self.known_terrain[tz] == "EMPTY":
                        self.known_terrain[tz] = "GRASS"

                terrain_map[tz] = self.known_terrain[tz]

            return terrain_map, entities

        except Exception:
            return {},[]

    def calculate_perfect_action(self, player_x, player_z, action_mask, terrain_map, entities):
        """
        SOTA Algorithmic Pathfinding via Spatio-Temporal Lookahead.
        Simulates parallel timelines using BFS to find the path that maximizes forward
        progress while guaranteeing survival against dynamic hitboxes.
        """
        import collections

        # --- TUNABLE PARAMETERS ---
        MAX_DEPTH = 6  # How many moves to look ahead (Depth 6 = ~1.5 seconds)
        DT = 0.25  # Time duration per action/jump in seconds
        PLAYER_WIDTH = 0.6  # Forgiving player hitbox
        BOUND_LEFT = -4.5  # Leftmost screen boundary
        BOUND_RIGHT = 4.5  # Rightmost screen boundary

        # Action mappings
        ACTIONS = [ACTION_UP, ACTION_LEFT, ACTION_RIGHT, ACTION_IDLE]

        # State queue: (current_x, current_z, elapsed_time, action_history)
        queue = collections.deque([(player_x, player_z, 0.0, [])])
        visited = set()

        # Track best outcomes
        best_score = -float('inf')
        best_first_action = ACTION_IDLE
        longest_survival_depth = -1

        while queue:
            x, z, t, path = queue.popleft()
            depth = len(path)

            # --- SCORE EVALUATION ---
            # Prioritize moving forward (Z), penalize deviating far from center (X)
            score = (z - player_z) * 10 - abs(x) * 0.5

            # Update the global best action if this path survives longer or reaches a higher score
            if depth > longest_survival_depth:
                longest_survival_depth = depth
                best_score = score
                if path: best_first_action = path[0]
            elif depth == longest_survival_depth:
                if score > best_score:
                    best_score = score
                    if path: best_first_action = path[0]

            if depth == MAX_DEPTH:
                continue  # Reached lookahead horizon

            # --- EXPLORE BRANCHES ---
            for action in ACTIONS:
                # 1. Obey hard logic constraints for immediate next moves (e.g., blocked by trees)
                if depth == 0 and action_mask[action] < -1.0:
                    continue

                # 2. Calculate intended positional shift
                nx, nz = x, z
                if action == ACTION_UP:
                    nz += 1
                elif action == ACTION_LEFT:
                    nx -= 1
                elif action == ACTION_RIGHT:
                    nx += 1

                # Prevent walking off the map edge
                if nx < BOUND_LEFT or nx > BOUND_RIGHT:
                    continue

                nt = t + DT
                tz = round(nz)
                terrain = terrain_map.get(tz, "GRASS")  # Assume Grass if unknown

                # Extract entities that exist in the target lane
                row_ents = [e for e in entities if abs(round(e['z']) - tz) < 0.5]

                is_safe = True
                drifted_x = nx
                on_platform = False

                # --- 3. SPATIO-TEMPORAL COLLISION DETECTION ---
                if terrain == "ROAD":
                    # Sub-stepping prevents fast entities (like Trains) from teleporting through the player
                    steps = 3
                    for i in range(1, steps + 1):
                        sub_t = t + DT * (i / steps)
                        for e in row_ents:
                            if e['type'] == 'CAR':
                                ext_x = e['x'] + (e['speed'] * sub_t)
                                if abs(ext_x - nx) < (e['width'] / 2.0 + PLAYER_WIDTH / 2.0):
                                    is_safe = False
                                    break
                        if not is_safe: break

                elif terrain == "RIVER":
                    # Must land on a log/lilypad at time `nt`
                    for e in row_ents:
                        if e['type'] in ['LOG', 'LILYPAD']:
                            ext_x = e['x'] + (e['speed'] * nt)
                            if abs(ext_x - nx) < (e['width'] / 2.0):
                                on_platform = True
                                # Calculate river drift for future steps
                                drifted_x += e['speed'] * DT
                                break

                    # If we jumped into the river and missed a platform, or drifted off-screen
                    if not on_platform or drifted_x < BOUND_LEFT or drifted_x > BOUND_RIGHT:
                        is_safe = False

                else:  # GRASS / OBSTACLES
                    for e in row_ents:
                        if e['type'] == 'OBSTACLE':
                            # Static hitboxes
                            if abs(e['x'] - nx) < (e['width'] / 2.0 + PLAYER_WIDTH / 2.0):
                                is_safe = False
                                break

                # --- 4. QUEUE VALID FUTURES ---
                if is_safe:
                    # Discretize X slightly to prevent state-space explosion due to floating point drift
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
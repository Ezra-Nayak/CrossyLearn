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

        # =========================================================================
        # [PLACEHOLDER 1]: THE TERRAIN MANAGER (ROW GENERATOR)
        # Crossy road generates terrain in "Rows" (Z-axis).
        # You need to find the array/list where the game stores the type of terrain
        # for each Z-coordinate (e.g., Z=10 is Grass, Z=11 is Road, Z=12 is River).
        #
        # HOW TO FIND THIS IN CHEAT ENGINE:
        # 1. Stand on grass. Search for an unknown initial value.
        # 2. Hop forward onto a road. Search for changed value.
        # 3. Hop onto a river log. Search for changed value.
        # 4. Once you find the terrain enum for the player's current row, look at
        #    the memory region. It will be an array mapping Z-index to Terrain ID.
        # =========================================================================
        self.TERRAIN_MANAGER_BASE = 0x00000000

        # =========================================================================
        # [PLACEHOLDER 2]: THE ENTITY LIST (HAZARDS) -> SOLVED VIA AOB HASH MAP
        # =========================================================================
        self.entity_vault_addr = None
        self.entity_history = {}
        if self.pm:
            self.inject_entity_tracker()

    def inject_entity_tracker(self):
        """Injects an Assembly Hash Map into the 3D Engine Transform Loop."""
        try:
            # AOB for: movss xmm5,[esi+08] followed by movss xmm0,[ecx+00000288]
            aob_sig = b"\xF3\x0F\x10\x6E\x08\xF3\x0F\x10\x81"
            module = pymem.process.module_from_name(self.pm.process_handle, "UnityPlayer.dll")
            if not module:
                print("[RAM ERROR] Could not find UnityPlayer.dll")
                return

            module_data = self.pm.read_bytes(module.lpBaseOfDll, module.SizeOfImage)
            sig_offset = module_data.find(aob_sig)

            if sig_offset == -1:
                print("[RAM ERROR] Could not find 3D Engine AOB signature. (Already injected?)")
                return

            target_addr = module.lpBaseOfDll + sig_offset

            # Allocate 18KB: 16384 for vault (1024 slots), 1024 for payload code
            alloc_addr = self.pm.allocate(17408)
            self.entity_vault_addr = alloc_addr
            code_addr = alloc_addr + 16384

            # Assembly Hash Map Payload
            # We are hooking `movss xmm5,[esi+08]`.
            # esi points directly to the active 3D Vector3 object (X, Y, Z).
            payload = bytearray()
            payload.extend(b"\x50\x52")  # push eax, push edx
            payload.extend(b"\x89\xF2")  # mov edx, esi
            payload.extend(b"\xC1\xEA\x03")  # shr edx, 3
            payload.extend(b"\x81\xE2\xFF\x03\x00\x00")  # and edx, 0x3FF (1024 slots)
            payload.extend(b"\xC1\xE2\x04")  # shl edx, 4 (16 bytes/slot)
            payload.extend(b"\x81\xC2" + struct.pack("<I", self.entity_vault_addr))  # add edx, vault

            payload.extend(b"\x89\x32")  # mov [edx], esi (Ptr)
            payload.extend(b"\x8B\x06\x89\x42\x04")  # mov eax, [esi]; mov[edx+4], eax (X)
            payload.extend(b"\x8B\x46\x04\x89\x42\x08")  # mov eax,[esi+4]; mov [edx+8], eax (Y)
            payload.extend(b"\x8B\x46\x08\x89\x42\x0C")  # mov eax, [esi+8]; mov [edx+12], eax (Z)

            payload.extend(b"\x5A\x58")  # pop edx, pop eax

            # Original Instruction
            payload.extend(b"\xF3\x0F\x10\x6E\x08")  # movss xmm5,[esi+08]

            # Return Jump (Instruction was exactly 5 bytes)
            return_addr = target_addr + 5
            jmp_rel = return_addr - (code_addr + len(payload) + 5)
            payload.extend(b"\xE9" + struct.pack("<i", jmp_rel))

            self.pm.write_bytes(code_addr, bytes(payload), len(payload))

            # Write Hook (jmp to our code)
            hook = bytearray()
            hook.extend(b"\xE9" + struct.pack("<i", code_addr - (target_addr + 5)))

            import ctypes
            kernel32 = ctypes.windll.kernel32
            old_protect = ctypes.c_ulong()
            # Unprotect, Overwrite exactly 5 bytes, Reprotect
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
            # Instant Garbage Collection for ghost busting
            self.pm.write_bytes(self.entity_vault_addr, b'\x00' * 16384, 16384)

            current_time = time.time()
            entities = []
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
                        delta_x = abs(x - last_x)
                        if dt > 0.001: speed = (x - last_x) / dt

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
                            # EXCLUSION: The Railway Pole is height ~1.19.
                            # We ignore it entirely so BFS doesn't think the path is blocked.
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

                    # Speed-based width for cars/trucks, fixed width for static objects
                    width = 3.5 if abs(speed) > 8.0 else 1.8
                    if ent_type == "LILYPAD": width = 1.2
                    if ent_type == "OBSTACLE": width = 0.8  # Tighter collision for trees

                    entities.append({
                        'x': x, 'y': y, 'z': z, 'type': ent_type, 'speed': speed, 'width': width
                    })

            stale_keys = [k for k, v in self.entity_history.items() if current_time - v[1] > 0.5]
            for k in stale_keys: del self.entity_history[k]

            terrain_map = {}
            for tz in range(int(player_z) - 5, int(player_z) + 30):
                votes = terrain_votes.get(tz, {'ROAD': 0, 'GRASS': 0, 'RIVER': 0, 'LILYPADS': 0, 'OBSTACLES': 0})

                # Ghost lilypad cancellation
                if votes['LILYPADS'] > 0 and votes['OBSTACLES'] > 0:
                    votes['RIVER'] -= votes['LILYPADS']

                if votes['RIVER'] > 0:
                    terrain_map[tz] = "RIVER"
                elif votes['ROAD'] > 0:
                    terrain_map[tz] = "ROAD"
                elif votes['GRASS'] > 0:
                    terrain_map[tz] = "GRASS"
                else:
                    terrain_map[tz] = "EMPTY"

            return terrain_map, entities

        except Exception:
            return {}, []

    def calculate_perfect_action(self, player_x, player_z, action_mask, terrain_map, entities):
        """
        BFS Pathfinding Brain.
        Simulates future states to find the optimal chess-like sequence of moves.
        """
        def simulate_step(cx, cz, ct, action):
            dt = 0.25  # Time per action
            new_ct = ct + dt
            new_cz = cz
            new_cx = cx

            if action == "UP": new_cz += 1
            elif action == "LEFT": new_cx -= 1
            elif action == "RIGHT": new_cx += 1

            spot_terrain = terrain_map.get(round(new_cz), "GRASS")
            spot_entities = [e for e in entities if abs(e['z'] - new_cz) <= 0.8]

            # 1. Static obstacles (Collision)
            for e in entities:
                if e['type'] == "OBSTACLE":
                    if abs(e['x'] - new_cx) < 0.7 and abs(e['z'] - new_cz) < 0.5:
                        return False, cx, cz, ct

            # 2. Roads (Vehicle collision prediction)
            if spot_terrain == "ROAD":
                for hazard in [e for e in spot_entities if e['type'] == "CAR"]:
                    hw = hazard.get('width', 1.8) / 2.0
                    for step in [0.0, 0.1, 0.2]:
                        sim_t = ct + step
                        future_x = hazard['x'] + (hazard['speed'] * sim_t)
                        if abs(new_cx - future_x) < (hw + 0.5):
                            return False, cx, cz, ct
                return True, new_cx, new_cz, new_ct

            # 4. Rivers (Platform presence and drifting)
            if spot_terrain == "RIVER":
                platforms = [e for e in spot_entities if e['type'] in ["LOG", "LILYPAD"]]
                safe = False
                best_plat = None
                for p in platforms:
                    # Where will the log be when we arrive?
                    plat_future_x = p['x'] + (p['speed'] * new_ct)
                    if abs(new_cx - plat_future_x) < (p.get('width', 2.0) / 2.0 - 0.2):
                        safe = True
                        best_plat = p
                        break
                if not safe: return False, cx, cz, ct

                # Apply log drift to our simulation coordinate
                if best_plat: new_cx += best_plat['speed'] * dt
                return True, new_cx, new_cz, new_ct

            return True, new_cx, new_cz, new_ct

        # --- BFS Loop ---
        queue = [(player_x, player_z, 0.0, [])]
        best_score = -9999
        best_path = []
        visited = set()
        max_depth = 5

        while queue:
            cx, cz, ct, actions = queue.pop(0)

            # Heuristic:
            # 1. Progress is the primary goal (10 points per Z)
            # 2. Staying centered is secondary (-0.5 per X)
            # 3. Efficiency penalty (-0.1 per move) to prevent loitering
            score = (cz - player_z) * 10.0 - abs(cx) * 0.5 - len(actions) * 0.1

            # If the current tile is a Railway (EMPTY/ROAD with no cars),
            # we increase the score slightly to encourage clearing it.
            if terrain_map.get(round(cz)) == "EMPTY":
                score += 2.0

            if score > best_score:
                best_score = score
                best_path = actions

            if len(actions) >= max_depth: continue

            state_key = (round(cx), round(cz), round(ct / 0.25))
            if state_key in visited: continue
            visited.add(state_key)

            for move in ["UP", "LEFT", "RIGHT", "IDLE"]:
                # Check if action is physically allowed by the environment mask
                move_idx = {"UP": ACTION_UP, "LEFT": ACTION_LEFT, "RIGHT": ACTION_RIGHT, "IDLE": ACTION_IDLE}[move]
                if action_mask[move_idx] < -1: continue

                safe, nx, nz, nt = simulate_step(cx, cz, ct, move)
                if safe:
                    queue.append((nx, nz, nt, actions + [move]))

        # Final conversion back to Action Enum
        if best_path:
            return {"UP": ACTION_UP, "LEFT": ACTION_LEFT, "RIGHT": ACTION_RIGHT, "IDLE": ACTION_IDLE}[best_path[0]]

        return ACTION_IDLE


def main():
    os.makedirs("expert_data", exist_ok=True)
    print("--- AUTOMATED RAM-EXPERT RECORDER ---")

    env = CrossyGameEnv(ui=None)
    bot = AlgorithmicExpert()

    if not bot.pm:
        print("[WARNING] RAM hook failed. Script will run, but bot will always IDLE.")

    print("\n[READY] Focus Crossy Road. Bot will play automatically.")
    print("Press CTRL+C in this terminal to stop recording.\n")

    trajectory = []
    recording = False
    episodes_recorded = 0

    try:
        while True:
            # 1. SENSE: Get Vision (for the Neural Net) and RAM state (for the Bot)
            latents, scalars, score, is_alive, action_mask = env.get_state()

            if latents is None:
                time.sleep(0.1)
                continue

            # 2. THINK: Bot reads raw memory to calculate the perfect move
            player_x = env.last_known_coords[0]
            player_z = env.last_known_coords[1]

            action = ACTION_IDLE
            terrain_ahead = "UNKNOWN"
            hazards_ahead = []
            if bot.pm:
                terrain_map, entities = bot.poll_world_state(player_x, player_z)
                terrain_ahead = terrain_map.get(round(player_z + 1), "GRASS")
                action = bot.calculate_perfect_action(player_x, player_z, action_mask, terrain_map, entities)

            # 3. RECORD & ACT
            if is_alive:
                if not recording:
                    print("\n[REC] Automated run started. Recording...")
                    recording = True
                    trajectory = []
                    env.steps_in_episode = 1  # Unlock the Action Mask

                # Record exactly what the VAE sees mapped to the Bot's perfect decision
                trajectory.append({
                    'latents': latents,
                    'scalars': scalars,
                    'mask': action_mask,
                    'action': action
                })

                # Debug Radar: Print the detected Terrain and Hazards
                total_tracked = len(bot.entity_history)
                if bot.pm:
                    hazards_ahead = [e for e in entities if abs(e['z'] - (player_z + 1)) <= 0.8]

                action_str = ['UP  ', 'LEFT', 'RGHT', 'IDLE'][action]
                print(
                    f"\r[RADAR] Ter: {terrain_ahead[:5]:<5} | Haz: {len(hazards_ahead):02d} | Z: {player_z:.0f} | Act: {action_str}    ",
                    end="")

                # Actually execute the bot's decision in the game using pydirectinput
                # (Handled seamlessly inside env.step)
                env.step(action)

            else:
                if recording:
                    recording = False

                    # Snip end-of-life frames to keep data clean, just like we discussed
                    if len(trajectory) > 60:
                        episodes_recorded += 1
                        filename = f"expert_data/ram_bot_run_{int(time.time())}.pt"
                        torch.save(trajectory, filename)
                        print(
                            f"\n[SAVE] Bot died. Saved {len(trajectory)} perfect frames to {filename} (Total Runs: {episodes_recorded})")
                    else:
                        print("\n[SKIP] Run too short, discarded.")

                    trajectory = []
                    env.steps_in_episode = 0

                    # Force env to reset the game and click "Retry" automatically
                    env.reset()

    except KeyboardInterrupt:
        print(f"\n\n[SHUTDOWN] Exiting. Successfully recorded {episodes_recorded} perfect runs.")


if __name__ == "__main__":
    main()
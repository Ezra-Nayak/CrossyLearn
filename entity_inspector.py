import cv2
import numpy as np
import time
import struct
import pymem
import math
from tracker import RamTracker

class EntityVaultInspector:
    def __init__(self):
        try:
            self.pm = pymem.Pymem("Crossy Road.exe")
            print("[INSPECTOR] Attached to Crossy Road memory.")
        except Exception as e:
            print(f"[INSPECTOR ERROR] Could not attach: {e}")
            self.pm = None

        self.entity_vault_addr = None
        if self.pm:
            self.inject()

    def inject(self):
        try:
            aob_sig = b"\xF3\x0F\x10\x6E\x08\xF3\x0F\x10\x81"
            module = pymem.process.module_from_name(self.pm.process_handle, "UnityPlayer.dll")
            if not module:
                print("Could not find UnityPlayer.dll")
                return

            module_data = self.pm.read_bytes(module.lpBaseOfDll, module.SizeOfImage)
            sig_offset = module_data.find(aob_sig)

            if sig_offset == -1:
                print("AOB not found. Already injected? Restart game if vault address is lost.")
                return

            target_addr = module.lpBaseOfDll + sig_offset
            alloc_addr = self.pm.allocate(17408)
            self.entity_vault_addr = alloc_addr
            code_addr = alloc_addr + 16384

            payload = bytearray()
            payload.extend(b"\x50\x52\x89\xF2\xC1\xEA\x03\x81\xE2\xFF\x03\x00\x00\xC1\xE2\x04")
            payload.extend(b"\x81\xC2" + struct.pack("<I", self.entity_vault_addr))
            payload.extend(b"\x89\x32\x8B\x06\x89\x42\x04\x8B\x46\x04\x89\x42\x08\x8B\x46\x08\x89\x42\x0C\x5A\x58")
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

            print(f"[INSPECTOR] Vault injected successfully at {hex(self.entity_vault_addr).upper()}")

        except Exception as e:
            print(f"[INSPECTOR ERROR] Injection Failed: {e}")

    def read_vault(self):
        if not self.pm or not self.entity_vault_addr: return[]
        try:
            data = self.pm.read_bytes(self.entity_vault_addr, 16384)
            self.pm.write_bytes(self.entity_vault_addr, b'\x00' * 16384, 16384)
            entities =[]
            for i in range(1024):
                offset = i * 16
                ptr, x, y, z = struct.unpack_from("<Ifff", data, offset)
                if ptr != 0:
                    entities.append({'ptr': ptr, 'x': x, 'y': y, 'z': z})
            return entities
        except Exception:
            return[]

def main():

    inspector = EntityVaultInspector()
    ram_tracker = RamTracker()

    cv2.namedWindow("Entity Radar")

    click_data = {'clicked': False, 'px': 0, 'py': 0}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param['clicked'] = True
            param['px'] = x
            param['py'] = y

    cv2.setMouseCallback("Entity Radar", mouse_callback, param=click_data)

    history = {}
    SCALE = 40  # pixels per game unit (Grid lines correspond to 1 unit perfectly)
    WIDTH, HEIGHT = 800, 800

    while True:
        coords = ram_tracker.get_coords()
        if coords:
            player_x, _, player_z = coords
        else:
            player_x, player_z = 0.0, 0.0

        raw_entities = inspector.read_vault()
        current_time = time.time()

        for e in raw_entities:
            ptr = e['ptr']
            if ptr in history:
                lx, ly, lz, lt, lvx, lvz = history[ptr]
                dt = current_time - lt
                if dt > 0.001:
                    vx = (e['x'] - lx) / dt
                    vz = (e['z'] - lz) / dt
                else:
                    vx, vz = lvx, lvz
            else:
                vx, vz = 0.0, 0.0

            history[ptr] = (e['x'], e['y'], e['z'], current_time, vx, vz)

        active_entities = []
        stale_ptrs = []
        for ptr, (hx, hy, hz, ht, hvx, hvz) in history.items():
            if current_time - ht > 0.1:  # Keep entities alive for 100ms to prevent flickering
                stale_ptrs.append(ptr)
            else:
                active_entities.append({
                    'ptr': ptr, 'x': hx, 'y': hy, 'z': hz, 'vx': hvx, 'vz': hvz
                })

        for ptr in stale_ptrs:
            del history[ptr]

        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)  # Sleek dark gray background

        # Draw Precision Grid (1 Box = 1 Game Coordinate Tile)
        for i in range(-15, 16):
            gx = int(i * SCALE + WIDTH / 2 - (player_x % 1.0) * SCALE)
            color = (90, 90, 90) if i == 0 else (45, 45, 45)
            thickness = 2 if i == 0 else 1
            cv2.line(canvas, (gx, 0), (gx, HEIGHT), color, thickness, cv2.LINE_AA)

        z_offset = player_z % 1.0
        for i in range(-5, 25):
            gz = int(HEIGHT - (i - z_offset) * SCALE - HEIGHT / 4)
            if 0 <= gz <= HEIGHT:
                color = (90, 90, 90) if i == 0 else (45, 45, 45)
                thickness = 2 if i == 0 else 1
                cv2.line(canvas, (0, gz), (WIDTH, gz), color, thickness, cv2.LINE_AA)

        # Draw Entities
        for e in active_entities:
            px = int((e['x'] - player_x) * SCALE + WIDTH / 2)
            pz = int(HEIGHT - (e['z'] - player_z) * SCALE - HEIGHT / 4)

            if px < -1000 or px > WIDTH + 1000 or pz < -1000 or pz > HEIGHT + 1000:
                continue

            # Pointer Click Detection Logic
            if click_data['clicked']:
                dist = math.hypot(px - click_data['px'], pz - click_data['py'])
                if dist < 12:
                    print(f"\n{'=' * 50}")
                    print(f"[TARGET ACQUIRED] Pointer: {hex(e['ptr']).upper()}")
                    print(f" -> X: {e['x']:.3f} | Y: {e['y']:.3f} | Z: {e['z']:.3f}")
                    print(f" -> Velocity: X:{e['vx']:.2f} | Z:{e['vz']:.2f}")

                    # DUMP SURROUNDING MEMORY TO FIND THE ID!
                    try:
                        start_dump = e['ptr'] - 0x20
                        dump_bytes = inspector.pm.read_bytes(start_dump, 128)

                        print("\n[MEMORY DUMP AROUND POINTER]")
                        print(f"{'Offset':<8} | {'Hex':<10} | {'Integer':<10} | {'Float'}")
                        print("-" * 50)

                        for offset_idx in range(0, 128, 4):
                            actual_offset = hex(offset_idx - 0x20).upper().replace('X', 'x')
                            chunk = dump_bytes[offset_idx:offset_idx + 4]

                            val_int = struct.unpack("<I", chunk)[0]
                            val_float = struct.unpack("<f", chunk)[0]
                            val_hex = chunk[::-1].hex().upper()  # reverse for endianness visualization

                            # Highlight exact coordinates
                            marker = ""
                            if val_float == e['x']:
                                marker = " <-- [X COORD]"
                            elif val_float == e['y']:
                                marker = " <-- [Y COORD]"
                            elif val_float == e['z']:
                                marker = " <-- [Z COORD]"

                            print(f"{actual_offset:<8} | 0x{val_hex:<8} | {val_int:<10} | {val_float:10.3f}{marker}")

                    except Exception as err:
                        print(f"Could not dump memory: {err}")

                    print(f"{'=' * 50}\n")
                    click_data['clicked'] = False

            speed = math.hypot(e['vx'], e['vz'])
            if speed > 0.5:
                # Draw dynamic arrowhead for moving objects
                angle = math.atan2(-e['vz'], e['vx'])  # Negative Z because screen Y is inverted
                length = 14

                # Calculate triangle points
                pt1 = (int(px + math.cos(angle) * length), int(pz + math.sin(angle) * length))
                pt2 = (int(px + math.cos(angle + 2.5) * 8), int(pz + math.sin(angle + 2.5) * 8))
                pt3 = (int(px + math.cos(angle - 2.5) * 8), int(pz + math.sin(angle - 2.5) * 8))
                pts = np.array([pt1, pt2, pt3], np.int32)

                # Velocity vector line (drawn first so it goes under the shape)
                end_px = int(px + e['vx'] * SCALE * 0.3)
                end_pz = int(pz - e['vz'] * SCALE * 0.3)
                cv2.line(canvas, (px, pz), (end_px, end_pz), (0, 140, 255), 2, cv2.LINE_AA)

                cv2.fillPoly(canvas, [pts], (70, 70, 255), cv2.LINE_AA)
                cv2.polylines(canvas, [pts], True, (120, 120, 255), 1, cv2.LINE_AA)
            else:
                # Sleek diamond for static objects
                pts = np.array([[px, pz - 6], [px + 6, pz], [px, pz + 6], [px - 6, pz]], np.int32)
                cv2.fillPoly(canvas, [pts], (50, 200, 50), cv2.LINE_AA)
                cv2.polylines(canvas, [pts], True, (100, 255, 100), 1, cv2.LINE_AA)

            # Floating text
            cv2.putText(canvas, f"{e['y']:.1f}", (px + 10, pz + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 190, 200), 1,
                        cv2.LINE_AA)

        click_data['clicked'] = False

        # --- Player Crosshair (Cyan Tech Style) ---
        cx, cy = WIDTH // 2, HEIGHT - HEIGHT // 4

        # Outer pulsing ring
        pulse_radius = int(18 + (math.sin(current_time * 6) * 4))
        cv2.circle(canvas, (cx, cy), pulse_radius, (255, 255, 0), 1, cv2.LINE_AA)

        # Inner core and reticle
        cv2.circle(canvas, (cx, cy), 14, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 2, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.line(canvas, (cx - 22, cy), (cx - 8, cy), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx + 8, cy), (cx + 22, cy), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - 22), (cx, cy - 8), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy + 8), (cx, cy + 22), (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Entity Radar", canvas)
        if cv2.waitKey(16) & 0xFF == ord('q'):  # ~60 FPS cap to save CPU and reduce tearing
            break

if __name__ == "__main__":
    main()
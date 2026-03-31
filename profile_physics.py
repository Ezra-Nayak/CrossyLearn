import cv2
import numpy as np
import time
import struct
import pymem
from collections import deque


class PhysicsVerifier:
    def __init__(self):
        try:
            self.pm = pymem.Pymem("Crossy Road.exe")
            print("[VERIFIER] Attached to Memory.")
        except Exception as e:
            print(f"[VERIFIER ERROR] Could not attach: {e}")
            self.pm = None

        self.entity_vault_addr = None
        if self.pm:
            self.inject()

    def inject(self):
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
            payload.extend(b"\x89\x32\x8B\x06\x89\x42\x04\x8B\x46\x04\x89\x42\x08\x8B\x46\x08\x89\x42\x0C\x5A\x58")
            payload.extend(b"\xF3\x0F\x10\x6E\x08")

            jmp_rel = (target_addr + 5) - (code_addr + len(payload) + 5)
            payload.extend(b"\xE9" + struct.pack("<i", jmp_rel))

            self.pm.write_bytes(code_addr, bytes(payload), len(payload))
            hook = bytearray(b"\xE9" + struct.pack("<i", code_addr - (target_addr + 5)))

            import ctypes
            kernel32 = ctypes.windll.kernel32
            old_protect = ctypes.c_ulong()
            kernel32.VirtualProtectEx(self.pm.process_handle, target_addr, 5, 0x40, ctypes.byref(old_protect))
            self.pm.write_bytes(target_addr, bytes(hook), 5)
            kernel32.VirtualProtectEx(self.pm.process_handle, target_addr, 5, old_protect, ctypes.byref(old_protect))
            print(f"[VERIFIER] Vault Injected: {hex(self.entity_vault_addr).upper()}")
        except Exception as e:
            pass

    def read_raw_entities(self):
        if not self.pm or not self.entity_vault_addr: return []
        try:
            data = self.pm.read_bytes(self.entity_vault_addr, 16384)
            self.pm.write_bytes(self.entity_vault_addr, b'\x00' * 16384, 16384)
            entities = []
            for i in range(1024):
                offset = i * 16
                ptr, x, y, z = struct.unpack_from("<Ifff", data, offset)
                if ptr != 0:
                    try:
                        # Reading Exact Bounding Boxes verified from your dump
                        bounds_data = self.pm.read_bytes(ptr + 0x0C, 12)
                        w, h, d = struct.unpack("<fff", bounds_data)
                        entities.append({'ptr': ptr, 'x': x, 'y': y, 'z': z, 'w': w, 'h': h, 'd': d})
                    except:
                        pass
            return entities
        except:
            return []


def main():
    verifier = PhysicsVerifier()
    cv2.namedWindow("Ground Truth Verifier")

    # Track history using a deque to stabilize velocity over a 10-frame time window
    history = {}

    SCALE = 50
    WIDTH, HEIGHT = 1000, 800

    while True:
        raw_entities = verifier.read_raw_entities()
        current_time = time.time()
        active_entities = []

        for e in raw_entities:
            ptr = e['ptr']
            if ptr not in history:
                history[ptr] = deque(maxlen=10)

            history[ptr].append({'x': e['x'], 'z': e['z'], 't': current_time})

            vx, vz = 0.0, 0.0
            # Calculate stable velocity using oldest and newest points in the window
            if len(history[ptr]) > 3:
                oldest = history[ptr][0]
                dt = current_time - oldest['t']
                if dt > 0.05:  # Only calculate if at least 50ms has passed
                    vx = (e['x'] - oldest['x']) / dt
                    vz = (e['z'] - oldest['z']) / dt

            # Ignore massive velocity spikes (Object wrap-around / pooling)
            if abs(vx) > 30.0: vx = 0.0

            active_entities.append({
                'ptr': ptr, 'x': e['x'], 'y': e['y'], 'z': e['z'],
                'w': e['w'], 'h': e['h'], 'd': e['d'], 'vx': vx, 'vz': vz
            })

        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Assuming player at 0, 0 for pure engine visualization
        player_x, player_z = 0.0, 0.0

        for i in range(-15, 16):
            gx = int(i * SCALE + WIDTH / 2)
            cv2.line(canvas, (gx, 0), (gx, HEIGHT), (30, 30, 30), 1)
        for i in range(-5, 25):
            gz = int(HEIGHT - i * SCALE - HEIGHT / 4)
            cv2.line(canvas, (0, gz), (WIDTH, gz), (30, 30, 30), 1)

        for e in active_entities:
            px = int((e['x'] - player_x) * SCALE + WIDTH / 2)
            pz = int(HEIGHT - (e['z'] - player_z) * SCALE - HEIGHT / 4)

            # Exact Bounding Box Math
            box_w = int(e['w'] * SCALE)
            box_d = int(e['d'] * SCALE)

            tl = (px - box_w // 2, pz - box_d // 2)
            br = (px + box_w // 2, pz + box_d // 2)

            speed = abs(e['vx']) + abs(e['vz'])
            is_moving = speed > 0.5

            color = (0, 0, 255) if is_moving else (0, 255, 0)

            # Draw Exact Bounding Box
            cv2.rectangle(canvas, tl, br, color, 1)
            cv2.circle(canvas, (px, pz), 3, (255, 255, 255), -1)

            # Draw Stabilized Velocity Vector
            if is_moving:
                end_px = int(px + e['vx'] * SCALE * 0.5)
                end_pz = int(pz - e['vz'] * SCALE * 0.5)
                cv2.line(canvas, (px, pz), (end_px, end_pz), (0, 255, 255), 2)

                # Render stable velocity text
                cv2.putText(canvas, f"V:{e['vx']:.1f}", (px - 15, pz - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.imshow("Ground Truth Verifier", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
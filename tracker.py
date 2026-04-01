# --- tracker.py ---
import cv2
import numpy as np
import time
import math
import struct
import ctypes
import pymem
import pymem.pattern


class ChickenTracker:
    def __init__(self, vision_params):
        self.params = vision_params

        # COASTING STATE
        self.is_locked = False
        self.last_valid_pos = None
        self.last_valid_box = None

        self.frames_since_valid_lock = 0
        # Tuned to 10 based on your logs (covers sideways hops + brief occlusions)
        self.MAX_COAST_FRAMES = 10

        self.MIN_SOLID_AREA = 20.0

    def track(self, frame):
        h, w, _ = frame.shape

        # 1. Masking
        angle_rad = math.radians(self.params['LINE_ANGLE_DEG'])
        slope = math.tan(angle_rad)
        search_y1 = self.params['SEARCH_ZONE_Y_INTERCEPT']
        search_y2 = int(slope * w + search_y1)

        search_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array([[0, search_y1], [w, search_y2], [w, h], [0, h]], dtype=np.int32)
        cv2.fillPoly(search_mask, [pts], 255)

        masked = cv2.bitwise_and(frame, frame, mask=search_mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.params['LOWER_BOUND'], self.params['UPPER_BOUND'])

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) > self.MIN_SOLID_AREA]

        if valid:
            # --- LOCKED ---
            c = max(valid, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            self.is_locked = True
            self.frames_since_valid_lock = 0
            self.last_valid_pos = (x + w // 2, y + h)
            self.last_valid_box = (x, y, w, h)

            return self.last_valid_pos, self.last_valid_box, "LOCKED"

        else:
            # --- COASTING OR LOST ---
            if self.is_locked:
                self.frames_since_valid_lock += 1

                if self.frames_since_valid_lock <= self.MAX_COAST_FRAMES:
                    # Freeze at last known spot (The Anti-Flicker)
                    return self.last_valid_pos, self.last_valid_box, "COASTING"
                else:
                    # It's been too long. The chicken is gone (Eagle? Drowned? Glitch?)
                    self.is_locked = False
                    return None, None, "LOST"
            else:
                return None, None, "SEARCHING"


class RamTracker:
    def __init__(self):
        self.pm = None
        self.pointer_location = None

        # UnityPlayer Signatures
        self.coord_sig = b"\xF3\x0F\x5C\x18\xF3\x0F\x5C\x60\x04\xF3\x0F\x5C\x50"

        self.vault_signature = b'\x50\x59\x52\x4C\x42\x49\x52\x44'  # PYRLBIRD

    def attach_and_inject(self):
        try:
            self.pm = pymem.Pymem("Crossy Road.exe")
        except Exception:
            return False

        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            old_protect = ctypes.c_ulong()

            # Check if already injected
            vault_addr = pymem.pattern.pattern_scan_all(self.pm.process_handle, self.vault_signature)
            if vault_addr:
                self.pointer_location = vault_addr + 8
                return True

            # Allocate Universal Vault
            alloc_addr = self.pm.allocate(1024)
            vault_addr = alloc_addr
            self.pointer_location = vault_addr + 8

            coord_code_addr = alloc_addr + 128

            # --- 1. COORDINATE TRACKER (UnityPlayer.dll) ---
            mod_unity = pymem.process.module_from_name(self.pm.process_handle, "UnityPlayer.dll")
            if mod_unity:
                mod_data = self.pm.read_bytes(mod_unity.lpBaseOfDll, mod_unity.SizeOfImage)
                sig_offset = mod_data.find(self.coord_sig)
                if sig_offset != -1:
                    inject_addr = mod_unity.lpBaseOfDll + sig_offset

                    payload = bytearray()
                    payload.extend(self.vault_signature)
                    payload.extend(b'\x00\x00\x00\x00')
                    payload.append(0xA3)
                    payload.extend(struct.pack("<I", self.pointer_location))
                    payload.extend(b"\xF3\x0F\x5C\x18\xF3\x0F\x5C\x60\x04")
                    payload.append(0xE9)
                    payload.extend(struct.pack("<i", (inject_addr + 9) - (coord_code_addr + 19)))

                    self.pm.write_bytes(vault_addr, bytes(payload[:12]), 12)
                    self.pm.write_bytes(coord_code_addr, bytes(payload[12:]), len(payload) - 12)

                    hook = bytearray(
                        b"\xE9" + struct.pack("<i", coord_code_addr - (inject_addr + 5)) + b"\x90\x90\x90\x90")
                    kernel32.VirtualProtectEx(self.pm.process_handle, inject_addr, 9, 0x40, ctypes.byref(old_protect))
                    self.pm.write_bytes(inject_addr, bytes(hook), 9)
                    kernel32.VirtualProtectEx(self.pm.process_handle, inject_addr, 9, old_protect,
                                              ctypes.byref(old_protect))

            print(f"[RAM] Coordinate hook injected! Vault located at: {hex(vault_addr)}")
            return True
        except Exception as e:
            self.pm = None
            self.pointer_location = None
            return False

    def get_coords(self):
        if not self.pm or not self.pointer_location:
            if not self.attach_and_inject():
                return None
        try:
            ptr = self.pm.read_int(self.pointer_location)
            if ptr > 0:
                x = self.pm.read_float(ptr)
                y = self.pm.read_float(ptr + 4)
                z = self.pm.read_float(ptr + 8)
                return x, y, z
        except Exception:
            self.pm = None
            self.pointer_location = None
        return None

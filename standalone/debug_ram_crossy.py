import pymem
import pymem.process
import struct
import sys

# Game Configuration
PROCESS_NAME = "Crossy Road.exe"
MODULE_NAME = "UnityPlayer.dll"

# The "Holy Grail" Offsets provided
# Z is at +8, Y is at +4, X is at +0
OFFSET_X = 0x0
OFFSET_Y = 0x4
OFFSET_Z = 0x8

# The Manager -> Player offset (from your instruction [ebx + 0xA8C])
OFFSET_PLAYER_FROM_MANAGER = 0xA8C

# Signature to find the movement code
# 0F 10 46 74 -> movups xmm0, [esi+74] (Example vector move)
# 0F 11 02    -> movups [edx], xmm0
SIGNATURE = b"\x0F\x10\x46\x74\x0F\x11\x02"


def get_pointer_addr(pm, base, offsets):
    """Recursively follows a pointer chain to get the final address."""
    try:
        addr = base
        for offset in offsets:
            if addr == 0: return 0
            addr = pm.read_int(addr) + offset
        return addr
    except:
        return 0


def main():
    print(f"--- Searching for {PROCESS_NAME} ---")
    try:
        pm = pymem.Pymem(PROCESS_NAME)
        print(f"[+] Attached to Process ID: {pm.process_id}")
    except Exception as e:
        print(f"[-] Could not find process. Make sure the game is running and you are Admin.")
        print(f"Error: {e}")
        return

    # 1. Get Module Base Address
    try:
        module = pymem.process.module_from_name(pm.process_handle, MODULE_NAME)
        module_base = module.lpBaseOfDll
        print(f"[+] Found {MODULE_NAME} at: {hex(module_base)}")
    except AttributeError:
        print(f"[-] Could not find {MODULE_NAME}. Is the game fully loaded?")
        return

    # 2. Scan for the Signature (AOB)
    print("\n--- Scanning for Signature ---")
    try:
        # We read the module memory to scan for bytes
        module_data = pm.read_bytes(module_base, module.SizeOfImage)
        sig_index = module_data.find(SIGNATURE)

        if sig_index != -1:
            final_sig_addr = module_base + sig_index
            print(f"[+] Signature found at: {hex(final_sig_addr)}")
            print("    This is the code location handling the movement vector.")
            print("    To get the pointer, you usually hook this address or read the register (EBX/ESI) at this point.")
        else:
            print("[-] Signature not found in current memory.")
    except Exception as e:
        print(f"[-] Scan failed: {e}")

    # 3. Reading the Coordinates (The Static Pointer Logic)
    # Since we don't have the "UnityPlayer.dll + ???" offset from your prompt,
    # we assume a standard Unity structure to demonstrate reading the XYZ.

    print("\n--- Coordinate Reader (Mockup) ---")
    print(f"To read the XYZ, we need the base pointer that leads to [EBX] in your instruction.")
    print(f"Logic: Manager -> Offset {hex(OFFSET_PLAYER_FROM_MANAGER)} -> Vector3")

    # Example: If you find the static base in Cheat Engine (e.g., UnityPlayer.dll + 0x01A2B3C)
    # You would plug it in here:
    # static_ptr_offset = 0x01234567  <-- REPLACE THIS AFTER FINDING IN CHEAT ENGINE

    # Pseudo-code for reading if you had the chain:
    # manager_addr = pm.read_int(module_base + static_ptr_offset)
    # player_addr = pm.read_int(manager_addr + 0x???) # Intermediate offsets often exist in Unity
    # final_vector_addr = player_addr + OFFSET_PLAYER_FROM_MANAGER

    # coord_x = pm.read_float(final_vector_addr + OFFSET_X)
    # coord_y = pm.read_float(final_vector_addr + OFFSET_Y)
    # coord_z = pm.read_float(final_vector_addr + OFFSET_Z)

    print("Script finished successfully.")


if __name__ == "__main__":
    main()
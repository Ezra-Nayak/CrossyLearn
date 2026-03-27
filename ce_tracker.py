import pymem
import pymem.pattern
import time


def attach_and_find_vault():
    """Attaches to the game and finds our custom memory vault."""
    try:
        pm = pymem.Pymem("Crossy Road.exe")
        print("Successfully attached to Crossy Road.exe")
    except pymem.exception.ProcessNotFound:
        print("Error: Could not find Crossy Road.exe. Is the game running?")
        return None, None

    print("Scanning memory for the PYRLBIRD vault...")
    # The hex equivalent of "PYRLBIRD"
    signature = b'\x50\x59\x52\x4C\x42\x49\x52\x44'

    vault_address = pymem.pattern.pattern_scan_all(pm.process_handle, signature)

    if not vault_address:
        print("Error: Vault not found! Make sure the Cheat Engine script is checked/activated.")
        return None, None

    print(f"Vault found successfully at: {hex(vault_address)}")
    # Our pointer is stored exactly 8 bytes after the signature
    return pm, vault_address + 8


def main():
    pm, pointer_location = attach_and_find_vault()

    if not pm or not pointer_location:
        return

    print("Starting RL observation loop (Press Ctrl+C to stop)...")
    print("-" * 60)

    try:
        while True:
            # 1. Read the dynamic 32-bit pointer from our vault
            current_pointer = pm.read_int(pointer_location)

            # If the pointer is valid (not 0)
            if current_pointer > 0:
                try:
                    # 2. Read the Raw Coordinates
                    # In Unity, Vector3s are contiguous floats (X, Y, Z)
                    raw_x = pm.read_float(current_pointer)  # Left/Right
                    raw_y = pm.read_float(current_pointer + 4)  # Jump Height / Up
                    raw_z = pm.read_float(current_pointer + 8)  # Forward Progress

                    # 3. Process for RL (Snap to Grid)
                    # This removes the "wobble" of jumping animations so your
                    # RL agent sees discrete, clean grid steps.
                    # (Adjust the 2.0 multiplier if Crossy Road's grid is a different size)
                    grid_x = round(raw_x / 2.0) * 2.0
                    grid_z = round(raw_z / 2.0) * 2.0

                    # Print dynamically on a single line
                    print(
                        f"Raw:[X: {raw_x:>6.2f} | Y: {raw_y:>6.2f} | Z: {raw_z:>6.2f}]  ==>  RL Grid:[X: {grid_x:>5.1f} | Z: {grid_z:>5.1f}]",
                        end="\r")

                except Exception as e:
                    # If reading the float fails (e.g., during a loading screen or respawn),
                    # we just pass and try again next frame.
                    pass

            # Pause to match your RL environment's step rate (e.g., ~60 FPS)
            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\nObservation loop stopped by user.")


if __name__ == "__main__":
    main()
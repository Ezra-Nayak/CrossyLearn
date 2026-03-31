import time
import pydirectinput
import pymem
from tracker import RamTracker


def measure_input_lag():
    rt = RamTracker()
    pm = pymem.Pymem("Crossy Road.exe")

    print("[LATENCY] Ready. Stand on Grass. I will press 'UP' in 2 seconds...")
    time.sleep(2)

    # Get Player Pointer from your Tracker
    coords = rt.get_coords()
    if not coords: return
    ptr = rt.pm.read_int(rt.pointer_location)

    start_z = pm.read_float(ptr + 8)

    print("[LATENCY] GO!")
    cmd_time = time.perf_counter()
    pydirectinput.press('up')

    # Poll RAM as fast as humanly possible
    while True:
        current_z = pm.read_float(ptr + 8)
        if abs(current_z - start_z) > 0.05:
            move_time = time.perf_counter()
            break

    latency = (move_time - cmd_time) * 1000
    print(f"\n[RESULTS] Total Input Latency: {latency:.2f}ms")
    print("This includes OS overhead, Python delay, and Game Animation start.")


if __name__ == "__main__":
    measure_input_lag()
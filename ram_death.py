import time
from tracker import RamTracker


def test_death_states():
    tracker = RamTracker()
    print("Waiting for game...")

    while not tracker.attach_and_inject():
        time.sleep(1)

    print("Hooked! Play the game and die in a few different ways.")
    print("Press CTRL+C to stop.\n")

    try:
        while True:
            coords = tracker.get_coords()
            if coords:
                x, y, z = coords
                print(f"[ALIVE?] Pointer Active -> X: {x: .3f} | Y: {y: .3f} | Z: {z: .3f}")
            else:
                print(f"[DEAD?] Pointer is NULL or lost!")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nTest complete.")


if __name__ == "__main__":
    test_death_states()
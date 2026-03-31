import pandas as pd
import numpy as np


def analyze_dump(filename="live_physics_dump.csv"):
    df = pd.read_csv(filename)

    # 1. Group by unique positions to find "True Ticks"
    # This removes all the duplicate reads where the engine hadn't updated yet
    ticks = df.groupby(['X', 'Z']).first().reset_index()
    ticks = ticks.sort_values(by='Timestamp')

    # 2. Calculate Deltas
    ticks['dt'] = ticks['Timestamp'].diff()
    ticks['dx'] = ticks['X'].diff()
    ticks['velocity'] = ticks['dx'] / ticks['dt']

    # Filter out the first NaN and any potential teleportation/static noise
    clean_ticks = ticks[(ticks['dt'] > 0.01) & (abs(ticks['velocity']) > 0.1)].copy()

    # 3. Statistics Extraction
    avg_v = clean_ticks['velocity'].mean()
    v_std = clean_ticks['velocity'].std()

    avg_w = df['W'].mean()
    avg_h = df['H'].mean()
    avg_d = df['D'].mean()

    print("\n" + "=" * 50)
    print("ENGINE GROUND TRUTH REPORT")
    print("=" * 50)
    print(f"True Physics Tick (Avg): {clean_ticks['dt'].mean() * 1000:.2f} ms")
    print(f"Target Velocity:         {avg_v:.4f} units/sec")
    print(f"Velocity Jitter (Std):   {v_std:.4f} (Lower is better)")
    print("-" * 50)
    print(f"RAM Bounding Box (W/H/D): {avg_w:.3f} / {avg_h:.3f} / {avg_d:.3f}")

    # CALCULATE THE "UNSAFE ZONE"
    # This is how much extra width you need to add to account for the car
    # moving during the 16.6ms tick interval.
    displacement_per_tick = abs(avg_v) * 0.0166
    print(f"Displacement per Tick:    {displacement_per_tick:.4f} units")
    print("=" * 50)

    print("\n[ACTIONABLE CONSTANTS FOR YOUR BOT]")
    print(f"VELOCITY_DEBOUNCE_MS = {round(clean_ticks['dt'].mean() * 1000) + 2}")
    print(f"COLLISION_BUFFER_X   = {displacement_per_tick / 2:.3f}")


if __name__ == "__main__":
    analyze_dump()
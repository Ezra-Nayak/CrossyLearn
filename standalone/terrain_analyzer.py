# --- terrain_analyzer.py ---

import cv2
import numpy as np
from enum import IntEnum

class Terrain(IntEnum):
    """Enumeration for terrain types. Using IntEnum for direct use in the state vector."""
    UNKNOWN = 0
    GRASS = 1
    ROAD = 2
    WATER = 3
    RAIL = 4
    LOG = 5
    LILLYPAD = 6
    TREE = 7
    ROCK = 8

# NOTE: These values are sensitive to display settings, lighting, and game updates.
# Use the terrain_tuner.py script to calibrate them for your system.
TERRAIN_HSV_RANGES = {
    Terrain.GRASS: ([41, 129, 109], [44, 165, 218]),
    Terrain.ROAD: ([109, 55, 90], [115, 60, 95]),
    Terrain.WATER: ([97, 140, 255], [99, 142, 255]),
    Terrain.RAIL: ([122, 55, 92], [125, 79, 170]),
    Terrain.LOG: ([0, 102, 110], [2, 126, 127]),
    Terrain.LILLYPAD: ([74, 213, 87], [80, 232, 187]),
    # Terrain.TREE: ([30, 188, 62], [40, 214, 190]), # top, not trunk
    Terrain.TREE: ([177, 101, 58], [179, 126, 66]), # trunk only (but also detects part of logs)
    Terrain.ROCK: ([128, 50, 110], [133, 62, 204]),
}

def get_terrain_type(patch):
    """
    Analyzes a small image patch and returns the dominant terrain type.
    """
    if patch is None or patch.shape[0] < 1 or patch.shape[1] < 1:
        return Terrain.UNKNOWN

    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv_patch, axis=(0, 1))
    h, s, v = avg_hsv

    for terrain, (lower, upper) in TERRAIN_HSV_RANGES.items():
        if (lower[0] <= h <= upper[0] and
                lower[1] <= s <= upper[1] and
                lower[2] <= v <= upper[2]):
            return terrain

    return Terrain.UNKNOWN
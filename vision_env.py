# --- vision_env.py ---
import cv2
import numpy as np
import torch
import mss
import win32gui
import pydirectinput
import time
import math
import os
from collections import deque
from train_vision import SplitBrainVAE, setup_device

# --- CONFIG ---
WINDOW_TITLE = "Crossy Road"
VAE_CHECKPOINT = "checkpoints/crossy_vae_latest.pth"
IMG_SIZE = 160
STACK_SIZE = 4
LATENT_DIM = 64


class VisionEnv:
    def __init__(self, render_mode=None):
        self.device = setup_device()
        print(f"[ENV] Loading VAE Eyes on {self.device}...")

        # Load the VAE
        self.vae = SplitBrainVAE().to(self.device)
        try:
            self.vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=self.device))
            self.vae.eval()  # Set to inference mode (Important!)
            print("[ENV] VAE Weights loaded successfully.")
        except FileNotFoundError:
            raise Exception("VAE Checkpoint not found! Train the vision model first.")

        self.sct = mss.mss()
        self.action_space = 4
        # State is now just the Latent Vector (64 floats)
        self.observation_space = (LATENT_DIM,)

        self.frame_buffer = deque(maxlen=STACK_SIZE)
        self._ensure_game_window()

    def _ensure_game_window(self):
        self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if not self.hwnd:
            print("Game window not found! Please open Crossy Road.")

    def get_window_geometry(self):
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        client_point = win32gui.ClientToScreen(self.hwnd, (left, top))
        return {
            "top": client_point[1] + 50,  # Skip Title bar
            "left": client_point[0],
            "width": right - left,
            "height": bottom - top - 50
        }

    def process_frame(self, frame):
        """
        1. CROP: Removes the top 15% (UI, Score, Pause Button).
        2. GRAYSCALE.
        3. RESIZE: To 160x160.
        4. NORMALIZE: 0-1.
        """
        h, w, _ = frame.shape

        # --- THE CUT ---
        # Crop top 15% (Approx 150px on a 1000px height)
        crop_h = int(h * 0.15)
        cropped = frame[crop_h:, :]
        # ----------------

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return resized / 255.0

    def reset(self):
        # Click Retry (Hardcoded coords for now, can use OCR trigger later)
        # Note: In PPO training, we usually handle the click loop carefully.
        # For now, we assume the game is running.

        self.frame_buffer.clear()

        # Fill buffer with initial frame
        monitor = self.get_window_geometry()
        img = np.array(self.sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        processed = self.process_frame(frame)

        for _ in range(STACK_SIZE):
            self.frame_buffer.append(processed)

        return self._get_latent_state()

    def _get_latent_state(self):
        # 1. Stack Frames
        # Shape: (4, 160, 160)
        stack = np.array(self.frame_buffer, dtype=np.float32)

        # 2. Add Batch Dimension -> (1, 4, 160, 160)
        tensor_in = torch.FloatTensor(stack).unsqueeze(0).to(self.device)

        # 3. VAE Inference
        with torch.no_grad():
            # We only need the Means (mu), not the logvars or reconstructions
            _, _, mu_c, _, mu_t, _ = self.vae(tensor_in)

            # Concatenate Context and Trend: (1, 32) + (1, 32) = (1, 64)
            latent_vector = torch.cat([mu_c, mu_t], dim=1)

        return latent_vector.cpu().numpy().flatten()

    def step(self, action):
        # Execute Action
        if action == 1:
            pydirectinput.press('up')
        elif action == 2:
            pydirectinput.press('left')
        elif action == 3:
            pydirectinput.press('right')
        # 0 is Idle

        # Wait for frame update (approx 15fps interval)
        time.sleep(0.06)

        # Capture New Frame
        monitor = self.get_window_geometry()
        img = np.array(self.sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Update Buffer
        processed = self.process_frame(frame)
        self.frame_buffer.append(processed)

        # Get New State (Latents)
        state = self._get_latent_state()

        # Reward/Done logic is still needed (we can bring back the OCR or death detection later)
        # For this test, we just return dummy values
        reward = 0
        done = False

        return state, reward, done


# --- TEST BLOCK ---
if __name__ == "__main__":
    print("Testing Vision Bridge...")
    env = VisionEnv()

    print("Resetting...")
    state = env.reset()

    print(f"State Shape: {state.shape}")
    print(f"State Example (First 10): {state[:10]}")

    print("Running Loop...")
    for i in range(50):
        s, r, d = env.step(1)  # Spam UP
        print(f"Step {i} | Vector Mean: {np.mean(s):.4f} | Vector Std: {np.std(s):.4f}")

        # Sanity Check: If Std is 0, the model is dead/outputting constant zeros
        if np.std(s) < 0.01:
            print("WARNING: Low variance in latent vector! Model might be outputting zeros.")
import torch
import time
import os
import numpy as np
import cv2

# Import the environment and network from your training script
from train_ppo import CrossyGameEnv, ActorCritic, PPO_DEVICE

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/ppo_crossy_0.pth"


def play():
    print(f"--- CROSSY ROAD AI EVALUATION (Spatial VAE Edition) ---")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        return

    # 1. Initialize Environment
    # We pass ui=None, but CrossyGameEnv still handles the VAE and Vision System
    env = CrossyGameEnv(ui=None)

    # 2. Load Policy (4 Actions: Up, Left, Right, Idle)
    policy = ActorCritic(action_dim=4).to(PPO_DEVICE)

    try:
        policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=PPO_DEVICE, weights_only=False))
        policy.eval()
        print(f"[SUCCESS] Loaded Model: {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return

    print("[INFO] Bring game to front. Starting in 3 seconds...")
    time.sleep(3)

    action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "IDLE"}

    try:
        while True:
            # env.reset() returns latents, scalars, mask
            latents, scalars, mask = env.reset()
            done = False
            steps = 0

            while not done:
                # Prepare Tensors
                lat_t = torch.FloatTensor(latents).to(PPO_DEVICE)
                sca_t = torch.FloatTensor(scalars).to(PPO_DEVICE)
                mask_t = torch.FloatTensor(mask).to(PPO_DEVICE)

                # Deterministic Action Selection
                with torch.no_grad():
                    # Get features via the Spatial CNN
                    features = policy._get_features(lat_t, sca_t)
                    logits = policy.actor(features)

                    if mask_t is not None:
                        logits = logits + mask_t

                    action = torch.argmax(logits, dim=1).item()

                # Execute Action
                latents, scalars, reward, done, mask = env.step(action)
                steps += 1

                # Print telemetry to console
                print(
                    f"\rStep: {steps:03d} | Action: {action_names.get(action):<5} | Z-Score: {env.last_score:<3} | Reward: {reward:>5.2f}",
                    end="")

                # Short delay to prevent terminal flicker, environment already has 150ms sleep
                if done:
                    print(f"\n[DEATH] Total Z-Score: {env.last_score} | Steps: {steps}")
                    time.sleep(2)

    except KeyboardInterrupt:
        print("\n[STOP] Evaluation ended by user.")


if __name__ == "__main__":
    play()
import torch
import time
import os
import numpy as np

# Import the environment and network from your training script
from train_ppo import CrossyGameEnv, ActorCritic, PPO_DEVICE

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/ppo_crossy_1000.pth"


def play():
    print(f"--- CROSSY ROAD AI EVALUATION ---")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        return

    # 1. Initialize Environment (UI=None ensures it runs quietly in console)
    env = CrossyGameEnv(ui=None)

    # Action Dim is 3 (Up, Left, Right), State Dim is 257
    policy = ActorCritic(257, 3).to(PPO_DEVICE)
    policy.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=PPO_DEVICE, weights_only=True))
    policy.eval()

    print(f"[SUCCESS] Loaded Model: {CHECKPOINT_PATH}")
    print("[INFO] Bring game to front. Starting in 3 seconds...")
    time.sleep(3)

    episodes = 10
    for ep in range(1, episodes + 1):
        state, action_mask = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            state_t = torch.FloatTensor(state).to(PPO_DEVICE)
            mask_t = torch.FloatTensor(action_mask).to(PPO_DEVICE)

            # Deterministic Action Selection (Choose best move, no exploration)
            with torch.no_grad():
                logits = policy.actor(state_t)
                if mask_t is not None:
                    logits = logits + mask_t
                action = torch.argmax(logits).item()

            # Execute
            state, reward, done, action_mask = env.step(action)
            ep_reward += reward
            steps += 1

            # Print telemetry to console
            action_names = {0: "UP", 1: "LEFT", 2: "RIGHT"}
            print(f"\rEp {ep} | Step: {steps} | Action: {action_names.get(action)} | Z-Score: {env.last_score}", end="")

        print(f"\n[DEATH] Episode {ep} ended. Total Z-Score: {env.last_score}")
        time.sleep(2)  # Wait on death screen before reset


if __name__ == "__main__":
    play()
import time
import os
import torch
import win32api
from train_ppo import CrossyGameEnv


# Action Map: 0:Up, 1:Left, 2:Right, 3:Idle
# VK Codes: UP = 0x26, LEFT = 0x25, RIGHT = 0x27

def get_human_action():
    """
    Reads the keyboard state asynchronously.
    0x8001 checks if the key is currently pressed OR was pressed since the last check.
    This ensures we don't miss quick 50ms human taps between the 100ms frame polls.
    """
    if win32api.GetAsyncKeyState(0x26) & 0x8001: return 0
    if win32api.GetAsyncKeyState(0x25) & 0x8001: return 1
    if win32api.GetAsyncKeyState(0x27) & 0x8001: return 2
    return 3


def main():
    os.makedirs("expert_data", exist_ok=True)
    print("--- CROSSY ROAD EXPERT RECORDER ---")
    print("Initializing Agent Senses (VAE + RAM)...")

    # Instantiate the exact environment the agent uses, but without the Rich UI
    env = CrossyGameEnv(ui=None)

    print("\n[READY] Bring Crossy Road into focus.")
    print("Play the game normally. Recording starts automatically when you are alive.")
    print("Press CTRL+C in this terminal to stop the script.\n")

    trajectory = []
    recording = False
    episodes_recorded = 0

    try:
        while True:
            # get_state() inherently waits for the OpenCV queue, limiting this loop to exactly 10 FPS
            latents, scalars, score, is_alive, action_mask = env.get_state()

            # Hardware glitch / VAE not ready
            if latents is None:
                time.sleep(0.1)
                continue

            # Capture human input during this 100ms window
            action = get_human_action()

            if is_alive:
                if not recording:
                    print("\n[REC] Game started! Recording trajectory...")
                    recording = True
                    trajectory = []
                    env.steps_in_episode = 1  # FIX: Unlock the Action Mask!

                # Store the exact format the PPO agent uses
                trajectory.append({
                    'latents': latents,  # [128, 20, 20] Split-Brain Latents
                    'scalars': scalars,  # [1] RAM X-Coord
                    'mask': action_mask,  # [4] Action Masking
                    'action': action  # Human Decision
                })

                print(
                    f"\r[LIVE] Steps: {len(trajectory):03d} | Score: {score} | Action: {['UP  ', 'LEFT', 'RGHT', 'IDLE'][action]}",
                    end="")
                env.steps_in_episode += 1  # FIX: Keep ticking to maintain normal boundary masks

            else:
                if recording:
                    recording = False
                    if len(trajectory) > 10:  # Only save runs that actually went somewhere
                        episodes_recorded += 1
                        filename = f"expert_data/run_{int(time.time())}.pt"
                        torch.save(trajectory, filename)
                        print(
                            f"\n[SAVE] Death detected. Saved {len(trajectory)} steps to {filename} (Total Runs: {episodes_recorded})")
                    else:
                        print("\n[SKIP] Run too short, discarded.")

                    trajectory = []
                    env.steps_in_episode = 0  # Reset for the next run

    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Exiting Expert Recorder.")


if __name__ == "__main__":
    main()
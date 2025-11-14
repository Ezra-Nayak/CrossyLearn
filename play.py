import torch
import time
import argparse
import sys

from crossy_env import CrossyRoadEnv
from agent.agent import DQNAgent

# --- CONFIGURATION ---
# To play with a specific model, set its path here.
# Example: MODEL_TO_PLAY = "models/crossy_agent_ep100.pth" or None
MODEL_TO_PLAY = "models/crossy_agent_ep450.pth"


def main(model_path):
    print("--- CrossyLearn Agent: Inference Mode ---")

    if not model_path:
        print("ERROR: No model path specified. Please set MODEL_TO_PLAY in the script or use the command line.")
        sys.exit(1)

    env = CrossyRoadEnv()

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    agent = DQNAgent(n_observations, n_actions)

    print(f"Loading model from: {model_path}")
    try:
        agent.policy_net.load_state_dict(torch.load(model_path))
        agent.policy_net.eval()
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load model. Reason: {e}")
        return

    print("Model loaded. Starting agent...")

    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    while True:
        try:
            state, _ = env.reset()
            time.sleep(2.0)
            state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

            done = False
            is_first_move = True

            while not done:
                with torch.no_grad():
                    q_values = agent.policy_net(state)
                    if is_first_move:
                        q_values[0, 1] = -float('inf')
                    action = q_values.max(1)[1].view(1, 1)

                if is_first_move:
                    is_first_move = False

                observation, _, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                if not done:
                    state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)

            print(f"Episode finished. Final Score: {info.get('raw_score', 0)}")

            # --- CRITICAL FIX ---
            # Wait a moment to ensure the VisionProducer has time to see the death screen
            # before the loop restarts and calls env.reset().
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopping inference.")
            break
        except Exception as e:
            print(f"An error occurred during play: {e}")
            break

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a trained DQN agent on Crossy Road.')
    parser.add_argument('model_path', type=str, nargs='?', default=None,
                        help='Path to the trained model (.pth) file. Overrides the hardcoded MODEL_TO_PLAY.')
    args = parser.parse_args()

    # Prioritize command-line arg, but fall back to the hardcoded variable
    model_to_run = args.model_path if args.model_path is not None else MODEL_TO_PLAY

    main(model_to_run)
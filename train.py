import torch
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

from crossy_env import CrossyRoadEnv
from agent.agent import DQNAgent, EPS_START, EPS_END, EPS_DECAY

# --- CONFIGURATION ---
NUM_EPISODES = 1000
# This cooldown is critical to prevent the 'left' input from opening the menu
POST_RESET_COOLDOWN = 2.0


def get_epsilon(steps_done):
    """Calculates the current epsilon value based on the decay formula."""
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)


def main():
    env = CrossyRoadEnv()

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    agent = DQNAgent(n_observations, n_actions)

    print("--- Starting Training ---")
    print(f"Agent will train for {NUM_EPISODES} episodes.")
    print(f"Device: {agent.device}")

    print("Performing initial reset to ensure clean start...")
    env.reset()
    time.sleep(POST_RESET_COOLDOWN)  # Apply cooldown after initial reset too

    episode_scores = []
    episode_avg_losses = []
    recent_scores = deque(maxlen=100)

    for i_episode in range(NUM_EPISODES):
        start_time = time.time()

        # This brief sleep ensures the vision thread sees the death screen before we reset
        time.sleep(0.5)
        state, _ = env.reset()

        # --- CRITICAL FIX: Wait for the game to be ready for input ---
        print(f"Applying {POST_RESET_COOLDOWN}s post-reset cooldown...")
        time.sleep(POST_RESET_COOLDOWN)

        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        done = False
        is_first_move = True
        episode_losses = []

        while not done:
            action_tensor = agent.select_action(state, is_first_move)
            action_item = action_tensor.item()

            # --- ASSERTION SAFETY NET ---
            if is_first_move:
                assert action_item != 1, f"FATAL: Agent selected 'left' (1) on first move!"
                is_first_move = False

            observation, reward, terminated, truncated, info = env.step(action_item)
            reward_tensor = torch.tensor([reward], device=agent.device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)

            agent.memory.push(state, action_tensor, next_state, reward_tensor)
            state = next_state

            loss = agent.optimize_model()
            if loss is not None:
                episode_losses.append(loss)

            agent.update_target_net()

        score = info.get('raw_score', 0)
        episode_scores.append(score)
        recent_scores.append(score)

        duration = time.time() - start_time
        avg_score = sum(recent_scores) / len(recent_scores)
        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
        episode_avg_losses.append(avg_loss)
        current_eps = get_epsilon(agent.steps_done)

        print(
            f"Ep {i_episode + 1} | "
            f"Score: {score} | "
            f"Avg Score: {avg_score:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {current_eps:.3f} | "
            f"Duration: {duration:.2f}s"
        )

    print('--- Training Complete ---')
    env.close()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title('Score per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(episode_scores, label='Score')
    rolling_avg = [np.mean(episode_scores[max(0, i - 100):i + 1]) for i in range(len(episode_scores))]
    plt.plot(rolling_avg, label='100-ep Average', color='orange')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(episode_avg_losses, label='Avg Loss', color='red')
    rolling_avg_loss = [np.mean(episode_avg_losses[max(0, i - 100):i + 1]) for i in range(len(episode_avg_losses))]
    plt.plot(rolling_avg_loss, label='100-ep Avg Loss', color='purple')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
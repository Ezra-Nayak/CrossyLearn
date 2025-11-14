import torch
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

from crossy_env import CrossyRoadEnv
from agent.agent import DQNAgent, EPS_START, EPS_END, EPS_DECAY

# --- CONFIGURATION ---
NUM_EPISODES = 500
POST_RESET_COOLDOWN = 2.0
SAVE_EVERY_N_EPISODES = 50  # Checkpoint saving frequency


def get_epsilon(steps_done):
    """Calculates the current epsilon value based on the decay formula."""
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)


def main():
    # --- SETUP ---
    env = CrossyRoadEnv()
    writer = SummaryWriter()  # NEW: Initialize TensorBoard writer

    # Create directories for saving models if they don't exist
    os.makedirs('models', exist_ok=True)

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    agent = DQNAgent(n_observations, n_actions)

    print("--- Starting Training ---")
    print(f"Agent will train for {NUM_EPISODES} episodes.")
    print(f"Device: {agent.device}")
    print(f"TensorBoard logs at: ./runs")

    print("Performing initial reset to ensure clean start...")
    env.reset()
    time.sleep(POST_RESET_COOLDOWN)

    episode_scores = []
    episode_avg_losses = []
    recent_scores = deque(maxlen=100)

    # --- MAIN TRAINING LOOP ---
    for i_episode in range(NUM_EPISODES):
        start_time = time.time()

        time.sleep(0.5)
        state, _ = env.reset()
        time.sleep(POST_RESET_COOLDOWN)

        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        done = False
        is_first_move = True
        episode_losses = []

        while not done:
            action_tensor = agent.select_action(state, is_first_move)
            action_item = action_tensor.item()

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

        # --- LOGGING AND SAVING ---
        score = info.get('raw_score', 0)
        duration = time.time() - start_time
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        current_eps = get_epsilon(agent.steps_done)

        episode_scores.append(score)
        recent_scores.append(score)
        avg_score = np.mean(recent_scores)

        # NEW: Log to TensorBoard
        writer.add_scalar('Score/Episode', score, i_episode)
        writer.add_scalar('Score/Average_100_Episodes', avg_score, i_episode)
        writer.add_scalar('Loss/Average_Episode', avg_loss, i_episode)
        writer.add_scalar('Meta/Epsilon', current_eps, i_episode)
        writer.add_scalar('Meta/Episode_Duration', duration, i_episode)

        print(
            f"Ep {i_episode + 1} | "
            f"Score: {score} | "
            f"Avg Score: {avg_score:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {current_eps:.3f} | "
            f"Duration: {duration:.2f}s"
        )

        # NEW: Checkpoint saving
        if (i_episode + 1) % SAVE_EVERY_N_EPISODES == 0:
            model_path = f"models/crossy_agent_ep{i_episode + 1}.pth"
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"--- Checkpoint saved to {model_path} ---")

    # --- CLEANUP ---
    print('--- Training Complete ---')
    final_model_path = "models/crossy_agent_final.pth"
    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    env.close()
    writer.close()


if __name__ == '__main__':
    main()
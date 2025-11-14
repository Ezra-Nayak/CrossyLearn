import torch
import time
from collections import deque
import numpy as np
import os
import re
import argparse
from torch.utils.tensorboard import SummaryWriter

from crossy_env import CrossyRoadEnv
from agent.agent import DQNAgent, EPS_START, EPS_END, EPS_DECAY

# --- CONFIGURATION ---
# To resume training, set this path. To start fresh, set it to None.
# Example: RESUME_FROM_MODEL = "models/crossy_agent_ep100.pth" or None
RESUME_FROM_MODEL = "models/crossy_agent_ep450.pth"

NUM_EPISODES = 1000
POST_RESET_COOLDOWN = 2.0
SAVE_EVERY_N_EPISODES = 50


def get_epsilon(steps_done):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)


def main(resume_path):
    env = CrossyRoadEnv()
    writer = SummaryWriter()
    os.makedirs('models', exist_ok=True)

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    agent = DQNAgent(n_observations, n_actions)
    start_episode = 0

    if resume_path:
        print(f"Resuming training from: {resume_path}")
        try:
            agent.policy_net.load_state_dict(torch.load(resume_path))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

            match = re.search(r'ep(\d+)\.pth', resume_path)
            if match:
                start_episode = int(match.group(1))
                agent.steps_done = start_episode * 20
                print(f"Resuming from episode {start_episode}. Epsilon will be ~{get_epsilon(agent.steps_done):.3f}")
            else:
                print("Could not determine episode number from filename, starting from 0.")
        except FileNotFoundError:
            print(f"WARNING: Resume model not found at {resume_path}. Starting from scratch.")
        except Exception as e:
            print(f"WARNING: Failed to load model ({e}). Starting from scratch.")

    print("--- Starting Training ---")
    print(f"Agent will train for {NUM_EPISODES} episodes, starting from {start_episode}.")
    print(f"Device: {agent.device}")

    print("Performing initial reset to ensure clean start...")
    env.reset()
    time.sleep(POST_RESET_COOLDOWN)

    recent_scores = deque(maxlen=100)

    for i_episode in range(start_episode, start_episode + NUM_EPISODES):
        start_time = time.time()

        time.sleep(0.5)
        state, _ = env.reset()
        time.sleep(POST_RESET_COOLDOWN)

        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        done = False;
        is_first_move = True;
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

            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32,
                                                              device=agent.device).unsqueeze(0)

            agent.memory.push(state, action_tensor, next_state, reward_tensor)
            state = next_state

            loss = agent.optimize_model()
            if loss is not None: episode_losses.append(loss)

            agent.update_target_net()

        score = info.get('raw_score', 0)
        duration = time.time() - start_time
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        current_eps = get_epsilon(agent.steps_done)

        recent_scores.append(score)
        avg_score = np.mean(recent_scores)

        writer.add_scalar('Score/Episode', score, i_episode)
        writer.add_scalar('Score/Average_100_Episodes', avg_score, i_episode)
        writer.add_scalar('Loss/Average_Episode', avg_loss, i_episode)
        writer.add_scalar('Meta/Epsilon', current_eps, i_episode)
        writer.add_scalar('Meta/Episode_Duration', duration, i_episode)

        status = info.get("status", "finished")
        print(
            f"Ep {i_episode + 1:<4} | "
            f"Status: {status.upper():<8} | "
            f"Score: {score:<3} | "
            f"Avg Score: {avg_score:<5.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epsilon: {current_eps:.3f} | "
            f"Duration: {duration:.2f}s"
        )

        if (i_episode + 1) % SAVE_EVERY_N_EPISODES == 0:
            model_path = f"models/crossy_agent_ep{i_episode + 1}.pth"
            torch.save(agent.policy_net.state_dict(), model_path)
            print(f"--- Checkpoint saved to {model_path} ---")

    print('--- Training Complete ---')
    final_model_path = f"models/crossy_agent_ep{start_episode + NUM_EPISODES}.pth"
    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    env.close()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DQN agent for Crossy Road.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a saved model to resume training from. Overrides the hardcoded RESUME_FROM_MODEL.')
    args = parser.parse_args()

    # Prioritize command-line arg, but fall back to the hardcoded variable
    model_to_load = args.resume if args.resume is not None else RESUME_FROM_MODEL

    main(model_to_load)
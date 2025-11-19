# --- train.py ---
import torch
import torch.optim as optim
import numpy as np
import random
import time
import sys
import os
from collections import deque
from model import DuelingDQN, setup_device
from crossy_env import CrossyEnv, GameCrashedError

# HYPERPARAMETERS
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 5000 # Increased decay since we run much faster now (more steps/sec)
TARGET_UPDATE = 15
MEMORY_SIZE = 20000
LR = 0.00025 # Lower LR for RMSprop
NUM_EPISODES = 2000

device = setup_device()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


def select_action(state, eps_threshold, policy_net, first_move=False):
    # Valid actions: 0:Idle, 1:Up, 2:Left, 3:Right
    # If first_move is True, 2 is invalid.

    if random.random() > eps_threshold:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_t)

            if first_move:
                # Set Q-value of Left (index 2) to negative infinity so it's never picked
                q_values[0, 2] = -float('inf')

            return q_values.max(1)[1].item()
    else:
        # Random sample
        valid_actions = [0, 1, 3] if first_move else [0, 1, 2, 3]
        return random.choice(valid_actions)


def main():
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    env = CrossyEnv()

    policy_net = DuelingDQN(env.state_dim, env.action_space).to(device)
    target_net = DuelingDQN(env.state_dim, env.action_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # RMSprop is standard for DQN and plays nice with DirectML
    optimizer = optim.RMSprop(policy_net.parameters(), lr=LR, alpha=0.95, eps=0.01)
    memory = ReplayBuffer(MEMORY_SIZE)

    steps_done = 0

    print(f"--- SOTA Training Started on {device} ---")
    print(f"--- Crash Recovery & Debugger Active ---")

    for i_episode in range(NUM_EPISODES):
        try:
            # CRASH RECOVERY WRAPPER
            state = env.reset()
            total_reward = 0

            while True:
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                np.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1

                # Pass 'first_move' flag to action selector
                is_first = (env.steps_in_episode == 0)
                action = select_action(state, eps_threshold, policy_net, first_move=is_first)

                # Step
                next_state, reward, done = env.step(action)

                memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Optimization Step
                if len(memory) >= BATCH_SIZE:
                    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                    states_t = torch.FloatTensor(states).to(device)
                    actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards_t = torch.FloatTensor(rewards).to(device)
                    next_states_t = torch.FloatTensor(next_states).to(device)
                    dones_t = torch.FloatTensor(dones).to(device)

                    with torch.no_grad():
                        best_actions = policy_net(next_states_t).max(1)[1].unsqueeze(1)
                        next_q_values = target_net(next_states_t).gather(1, best_actions).squeeze(1)
                        target_q_values = rewards_t + (GAMMA * next_q_values * (1 - dones_t))

                    current_q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)
                    loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                    print(
                        f"Ep {i_episode} | R: {total_reward:.2f} | Eps: {eps_threshold:.2f} | Steps: {env.steps_in_episode}")
                    break

            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if i_episode % 50 == 0:
                torch.save(policy_net.state_dict(), f"checkpoints/crossy_ep{i_episode}.pth")

        except GameCrashedError:
            print(f"\n[CRITICAL] Game crash detected in Episode {i_episode}!")
            print("[TRAINER] Discarding current run and restarting environment...")
            # We do NOT increment i_episode automatically, but in a for loop we can't easily decrement.
            # We accept the lost episode number, but the bad data is not in memory (if we clear or just ignore).
            # Actually, since we push to memory at every step, some steps from the crashed run are in memory.
            # This is acceptable as long as the final state wasn't garbage.
            # The `reset()` call at start of next loop will handle the re-launch.
            time.sleep(5)
            continue

        except KeyboardInterrupt:
            print("Saving and Exiting...")
            torch.save(policy_net.state_dict(), "crossy_final.pth")
            break


if __name__ == "__main__":
    main()
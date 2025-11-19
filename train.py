# --- train.py ---
import torch
import torch.optim as optim
import numpy as np
import random
import time
from collections import deque
from model import DuelingDQN, setup_device
from crossy_env import CrossyEnv

# HYPERPARAMETERS
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 2000
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LR = 0.001
NUM_EPISODES = 1000

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


def main():
    env = CrossyEnv()

    # Initialize Networks
    policy_net = DuelingDQN(env.state_dim, env.action_space).to(device)
    target_net = DuelingDQN(env.state_dim, env.action_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    steps_done = 0

    print(f"--- Starting Training on {device} ---")

    try:
        for i_episode in range(NUM_EPISODES):
            state = env.reset()
            total_reward = 0

            while True:
                # Select Action (Epsilon Greedy)
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                np.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1

                if random.random() > eps_threshold:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_values = policy_net(state_t)
                        action = q_values.max(1)[1].item()
                else:
                    action = random.randint(0, env.action_space - 1)

                # Step
                next_state, reward, done = env.step(action)
                memory.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                # Optimize Model
                if len(memory) >= BATCH_SIZE:
                    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                    states_t = torch.FloatTensor(states).to(device)
                    actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards_t = torch.FloatTensor(rewards).to(device)
                    next_states_t = torch.FloatTensor(next_states).to(device)
                    dones_t = torch.FloatTensor(dones).to(device)

                    # Double DQN Logic
                    with torch.no_grad():
                        # Select best action using Policy Net
                        best_actions = policy_net(next_states_t).max(1)[1].unsqueeze(1)
                        # Evaluate that action using Target Net
                        next_q_values = target_net(next_states_t).gather(1, best_actions).squeeze(1)

                        target_q_values = rewards_t + (GAMMA * next_q_values * (1 - dones_t))

                    current_q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)

                    loss = torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                    print(f"Episode {i_episode} | Reward: {total_reward:.2f} | Epsilon: {eps_threshold:.2f}")
                    break

            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Periodic Save
            if i_episode % 50 == 0:
                torch.save(policy_net.state_dict(), f"crossy_model_ep{i_episode}.pth")

    except KeyboardInterrupt:
        print("Training Interrupted. Saving Model...")
        torch.save(policy_net.state_dict(), "crossy_model_interrupted.pth")


if __name__ == "__main__":
    main()
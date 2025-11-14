import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math

from agent.DQN import DQN, setup_device
from agent.memory import ReplayMemory, Transition

# --- HYPERPARAMETERS ---
# These values are critical for tuning the agent's learning behavior.
BATCH_SIZE = 128  # Number of transitions sampled from the replay buffer.
GAMMA = 0.99  # Discount factor for future rewards.
EPS_START = 0.9  # Starting value of epsilon (probability of taking a random action).
EPS_END = 0.05  # Minimum value of epsilon.
EPS_DECAY = 1000  # Controls how fast epsilon decays.
TAU = 0.005  # The update rate of the target network.
LR = 1e-4  # The learning rate of the AdamW optimizer.


class DQNAgent:
    def __init__(self, n_observations, n_actions):
        self.device = setup_device()
        self.n_actions = n_actions

        # Create two networks: one for policy, one for target.
        # The policy network is updated every step.
        # The target network is updated slowly, which stabilizes learning.
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for evaluation

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(10000)  # Store up to 10,000 experiences

        self.steps_done = 0

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        With probability epsilon, take a random action (explore).
        Otherwise, take the best action according to the policy network (exploit).
        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # state.max(1) returns the largest value in each row of the tensor.
                # .view(1, 1) reshapes it to the desired dimensions for the action.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Return a random action
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """Performs one step of the optimization (learning)."""
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough memories to learn from yet

        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details).
        batch = Transition(*zip(*transitions))

        # Create batches of states, actions, rewards, etc.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state, according to policy_net.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()  # Return the loss value for logging

    def update_target_net(self):
        """
        Soft update of the target network's weights:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)
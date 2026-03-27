# --- train_ppo.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import cv2
import time
import os
import queue
import mss
import win32gui
import pydirectinput
from collections import deque
from train_vision import SplitBrainVAE, setup_device
from vision import VisionSystem
from tracker import RamTracker

import subprocess
import os

EXECUTABLE_PATH = r"C:\Program Files\WindowsApps\Yodo1Ltd.CrossyRoad_1.3.4.0_x86__s3s3f300emkze\Crossy Road.exe"
RETRY_BUTTON_COORDS = (767, 862) # Standard Retry button

# --- HYPERPARAMETERS ---
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95  # SOTA GAE Advantage smoothing
EPS_CLIP = 0.2
K_EPOCHS = 4
UPDATE_TIMESTEP = 2000
MAX_EPISODES = 10000
HIDDEN_DIM = 512    # Increased to handle the larger 256 latent vector
LATENT_DIM = 256    # SYNCED with train_vision.py
STACK_SIZE = 4
IMG_SIZE = 160

# --- CONFIG ---
VAE_CHECKPOINT = "checkpoints/crossy_vae_latest.pth"
WINDOW_TITLE = "Crossy Road"

# HYBRID COMPUTE SETUP
# VAE (Images) -> GPU (DirectML) for Speed
# PPO (Logic)  -> CPU for Stability (Fixes DirectML Scatter/Backward Crash)
VAE_DEVICE = setup_device()
PPO_DEVICE = torch.device("cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states =[]
        self.logprobs = []
        self.rewards = []
        self.is_terminals =[]
        self.values = []
        self.action_masks =[]

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
        del self.action_masks[:]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # SOTA: Disjoint networks prevent destructive interference between Actor and Critic
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_DIM, HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_DIM, action_dim), std=0.01)  # Low std for initial exploration
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_DIM, HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_DIM, 1), std=1.0)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, action_mask):
        action_logits = self.actor(state)

        # Action Masking: Add massive negative penalty to logits of invalid actions
        if action_mask is not None:
            action_logits = action_logits + action_mask

        # Passing logits directly is numerically more stable than explicit Softmax
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(state)

        return action.item(), action_logprob.item(), state_value.item()

    def evaluate(self, state, action, action_mask):
        action_logits = self.actor(state)

        if action_mask is not None:
            action_logits = action_logits + action_mask

        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim):
        # Force PPO to CPU
        self.policy = ActorCritic(state_dim, action_dim).to(PPO_DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim).to(PPO_DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # SOTA: Generalized Advantage Estimation (GAE)
        advantages = []
        last_gae_lam = 0

        for step in reversed(range(len(memory.rewards))):
            if step == len(memory.rewards) - 1:
                next_non_terminal = 1.0 - memory.is_terminals[step]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - memory.is_terminals[step]
                next_value = memory.values[step + 1]

            delta = memory.rewards[step] + GAMMA * next_value * next_non_terminal - memory.values[step]
            last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)

        returns = [adv + val for adv, val in zip(advantages, memory.values)]

        # Convert to tensors
        returns = torch.tensor(returns, dtype=torch.float32).to(PPO_DEVICE)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(PPO_DEVICE)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Stack tensors (CPU)
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(PPO_DEVICE)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(PPO_DEVICE)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(PPO_DEVICE)
        old_masks = torch.squeeze(torch.stack(memory.action_masks, dim=0)).detach().to(PPO_DEVICE)

        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_masks)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs)

            # Actor loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            actor_loss = -torch.min(surr1, surr2)

            # Critic loss
            critic_loss = 0.5 * self.MseLoss(state_values, returns)

            # Total loss
            loss = actor_loss + critic_loss - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


class CrossyGameEnv:
    def __init__(self):
        self.vae = SplitBrainVAE().to(VAE_DEVICE)
        self.vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=VAE_DEVICE))
        self.vae.eval()

        self.vision_q = queue.Queue(maxsize=1)
        self.vis = VisionSystem(WINDOW_TITLE, self.vision_q, show_preview=False)
        self.vis.start()

        self.ram_tracker = RamTracker()

        self.frame_buffer = deque(maxlen=STACK_SIZE)
        self.last_score = 0
        self.next_milestone = 10 # Tracks the next +5 reward target
        self.steps_stationary = 0
        self.steps_in_episode = 0
        self.last_known_coords = (0.0, 0.0)
        self.sct = mss.mss()
        self._ensure_game_running()

    def _ensure_game_running(self):
        self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if not self.hwnd:
            print("[RECOVERY] Game window not found. Launching Crossy Road...")
            try:
                # Launching WindowsApps sometimes requires 'start' shell command
                os.system(f'start "" "{EXECUTABLE_PATH}"')

                # Wait for load (WindowsApps can be slow)
                print("[RECOVERY] Waiting 9 seconds for startup...")
                time.sleep(9)

                self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
                if not self.hwnd:
                    print("[ERROR] Failed to find window after launch.")
                    return False

                # Focus and pass splash screen
                win32gui.SetForegroundWindow(self.hwnd)
                time.sleep(2)
                pydirectinput.press('space')
                print("[RECOVERY] Splash screen passed.")
                time.sleep(2)
            except Exception as e:
                print(f"[RECOVERY] Failed to launch: {e}")
                return False
        return True

    def process_frame(self, frame):
        h, w, _ = frame.shape
        crop_h = int(h * 0.16)
        cropped = frame[crop_h:, :]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return resized / 255.0

    def get_geometry(self):
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        pt = win32gui.ClientToScreen(self.hwnd, (left, top))
        return {"top": pt[1] + 50, "left": pt[0], "width": right - left, "height": bottom - top - 50}

    def get_state(self):
        # 1. Get Visual Data
        try:
            data = self.vision_q.get(timeout=1.0)
        except queue.Empty:
            return None, 0, True, {}  # Fail state

        frame = data['frame']

        # 2. Process VAE Input
        processed = self.process_frame(frame)
        self.frame_buffer.append(processed)

        # Need 4 frames to see
        if len(self.frame_buffer) < STACK_SIZE:
            # Pad with current frame
            while len(self.frame_buffer) < STACK_SIZE:
                self.frame_buffer.append(processed)

        stack = np.array(self.frame_buffer, dtype=np.float32)
        tensor_in = torch.FloatTensor(stack).unsqueeze(0).to(VAE_DEVICE)

        with torch.no_grad():
            # Original VAE returns: recon_static, pred_next, mu_c, log_c, mu_t, log_t
            _, _, mu_c, _, mu_t, _ = self.vae(tensor_in)
            # Concatenate Context (128) and Trend (128) = 256
            latents = torch.cat([mu_c, mu_t], dim=1).cpu().numpy().flatten()

        # 3. Proprioception & Score (RAM Tracking)
        # Fetch coordinates automatically bypassing CV completely
        coords = self.ram_tracker.get_coords()
        if coords:
            raw_x, raw_y, raw_z = coords
            # Grid align to prevent jump animation jitter (1.0 units per tile)
            grid_x = round(raw_x)
            grid_z = round(raw_z)
            self.last_known_coords = (grid_x, grid_z)

        # SOTA Normalization: Remove Z (infinitely scaling score) from State.
        # Normalize X (approx bounds -4 to 4) by dividing by 5.0 to map it tightly to [-1, 1]
        norm_x = self.last_known_coords[0] / 5.0
        state_vec = np.concatenate([latents, [norm_x]])

        # Action Masking: 0:Idle, 1:Up, 2:Left, 3:Right
        # Prevent the agent from walking off the visible map edges
        action_mask = np.zeros(4, dtype=np.float32)
        if self.last_known_coords[0] <= -4:
            action_mask[2] = -1e8  # Heavily penalize Left
        elif self.last_known_coords[0] >= 4:
            action_mask[3] = -1e8  # Heavily penalize Right

        # Use Z axis directly as the continuous score (progress tracker)
        current_score = self.last_known_coords[1]

        return state_vec, current_score, data['pause_visible'], action_mask

    def reset(self):
        # 1. Crash Check
        if not self._ensure_game_running():
            print("[CRITICAL] Could not recover game.")
            time.sleep(5)
            return self.reset()  # Recursive retry

        # 2. Focus and Click Retry
        try:
            win32gui.SetForegroundWindow(self.hwnd)
        except:
            pass

        geo = self.get_geometry()
        rx, ry = geo['left'] + RETRY_BUTTON_COORDS[0], geo['top'] + RETRY_BUTTON_COORDS[1]

        # Click Retry / Tap to Start
        pydirectinput.moveTo(rx, ry)
        pydirectinput.click()
        time.sleep(0.5)
        pydirectinput.press('space')

        # 3. Clear memory for new run
        self.frame_buffer.clear()
        self.steps_stationary = 0
        self.steps_in_episode = 0
        self.last_known_coords = (0.0, 0.0)

        # 4. WAIT FOR GAME START (Sync Logic)
        # We loop here until the Vision System confirms the Pause Button is visible.
        timeout_start = time.time()
        while time.time() - timeout_start < 5.0:  # 5 second timeout
            time.sleep(0.1)
            state, z_score, is_alive, action_mask = self.get_state()

            # If vision is working AND game thinks we are alive:
            if state is not None and is_alive:
                # Sync initial score properly so it doesn't instantly penalize/reward
                self.last_score = z_score
                self.next_milestone = self.last_score + 10
                return state, action_mask

        # If we reach here, the game didn't start (missed click?). Retry reset.
        return self.reset()

    def step(self, action):
        # --- ACTION SHIELDING ---
        # If this is the very first move of the run, we are still in the menu.
        # Force 'Up' (1) to start the game and prevent entering Costume/Character menus.
        if self.steps_in_episode == 0:
            if action != 1:
                # Override whatever the agent suggested (usually Left or Idle)
                action = 1

                # Action: 0:Idle, 1:Up, 2:Left, 3:Right
        if action == 1:
            pydirectinput.press('up')
        elif action == 2:
            pydirectinput.press('left')
        elif action == 3:
            pydirectinput.press('right')

        self.steps_in_episode += 1  # Increment step counter

        # Latency Wait
        time.sleep(0.08)  # 80ms reaction time

        next_state, score, is_alive, action_mask = self.get_state()

        # --- REWARD ENGINEERING ---
        # BASE STEP PENALTY: Super small. Agent can survive and wait for cars to pass.
        reward = -0.002
        done = False

        # 1. Progression Reward (Using Z Coordinate)
        if score > self.last_score:
            # Z increases by exactly 1.0 per grid hop forward
            reward += 2.0 * (score - self.last_score)  # Slightly more aggressive progress reward

            # --- MILESTONE BONUS ---
            if score >= self.next_milestone:
                reward += 5.0
                print(f"[BONUS] Milestone reached! Z-Score: {score}")
                self.next_milestone += 10
            # -----------------------

            self.last_score = score
            self.steps_stationary = 0
        elif score < self.last_score:
            # INNOVATION: Penalize stepping backward.
            # We use 2.0 to match the forward reward, making "back-and-forth" a net zero or loss.
            reward -= 2.0 * (self.last_score - score)
            self.last_score = score
            self.steps_stationary = 0

        # 2. Idle Penalty / Sideways Penalty
        if action == 0:
            reward -= 0.015  # SOTA: Allow agent to be patient. Was -0.1.
            self.steps_stationary += 1
        elif action == 2 or action == 3:
            reward -= 0.001  # Almost negligible, allows tactical dodging.

        # 3. Death Detection
        if not is_alive:
            reward = -10.0
            done = True

        return next_state, reward, done, action_mask


import glob


def train():
    env = CrossyGameEnv()
    # State Dim is now 257 (256 Latents + 1 Normalized X)
    ppo = PPO(257, 4)
    memory = Memory()

    start_episode = 1

    # --- AUTO RESUME LOGIC ---
    checkpoints = glob.glob("checkpoints/ppo_crossy_*.pth")
    if checkpoints:
        # Find the one with the highest number
        latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"[RESUME] Loading checkpoint: {latest_cp}")

        # Load weights
        ppo.policy.load_state_dict(torch.load(latest_cp, map_location=PPO_DEVICE))
        ppo.policy_old.load_state_dict(ppo.policy.state_dict())

        # Parse episode number
        start_episode = int(latest_cp.split('_')[-1].split('.')[0]) + 1
        print(f"[RESUME] Resuming from Episode {start_episode}")
    else:
        print("--- STARTING NEW TRAINING SESSION ---")
    print("Focus the game window!")
    time.sleep(3)

    time_step = 0

    for i_episode in range(start_episode, MAX_EPISODES + 1):
        state, action_mask = env.reset()
        current_ep_reward = 0

        # Safety Loop Break
        for t in range(1000):
            time_step += 1

            # Select Action (Move state to CPU for PPO)
            state_t = torch.FloatTensor(state).to(PPO_DEVICE)
            mask_t = torch.FloatTensor(action_mask).to(PPO_DEVICE)

            action, logprob, value = ppo.policy_old.act(state_t, mask_t)

            # Execute
            next_state, reward, done, next_action_mask = env.step(action)

            # Handle Step Failure
            if next_state is None:
                print("[PPO] Step failed. Resetting...")
                done = True
                reward = -10.0

            # Store (CPU)
            memory.states.append(state_t)
            memory.actions.append(torch.tensor(action).to(PPO_DEVICE))
            memory.logprobs.append(torch.tensor(logprob).to(PPO_DEVICE))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.values.append(value)
            memory.action_masks.append(mask_t)

            state = next_state
            action_mask = next_action_mask
            current_ep_reward += reward

            # Update Policy
            if time_step % UPDATE_TIMESTEP == 0:
                print(f"[PPO] Updating Policy at Step {time_step}...")
                ppo.update(memory)
                memory.clear()
                time_step = 0

            if done:
                break

        print(f"Ep {i_episode} | Reward: {current_ep_reward:.2f} | Score: {env.last_score}")

        # Save Checkpoint
        if i_episode % 50 == 0:
            torch.save(ppo.policy.state_dict(), f"checkpoints/ppo_crossy_{i_episode}.pth")


if __name__ == "__main__":
    train()
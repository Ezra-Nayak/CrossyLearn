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

import subprocess
import os

EXECUTABLE_PATH = r"C:\Program Files\WindowsApps\Yodo1Ltd.CrossyRoad_1.3.4.0_x86__s3s3f300emkze\Crossy Road.exe"
RETRY_BUTTON_COORDS = (767, 862) # Standard Retry button

# --- HYPERPARAMETERS ---
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
UPDATE_TIMESTEP = 2000
MAX_EPISODES = 10000
HIDDEN_DIM = 256
LATENT_DIM = 64
STACK_SIZE = 4
IMG_SIZE = 160

# --- CONFIG ---
VAE_CHECKPOINT = "checkpoints/crossy_vae_latest.pth"
WINDOW_TITLE = "Crossy Road"
DEVICE = setup_device()


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # State Dim = 64 (VAE) + 2 (XY Coords) = 66
        self.layer_common = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh()
        )

        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        x = self.layer_common(state)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item()

    def evaluate(self, state, action):
        x = self.layer_common(state)
        action_probs = self.actor(x)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(x)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(DEVICE)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(DEVICE)

        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


class CrossyGameEnv:
    def __init__(self):
        self.vae = SplitBrainVAE().to(DEVICE)
        self.vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=DEVICE))
        self.vae.eval()

        self.vision_q = queue.Queue(maxsize=1)
        self.vis = VisionSystem(WINDOW_TITLE, self.vision_q, show_preview=False)
        self.vis.start()

        self.frame_buffer = deque(maxlen=STACK_SIZE)
        self.last_score = 0
        self.steps_stationary = 0
        self.steps_in_episode = 0 # NEW: Tracks moves in current run
        self.last_known_coords = (0.5, 0.8)
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
                print("[RECOVERY] Waiting 15 seconds for startup...")
                time.sleep(15)

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
        crop_h = int(h * 0.15)
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
        score_str = data['score']

        # 2. Process VAE Input
        processed = self.process_frame(frame)
        self.frame_buffer.append(processed)

        # Need 4 frames to see
        if len(self.frame_buffer) < STACK_SIZE:
            # Pad with current frame
            while len(self.frame_buffer) < STACK_SIZE:
                self.frame_buffer.append(processed)

        stack = np.array(self.frame_buffer, dtype=np.float32)
        tensor_in = torch.FloatTensor(stack).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _, _, mu_c, _, mu_t, _ = self.vae(tensor_in)
            latents = torch.cat([mu_c, mu_t], dim=1).cpu().numpy().flatten()

        # 3. Proprioception (Chicken XY)
        # --- MODIFIED COORDINATE LOGIC ---
        cx, cy = self.last_known_coords # Start with memory

        if data['chicken_pos']:
            # We have a lock (or a coast), update memory
            cx = data['chicken_pos'][0] / frame.shape[1]
            cy = data['chicken_pos'][1] / frame.shape[0]
            self.last_known_coords = (cx, cy)
        else:
            # Tracker is totally LOST (behind truck for >10 frames).
            # We keep using self.last_known_coords.
            # This simulates "Object Permanence" - we assume it's where we last saw it.
            pass

        state_vec = np.concatenate([latents, [cx, cy]])

        # 4. Score Logic
        current_score = 0
        try:
            if score_str: current_score = int(score_str)
        except:
            pass

        return state_vec, current_score, data['pause_visible'], data

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
        self.last_score = 0
        self.steps_stationary = 0
        self.steps_in_episode = 0
        self.last_known_coords = (0.5, 0.8)  # Reset coords to safe default

        # 4. WAIT FOR GAME START (Sync Logic)
        # We loop here until the Vision System confirms the Pause Button is visible.
        # This prevents "Ghost Episodes" where the agent acts while the menu is fading out.
        timeout_start = time.time()
        while time.time() - timeout_start < 5.0:  # 5 second timeout
            time.sleep(0.1)
            state, _, is_alive, _ = self.get_state()

            # If vision is working AND game thinks we are alive:
            if state is not None and is_alive:
                return state

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

        next_state, score, is_alive, info = self.get_state()

        # --- REWARD ENGINEERING ---
        reward = -0.01  # Existence penalty
        done = False

        # 1. Progression Reward
        if score > self.last_score:
            reward += 1.0 * (score - self.last_score)
            self.last_score = score
            self.steps_stationary = 0

        # 2. Idle Penalty / Backward Penalty
        if action == 0:
            reward -= 0.1
            self.steps_stationary += 1
        elif action == 2 or action == 3:
            # Moving sideways is okay but not great
            reward -= 0.05

        # 3. Death Detection
        # Rely on 'pause_visible' (True = Alive, False = Dead/Menu)
        # Wait, if pause button is missing, we are likely dead.
        if not is_alive:
            reward = -10.0
            done = True

        return next_state, reward, done, info


def train():
    env = CrossyGameEnv()

    # State dim = 64 (Latent) + 2 (XY) = 66
    # Action dim = 4
    ppo = PPO(66, 4)
    memory = Memory()

    print("--- STARTING PPO TRAINING ---")
    print("Focus the game window!")
    time.sleep(3)

    time_step = 0

    for i_episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        current_ep_reward = 0

        # Safety Loop Break
        for t in range(1000):
            time_step += 1

            # Select Action
            state_t = torch.FloatTensor(state).to(DEVICE)
            action, logprob = ppo.policy_old.act(state_t)

            # Execute
            next_state, reward, done, _ = env.step(action)

            # Handle Step Failure (e.g. game crashed mid-episode)
            if next_state is None:
                print("[PPO] Step failed (Game Crash?). Forcing Reset...")
                done = True
                reward = -10.0 # Penalty for crashing/losing state

            # Store
            memory.states.append(state_t)
            memory.actions.append(torch.tensor(action).to(DEVICE))
            memory.logprobs.append(torch.tensor(logprob).to(DEVICE))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state
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
        if i_episode % 20 == 0:
            torch.save(ppo.policy.state_dict(), f"checkpoints/ppo_crossy_{i_episode}.pth")


if __name__ == "__main__":
    train()
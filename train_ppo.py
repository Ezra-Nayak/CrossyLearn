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
# Remove the default 0.1s delay between inputs to unlock loop speed
pydirectinput.PAUSE = 0

from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich.table import Table

from train_vision import SplitBrainVAE, setup_device
from vision import VisionSystem
from tracker import RamTracker

import subprocess

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


class TrainingUI:
    def __init__(self):
        self.console = Console()
        self.layout = Layout(name="root")
        self.log_messages = deque(maxlen=20)
        self.start_time = time.time()

        # Stats
        self.best_score = 0
        self.episodes_completed = 0
        self.total_steps = 0
        self.policy_updates = 0
        self.recent_scores = deque(maxlen=50)

        # Latency Metrics (ms)
        self.vae_latency = 0.0
        self.ppo_latency = 0.0
        self.env_latency = 0.0

        # Plotting Data Tracking
        self.all_scores = []
        self.all_rewards = []

        # Telemetry state
        self.current_ep = 0
        self.current_step = 0
        self.current_score = 0
        self.last_action = "Idle"
        self.current_reward = 0.0
        self.game_state = "UNKNOWN"
        self.ram_x = 0
        self.ram_z = 0
        self.fps = 0.0
        self.action_mask_status = ""

        self._setup_layout()

    def _setup_layout(self):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="side", size=40),
            Layout(name="body", ratio=2)
        )
        self.layout["side"].split(
            Layout(name="config", size=11),
            Layout(name="stats")
        )
        self.layout["body"].split(
            Layout(name="telemetry", size=15),
            Layout(name="log")
        )

        self.layout["header"].update(
            Panel(Align.center("[bold]CROSSY ROAD PPO AGENT[/bold]"), style="bold cyan", border_style="cyan"))
        self.layout["footer"].update(
            Panel(Align.center("[dim]Training in progress... Press CTRL+C to abort & save plots[/dim]"),
                  border_style="cyan"))
        self.update()
        self.episodes_completed = 0
        self.total_steps = 0
        self.policy_updates = 0
        self.recent_scores = deque(maxlen=50)

        # Plotting Data Tracking
        self.all_scores = []
        self.all_rewards = []

        # Telemetry state
        self.current_ep = 0
        self.current_step = 0
        self.current_score = 0
        self.last_action = "Idle"
        self.current_reward = 0.0
        self.game_state = "UNKNOWN"
        self.ram_x = 0
        self.ram_z = 0
        self.fps = 0.0
        self.action_mask_status = ""

        self._setup_layout()

    def _setup_layout(self):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(
            Layout(name="side", size=40),
            Layout(name="body", ratio=2)
        )
        self.layout["side"].split(
            Layout(name="config", size=11),
            Layout(name="stats")
        )
        self.layout["body"].split(
            Layout(name="telemetry", size=14),
            Layout(name="log")
        )

        self.layout["header"].update(
            Panel(Align.center("[bold]CROSSY ROAD PPO AGENT[/bold]"), style="bold cyan", border_style="cyan"))
        self.layout["footer"].update(
            Panel(Align.center("[dim]Training in progress... Press CTRL+C to abort & save plots[/dim]"),
                  border_style="cyan"))
        self.update()

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[[dim]{timestamp}[/dim]] {message}")
        # Intentionally NOT calling self.update() here to prevent burst-log flickering.
        # It will be naturally drawn on the next scheduled update.

    def get_config_table(self):
        table = Table(box=None, show_header=False, expand=True)
        table.add_column(style="bold dim", width=18)
        table.add_column(style="bright_white")
        table.add_row("Device (VAE):", str(VAE_DEVICE))
        table.add_row("Device (PPO):", str(PPO_DEVICE))
        table.add_row("Learning Rate:", str(LR))
        table.add_row("Gamma / GAE:", f"{GAMMA} / {GAE_LAMBDA}")
        table.add_row("Latent Dim:", str(LATENT_DIM))
        table.add_row("Hidden Dim:", str(HIDDEN_DIM))
        table.add_row("Update Step:", str(UPDATE_TIMESTEP))
        return table

    def get_stats_table(self):
        table = Table(box=None, show_header=False, expand=True)
        table.add_column(style="bold dim", width=18)
        table.add_column(style="bright_white")

        avg_score = sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0.0
        elapsed = time.time() - self.start_time
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)

        table.add_row("Runtime:", f"{int(hours):02d}h {int(mins):02d}m {int(secs):02d}s")
        table.add_row("Total Steps:", f"{self.total_steps:,}")
        table.add_row("Episodes:", f"{self.episodes_completed:,}")
        table.add_row("Policy Updates:", f"{self.policy_updates:,}")
        table.add_row("Best Score:", f"[bold green]{self.best_score}[/bold green]")
        table.add_row("Avg Score (50):", f"{avg_score:.1f}")
        return table

    def get_telemetry_table(self):
        table = Table(box=None, show_header=False, expand=True)
        table.add_column(style="bold dim", width=16)
        table.add_column()

        state_color = "green" if self.game_state == "PLAYING" else "red"

        # Standardization breakdown
        budget_used = self.vae_latency + self.ppo_latency + self.env_latency
        budget_color = "green" if budget_used < 100 else "red"  # 100ms is the 10fps budget

        table.add_row("Game State:", f"[{state_color}]{self.game_state}[/{state_color}]")
        table.add_row("Loop FPS:", f"[bold cyan]{self.fps:.1f}[/bold cyan] / 10.0")
        table.add_row("VAE Latency:", f"{self.vae_latency:.1f} ms")
        table.add_row("PPO Latency:", f"{self.ppo_latency:.1f} ms")
        table.add_row("Env Latency:", f"{self.env_latency:.1f} ms")
        table.add_row("Total Cycle:", f"[{budget_color}]{budget_used:.1f} ms[/ {budget_color}]")
        table.add_row("Z-Score:", f"[bold yellow]{self.current_score}[/bold yellow] (X: {self.ram_x:.1f})")
        table.add_row("Last Action:", f"[bold cyan]{self.last_action}[/bold cyan]")
        table.add_row("Ep Reward:", f"{self.current_reward:.2f}")

        return table

    def update(self):
        # We removed the throttle. To fix flickering, we ensure the Panel titles
        # and borders are consistent. The 'Live' context in train() will handle refresh.
        self.layout["config"].update(Panel(self.get_config_table(), title="[bold]Configuration", border_style="blue"))
        self.layout["stats"].update(Panel(self.get_stats_table(), title="[bold]Performance", border_style="magenta"))
        self.layout["telemetry"].update(
            Panel(self.get_telemetry_table(), title="[bold]Live Telemetry", border_style="green"))

        log_content = "\n".join(self.log_messages)
        self.layout["log"].update(Panel(log_content, title="[bold]Event Log", border_style="dim green"))


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
    def __init__(self, ui=None):
        self.ui = ui
        self.vae = SplitBrainVAE().to(VAE_DEVICE)
        self.vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=VAE_DEVICE, weights_only=False))
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

    def _log(self, msg):
        if self.ui:
            self.ui.log(msg)
        else:
            print(msg)

    def _ensure_game_running(self):
        self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
        if not self.hwnd:
            self._log("[RECOVERY] Game window not found. Launching Crossy Road...")
            try:
                # Launching WindowsApps sometimes requires 'start' shell command
                os.system(f'start "" "{EXECUTABLE_PATH}"')

                # Wait for load (WindowsApps can be slow)
                self._log("[RECOVERY] Waiting 9 seconds for startup...")
                time.sleep(9)

                self.hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
                if not self.hwnd:
                    self._log("[ERROR] Failed to find window after launch.")
                    return False

                # Focus and pass splash screen
                win32gui.SetForegroundWindow(self.hwnd)
                time.sleep(2)
                pydirectinput.press('space')
                self._log("[RECOVERY] Splash screen passed.")
                time.sleep(2)
            except Exception as e:
                self._log(f"[RECOVERY] Failed to launch: {e}")
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
        # 1. Get Visual Data (FLUSH QUEUE FOR LATEST FRAME)
        # Standardizing to 10 FPS requires grabbing the freshest capture.
        data = None
        try:
            # Drain the queue to ensure we aren't looking at a 100ms-old "buffered" frame
            while not self.vision_q.empty():
                data = self.vision_q.get_nowait()

            # If the queue was already drained, wait for the next 10 FPS heartbeat
            if data is None:
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

        vae_start = time.perf_counter()
        with torch.no_grad():
            # Original VAE returns: recon_static, pred_next, mu_c, log_c, mu_t, log_t
            _, _, mu_c, _, mu_t, _ = self.vae(tensor_in)
            # Concatenate Context (128) and Trend (128) = 256
            latents = torch.cat([mu_c, mu_t], dim=1).cpu().numpy().flatten()

        if self.ui:
            self.ui.vae_latency = (time.perf_counter() - vae_start) * 1000

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

        if self.ui:
            self.ui.ram_x = self.last_known_coords[0]
            self.ui.ram_z = current_score
            self.ui.game_state = "PLAYING" if data['pause_visible'] else "DEAD/MENU"

            mask_str = []
            if action_mask[2] < -1: mask_str.append("L")
            if action_mask[3] < -1: mask_str.append("R")
            self.ui.action_mask_status = f"Restricted: {','.join(mask_str)}" if mask_str else "Free"

        return state_vec, current_score, data['pause_visible'], action_mask

    def reset(self):
        # 1. Crash Check
        if not self._ensure_game_running():
            self._log("[CRITICAL] Could not recover game.")
            time.sleep(5)
            return self.reset()  # Recursive retry

        # 2. Focus and Click Retry
        try:
            win32gui.SetForegroundWindow(self.hwnd)
            geo = self.get_geometry()
            rx, ry = geo['left'] + RETRY_BUTTON_COORDS[0], geo['top'] + RETRY_BUTTON_COORDS[1]

            # Click Retry / Tap to Start
            pydirectinput.moveTo(rx, ry)
            pydirectinput.click()
            time.sleep(0.5)
            pydirectinput.press('space')
        except Exception as e:
            self._log(f"[RECOVERY] Window manipulation failed (Target died): {e}")
            time.sleep(2)
            # Window process might have crashed out, fall through back into recursive reset to restart the executable
            return self.reset()

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

        # SYNCED EXECUTION:
        # We removed the manual sleep. The loop now blocks on 'get_state()',
        # which effectively waits for the Vision thread's 10 FPS (100ms) heartbeat.
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
                self._log(f"[BONUS] Milestone reached! Z-Score: {score}")
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
import matplotlib.pyplot as plt


def save_training_plots(ui_data):
    """Generates and saves visual training plots on program shutdown."""
    if not ui_data.all_scores:
        print("\n[INFO] No episode data collected to plot.")
        return

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(12, 8))

    # Subplot 1: Scores with Moving Average
    plt.subplot(2, 1, 1)
    plt.plot(ui_data.all_scores, label="Max Z-Score", alpha=0.4, color='blue')
    window = min(10, len(ui_data.all_scores))
    if window > 0:
        ma = np.convolve(ui_data.all_scores, np.ones(window) / window, mode='valid')
        # Shift X to align properly with the moving average array size
        plt.plot(range(window - 1, len(ui_data.all_scores)), ma, label=f"{window}-Ep Moving Avg", color='red',
                 linewidth=2)
    plt.title("Agent Progression over Episodes")
    plt.ylabel("Z-Score (Distance)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Episode Rewards
    plt.subplot(2, 1, 2)
    plt.plot(ui_data.all_rewards, label="Total Ep Reward", color='green', alpha=0.7)
    plt.title("Cumulative Reward per Episode")
    plt.xlabel("Episode Index")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    filepath = f"logs/training_metrics_{timestamp}.png"
    plt.savefig(filepath)
    plt.close()
    print(f"\n[INFO] Successfully saved training plots to {filepath}")


def train():
    ui = TrainingUI()
    env = CrossyGameEnv(ui)

    # State Dim is now 257 (256 Latents + 1 Normalized X)
    ppo = PPO(257, 4)
    memory = Memory()

    start_episode = 1

    # Reduced refresh rate slightly to 5 fps to prevent Rich tearing
    with Live(ui.layout, console=ui.console, screen=True, refresh_per_second=5) as live:
        try:
            # --- AUTO RESUME LOGIC ---
            checkpoints = glob.glob("checkpoints/ppo_crossy_*.pth")
            if checkpoints:
                latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                ui.log(f"[RESUME] Loading checkpoint: {latest_cp}")
                ppo.policy.load_state_dict(torch.load(latest_cp, map_location=PPO_DEVICE, weights_only=False))
                ppo.policy_old.load_state_dict(ppo.policy.state_dict())
                start_episode = int(latest_cp.split('_')[-1].split('.')[0]) + 1
                ui.log(f"[RESUME] Resuming from Episode {start_episode}")
            else:
                ui.log("--- STARTING NEW TRAINING SESSION ---")

            ui.log("Focus the game window! Starting in 3 seconds...")
            time.sleep(3)

            time_step = 0

            for i_episode in range(start_episode, MAX_EPISODES + 1):
                ui.current_ep = i_episode
                state, action_mask = env.reset()
                current_ep_reward = 0

                # Safety Loop Break
                for t in range(1000):
                    step_start_time = time.perf_counter()
                    time_step += 1

                    # Select Action (Move state to CPU for PPO)
                    state_t = torch.FloatTensor(state).to(PPO_DEVICE)
                    mask_t = torch.FloatTensor(action_mask).to(PPO_DEVICE)

                    ppo_start = time.perf_counter()
                    action, logprob, value = ppo.policy_old.act(state_t, mask_t)
                    ui.ppo_latency = (time.perf_counter() - ppo_start) * 1000

                    # Record action in UI
                    action_map = {0: "Idle", 1: "Up", 2: "Left", 3: "Right"}
                    ui.last_action = action_map.get(action, "Unknown")

                    # Execute
                    env_start = time.perf_counter()
                    next_state, reward, done, next_action_mask = env.step(action)
                    ui.env_latency = (time.perf_counter() - env_start) * 1000

                    # Handle Step Failure
                    if next_state is None:
                        ui.log("[PPO] Step failed. Resetting...")
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

                    # Telemetry Update
                    ui.current_step = t
                    ui.current_reward = current_ep_reward
                    ui.total_steps += 1

                    # High-Precision FPS Calculation
                    step_time = time.perf_counter() - step_start_time
                    current_fps = 1.0 / step_time if step_time > 0 else 0.0
                    ui.fps = current_fps if ui.fps == 0.0 else (0.9 * ui.fps) + (0.1 * current_fps)

                    # Update UI data every step to ensure real-time log scrolling
                    ui.current_step = t
                    ui.current_reward = current_ep_reward
                    ui.total_steps += 1
                    ui.update()

                    # Update Policy
                    if time_step % UPDATE_TIMESTEP == 0:
                        ui.log(f"[PPO] Updating Policy at Step {time_step}...")
                        ppo.update(memory)
                        memory.clear()
                        time_step = 0
                        ui.policy_updates += 1
                        ui.log("[PPO] Policy Update Complete.")

                    if done:
                        break

                ui.episodes_completed += 1
                ui.recent_scores.append(env.last_score)
                ui.all_scores.append(env.last_score)
                ui.all_rewards.append(current_ep_reward)

                if env.last_score > ui.best_score:
                    ui.best_score = env.last_score
                    ui.log(f"[NEW BEST] Reached Z-Score {env.last_score}!")

                ui.log(f"Ep {i_episode} ended | Reward: {current_ep_reward:.2f} | Score: {env.last_score}")

                # Save Checkpoint
                if i_episode % 50 == 0:
                    torch.save(ppo.policy.state_dict(), f"checkpoints/ppo_crossy_{i_episode}.pth")
                    ui.log(f"[SAVE] Checkpoint saved for Ep {i_episode}")

        except KeyboardInterrupt:
            ui.log("[SHUTDOWN] Keyboard interrupt detected. Preparing to exit...")
            time.sleep(1)  # Give UI a moment to show the shutdown message before destroying Live context

    # Executed after the Live context has exited (Terminal returns to normal)
    save_training_plots(ui)


if __name__ == "__main__":
    train()
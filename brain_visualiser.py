# --- brain_visualizer.py ---
import cv2
import numpy as np
import torch
import time
import win32gui
import mss
from collections import deque
from train_vision import SplitBrainVAE, setup_device

# --- CONFIG ---
WINDOW_TITLE = "Crossy Road"
VAE_CHECKPOINT = "checkpoints/crossy_vae_ep500_KLD.pth"
IMG_SIZE = 160
STACK_SIZE = 4
LATENT_DIM = 64


def process_frame(frame):
    # MATCH THIS TO YOUR COLLECT_DATA SCRIPT!
    h, w, _ = frame.shape
    crop_h = int(h * 0.16) # The 16% Crop
    cropped = frame[crop_h:, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized / 255.0


def draw_bar_chart(activations, width=320, height=160):
    """ Draws a visualizer for the 64 latent neurons """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Split into Context (Top) and Trend (Bottom)
    ctx = activations[:32]
    trd = activations[32:]

    bar_w = width // 32

    # Draw Context (Cyan)
    for i, val in enumerate(ctx):
        # Normalize roughly for visualization (-3 to 3 range)
        h = int(np.clip(abs(val) * 20, 0, height // 2 - 2))
        color = (255, 255, 0)  # Cyan-ish
        x = i * bar_w
        # Draw from center line up
        cv2.rectangle(img, (x, height // 2 - h), (x + bar_w - 1, height // 2), color, -1)

    # Draw Trend (Magenta)
    for i, val in enumerate(trd):
        h = int(np.clip(abs(val) * 20, 0, height // 2 - 2))
        color = (255, 0, 255)  # Magenta
        x = i * bar_w
        # Draw from center line down
        cv2.rectangle(img, (x, height // 2), (x + bar_w - 1, height // 2 + h), color, -1)

    cv2.line(img, (0, height // 2), (width, height // 2), (100, 100, 100), 1)
    return img


def main():
    device = setup_device()
    print("Loading VAE...")
    vae = SplitBrainVAE().to(device)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=device))
    vae.eval()

    sct = mss.mss()
    hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
    if not hwnd:
        print("Game not found!")
        return

    frame_buffer = deque(maxlen=STACK_SIZE)

    # Pre-fill buffer
    print("Filling buffer...")
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    c_pt = win32gui.ClientToScreen(hwnd, (left, top))
    monitor = {"top": c_pt[1] + 50, "left": c_pt[0], "width": right - left, "height": bottom - top - 50}

    for _ in range(STACK_SIZE):
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        frame_buffer.append(process_frame(frame))

    print("--- NEURAL DASHBOARD ACTIVE ---")
    print("Top Left: Input | Top Right: Reconstruction")
    print("Bot Left: Prediction | Bot Right: Neural Activity")

    while True:
        # 1. Capture
        img = np.array(sct.grab(monitor))
        raw = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        processed = process_frame(raw)
        frame_buffer.append(processed)

        # 2. VAE Forward
        stack = np.array(frame_buffer, dtype=np.float32)
        tensor_in = torch.FloatTensor(stack).unsqueeze(0).to(device)

        with torch.no_grad():
            recon, pred, mu_c, _, mu_t, _ = vae(tensor_in)

        # 3. Process Outputs for Display
        # Input (Show the latest frame 't')
        show_in = (stack[-1] * 255).astype(np.uint8)
        show_in = cv2.cvtColor(show_in, cv2.COLOR_GRAY2BGR)

        # Recon (What Head A sees)
        show_recon = (recon[0].cpu().numpy().squeeze() * 255).astype(np.uint8)
        show_recon = cv2.cvtColor(show_recon, cv2.COLOR_GRAY2BGR)

        # Pred (What Head B predicts)
        show_pred = (pred[0].cpu().numpy().squeeze() * 255).astype(np.uint8)
        show_pred = cv2.cvtColor(show_pred, cv2.COLOR_GRAY2BGR)

        # Latents (Bar Chart)
        # Combine Context and Trend means
        latents = torch.cat([mu_c, mu_t], dim=1).cpu().numpy().flatten()
        show_chart = draw_bar_chart(latents, width=IMG_SIZE * 2, height=IMG_SIZE)

        # 4. Construct Dashboard Grid
        # Row 1: Input | Recon
        # Row 2: Pred  | Chart
        # Since Chart is double width, we might need to be creative or resize images
        # Let's resize images to be bigger for viewability

        DISP_SCALE = 2
        d_size = (IMG_SIZE * DISP_SCALE, IMG_SIZE * DISP_SCALE)

        img_1 = cv2.resize(show_in, d_size, interpolation=cv2.INTER_NEAREST)
        img_2 = cv2.resize(show_recon, d_size, interpolation=cv2.INTER_NEAREST)
        img_3 = cv2.resize(show_pred, d_size, interpolation=cv2.INTER_NEAREST)

        chart_resized = cv2.resize(show_chart, d_size, interpolation=cv2.INTER_NEAREST)

        # Add Labels
        cv2.putText(img_1, "INPUT (t)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_2, "RECON (t)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img_3, "PRED (t+1)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(chart_resized, "BRAIN ACTIVITY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Stack
        row1 = np.hstack([img_1, img_2])
        row2 = np.hstack([img_3, chart_resized])
        dashboard = np.vstack([row1, row2])

        cv2.imshow("Crossy Neural Dashboard", dashboard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
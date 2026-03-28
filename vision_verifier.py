import cv2
import torch
import numpy as np
import queue
import time
import os
from collections import deque

from train_vision import SplitBrainVAE, setup_device, IMG_SIZE, STACK_SIZE, LATENT_DIM
from vision import VisionSystem

# --- CONFIG ---
WINDOW_TITLE = "Crossy Road"
VAE_CHECKPOINT = "checkpoints/crossy_vae_best.pth"
DISPLAY_SCALE = 5  # Scales up the 160x160 images for easier viewing


def process_frame(frame):
    """Exact replication of the agent's visual preprocessing."""
    h, w, _ = frame.shape
    crop_h = int(h * 0.16)
    cropped = frame[crop_h:, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized / 255.0


def calculate_metrics(input_img, recon_img, mu_c, mu_t):
    """Calculates empirical metrics to detect VAE failure states."""
    # 1. Mean Squared Error
    mse = np.mean((input_img - recon_img) ** 2)

    # 2. Variance Preservation (Detects "Grey Mush" posterior collapse)
    in_var = np.var(input_img)
    recon_var = np.var(recon_img)
    var_ratio = (recon_var / (in_var + 1e-8)) * 100

    # 3. Latent Activity (Are the neurons actually firing?)
    # mu_c and mu_t are the means of the latent distributions.
    mu_c_act = np.mean(np.abs(mu_c))
    mu_t_act = np.mean(np.abs(mu_t))

    return mse, in_var, recon_var, var_ratio, mu_c_act, mu_t_act


def main():
    print("--- VAE Vision Verifier (256-Latent Version) ---")

    if not os.path.exists(VAE_CHECKPOINT):
        print(f"[ERROR] Could not find VAE checkpoint at {VAE_CHECKPOINT}")
        return

    # 1. Setup Device & Load Model
    device = setup_device()
    vae = SplitBrainVAE().to(device)

    try:
        vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=device))
        vae.eval()
        print(f"[SUCCESS] Loaded SplitBrainVAE Checkpoint: {VAE_CHECKPOINT}")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint. Architecture mismatch? Error: {e}")
        return

    # 2. Setup Vision Feed
    vision_q = queue.Queue(maxsize=1)
    vis = VisionSystem(WINDOW_TITLE, vision_q, show_preview=False)
    vis.start()

    frame_buffer = deque(maxlen=STACK_SIZE)
    frame_count = 0
    start_time = time.time()

    print("\n[INFO] Bring Crossy Road into focus. Press 'q' on the OpenCV window to exit.\n")

    try:
        while True:
            try:
                data = vision_q.get(timeout=1.0)
            except queue.Empty:
                continue

            # Process exactly as the agent does
            processed = process_frame(data['frame'])
            frame_buffer.append(processed)

            if len(frame_buffer) < STACK_SIZE:
                while len(frame_buffer) < STACK_SIZE:
                    frame_buffer.append(processed)

            # Build Tensor
            stack = np.array(frame_buffer, dtype=np.float32)
            tensor_in = torch.FloatTensor(stack).unsqueeze(0).to(device)

            # --- FORWARD PASS ---
            with torch.no_grad():
                # Reverted VAE returns 6 values: recon, pred, mu_c, log_c, mu_t, log_t
                recon, pred, mu_c, _, mu_t, _ = vae(tensor_in)

            # Extract images to CPU numpy (Convert back to 0-255 uint8)
            img_input = stack[3]  # The current reference frame
            img_recon = recon[0].cpu().numpy().squeeze()
            img_pred = pred[0].cpu().numpy().squeeze()

            # --- CALCULATE METRICS ---
            mse, in_var, recon_var, var_ratio, act_c, act_t = calculate_metrics(
                img_input, img_recon,
                mu_c.cpu().numpy(), mu_t.cpu().numpy()
            )

            # --- VISUALIZATION HUD ---
            # Create an Error Heatmap (Absolute difference between input and recon)
            error_diff = np.abs(img_input - img_recon)
            error_heatmap = cv2.applyColorMap((error_diff * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Convert grayscale to BGR for concatenation with heatmap
            vis_input = cv2.cvtColor((img_input * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis_recon = cv2.cvtColor((img_recon * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vis_pred = cv2.cvtColor((img_pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Add Text Labels
            cv2.putText(vis_input, "1. Input (What Agent Sees)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(vis_recon, "2. Recon (Context Brain)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100),
                        1)
            cv2.putText(vis_pred, "3. Trend (Next Frame Prediction)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (100, 255, 100), 1)
            cv2.putText(error_heatmap, "4. Error Map (Red = Bad)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)

            # Assemble Grid
            top_row = cv2.hconcat([vis_input, vis_recon])
            bottom_row = cv2.hconcat([vis_pred, error_heatmap])
            grid = cv2.vconcat([top_row, bottom_row])

            # Scale up for modern monitors
            h, w = grid.shape[:2]
            grid_scaled = cv2.resize(grid, (w * DISPLAY_SCALE, h * DISPLAY_SCALE), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("SplitBrainVAE Diagnostic", grid_scaled)

            # --- CONSOLE TELEMETRY (Print every 15 frames) ---
            frame_count += 1
            if frame_count % 15 == 0:
                fps = 15 / (time.time() - start_time)

                # ANSI Escape codes to clear the line and print fresh
                print(f"\r[FPS: {fps:.1f}] "
                      f"MSE: {mse:.4f} | "
                      f"VarPreserve: {var_ratio:>5.1f}% | "
                      f"Latent Firing: (Ctx:{act_c:.3f}, Trnd:{act_t:.3f})    ", end="")

                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n[INFO] Shutting down...")
        vis.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
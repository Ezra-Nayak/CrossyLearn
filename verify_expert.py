import torch
import glob
import cv2
import numpy as np
import time
import os
import shutil

from train_vision import SpatialVQVAE, setup_device

# --- CONFIG ---
VAE_CHECKPOINT = "checkpoints/crossy_vae_best.pth"
DATA_DIR = "expert_data"
DISPLAY_SCALE = 6


def main():
    print("--- EXPERT DATA VISUAL VERIFIER ---")
    device = setup_device()

    # Load the VAE to decode the Agent's "Memories"
    vae = SpatialVQVAE().to(device)
    try:
        vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=device, weights_only=False))
        vae.eval()
        print(f"[SUCCESS] Loaded VAE from {VAE_CHECKPOINT}")
    except Exception as e:
        print(f"[ERROR] Could not load VAE. Make sure path is correct. {e}")
        return

    pass_dir = os.path.join(DATA_DIR, "pass")
    fail_dir = os.path.join(DATA_DIR, "fail")
    os.makedirs(pass_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    # Filter out files that have already been processed
    all_files = glob.glob(f"{DATA_DIR}/*.pt")
    files = [f for f in all_files if "pass" not in f and "fail" not in f]

    if not files:
        print(f"[INFO] No pending expert data found in {DATA_DIR}. Check subfolders.")
        return

    action_map = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "IDLE"}

    for file_idx, file in enumerate(files):
        trajectory = torch.load(file, weights_only=False)
        filename = os.path.basename(file)

        # --- AUTO ANALYZER ---
        idle_count = sum(1 for step in trajectory if step['action'] == 3)
        idle_pct = idle_count / len(trajectory) if trajectory else 0
        eagle_warning = idle_pct > 0.65

        frame_idx = 0
        paused = False
        decision_made = False

        while not decision_made:
            # Handle End of Video
            if frame_idx >= len(trajectory):
                frame_idx = len(trajectory) - 1
                paused = True
            elif frame_idx < 0:
                frame_idx = 0

            step = trajectory[frame_idx]
            latents = torch.FloatTensor(step['latents']).to(device)
            scalar = step['scalars'][0]
            mask = step['mask']
            action = step['action']

            z_c, z_t = torch.split(latents, 64, dim=0)

            with torch.no_grad():
                recon = vae.decoder(z_c.unsqueeze(0))

            img = recon[0].cpu().numpy().squeeze()
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            h, w = img_bgr.shape[:2]
            display = cv2.resize(img_bgr, (w * DISPLAY_SCALE, h * DISPLAY_SCALE), interpolation=cv2.INTER_NEAREST)

            # --- DRAW VIDEO HUD ---
            act_str = action_map.get(action, "UNKNOWN")
            color = (0, 255, 0) if act_str != "IDLE" else (150, 150, 150)

            cv2.putText(display, f"Action: {act_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"RAM X: {scalar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            mask_status = []
            if mask[1] < -1: mask_status.append("LEFT BLOCKED")
            if mask[2] < -1: mask_status.append("RIGHT BLOCKED")
            if mask_status:
                cv2.putText(display, " | ".join(mask_status), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # --- DRAW CONTROL PANEL ---
            panel_height = 160
            canvas = np.zeros((h * DISPLAY_SCALE + panel_height, w * DISPLAY_SCALE, 3), dtype=np.uint8)
            canvas[:h * DISPLAY_SCALE, :] = display

            panel_y = h * DISPLAY_SCALE
            cv2.rectangle(canvas, (0, panel_y), (w * DISPLAY_SCALE, panel_y + panel_height), (30, 30, 30), -1)
            cv2.line(canvas, (0, panel_y), (w * DISPLAY_SCALE, panel_y), (255, 255, 255), 2)

            # Progress Bar
            progress = frame_idx / max(1, len(trajectory) - 1)
            bar_w = int(progress * (w * DISPLAY_SCALE - 40))
            cv2.rectangle(canvas, (20, panel_y + 15), (w * DISPLAY_SCALE - 20, panel_y + 25), (60, 60, 60), -1)

            bar_color = (0, 50, 255) if eagle_warning else (0, 200, 100)
            cv2.rectangle(canvas, (20, panel_y + 15), (20 + bar_w, panel_y + 25), bar_color, -1)

            # Text Info
            cv2.putText(canvas, f"File: {filename} ({file_idx + 1}/{len(files)})", (20, panel_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(canvas, f"Frame: {frame_idx}/{len(trajectory) - 1}", (20, panel_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            idle_color = (0, 0, 255) if eagle_warning else (200, 200, 200)
            cv2.putText(canvas, f"Idle Rate: {idle_pct * 100:.1f}%", (20, panel_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        idle_color, 1)

            if eagle_warning:
                cv2.putText(canvas, "WARNING: EAGLE BAIT / LOOP DETECTED", (20, panel_y + 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Controls Help
            controls_x = w * DISPLAY_SCALE - 300
            cv2.putText(canvas, "[SPACE] Play/Pause | [A/D] Scrub", (controls_x, panel_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(canvas, "[Y] Pass | [N] Fail", (controls_x, panel_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(canvas, "[T] Trim (Keep up to this frame)", (controls_x, panel_y + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
            cv2.putText(canvas, "[S] Skip | [Q] Quit Viewer", (controls_x, panel_y + 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1)

            if paused:
                cv2.putText(canvas, "PAUSED", (w * DISPLAY_SCALE // 2 - 50, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

            cv2.imshow("Triage Control Center", canvas)

            # --- INPUT HANDLING ---
            key = cv2.waitKey(0 if paused else 100) & 0xFF

            if key == ord(' '):
                paused = not paused
            elif key == ord('d'):
                frame_idx += 1
                paused = True
            elif key == ord('a'):
                frame_idx -= 1
                paused = True
            elif key == ord('y'):
                shutil.move(file, os.path.join(pass_dir, filename))
                print(f"[PASSED] {filename}")
                decision_made = True
            elif key == ord('n'):
                shutil.move(file, os.path.join(fail_dir, filename))
                print(f"[FAILED] {filename}")
                decision_made = True
            elif key == ord('t'):
                # Save only the frames up to the current frame index
                trimmed_trajectory = trajectory[:frame_idx + 1]
                save_path = os.path.join(pass_dir, filename)
                torch.save(trimmed_trajectory, save_path)
                os.remove(file)  # Delete original to prevent duplicates
                print(f"[TRIMMED & PASSED] {filename} (Kept {len(trimmed_trajectory)} frames)")
                decision_made = True
            elif key == ord('s'):
                print(f"[SKIPPED] {filename}")
                decision_made = True
            elif key == ord('q'):
                print("[STOP] Triage aborted by user.")
                cv2.destroyAllWindows()
                return

            if not paused and not decision_made:
                frame_idx += 1

    cv2.destroyAllWindows()
    print("\n[FINISHED] All pending expert runs have been triaged.")


if __name__ == "__main__":
    main()
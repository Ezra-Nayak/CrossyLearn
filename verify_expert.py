import torch
import glob
import cv2
import numpy as np
import time

from train_vision import SpatialVQVAE, setup_device

# --- CONFIG ---
VAE_CHECKPOINT = "checkpoints/crossy_vae_best.pth"
DATA_DIR = "expert_data"
DISPLAY_SCALE = 4


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

    files = glob.glob(f"{DATA_DIR}/*.pt")
    if not files:
        print("[ERROR] No expert data found.")
        return

    action_map = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "IDLE"}

    for file in files:
        trajectory = torch.load(file, weights_only=False)
        print(f"\n[PLAYING] {file} ({len(trajectory)} frames)")

        for i, step in enumerate(trajectory):
            latents = torch.FloatTensor(step['latents']).to(device)  # [128, 20, 20]
            scalar = step['scalars'][0]
            mask = step['mask']
            action = step['action']

            # Split the 128 channels back into Context (64) and Trend (64)
            z_c, z_t = torch.split(latents, 64, dim=0)

            # Decode the Context Latents back into an image
            with torch.no_grad():
                # Add batch dimension:[1, 64, 20, 20]
                recon = vae.decoder(z_c.unsqueeze(0))

            # Convert to OpenCV format (0-255 grayscale)
            img = recon[0].cpu().numpy().squeeze()
            img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Scale up for easy viewing
            h, w = img_bgr.shape[:2]
            display = cv2.resize(img_bgr, (w * DISPLAY_SCALE, h * DISPLAY_SCALE), interpolation=cv2.INTER_NEAREST)

            # Draw HUD
            act_str = action_map.get(action, "UNKNOWN")
            color = (0, 255, 0) if act_str != "IDLE" else (150, 150, 150)

            cv2.putText(display, f"Action: {act_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"RAM X: {scalar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            mask_status = []
            if mask[1] < -1: mask_status.append("LEFT BLOCKED")
            if mask[2] < -1: mask_status.append("RIGHT BLOCKED")
            if mask_status:
                cv2.putText(display, " | ".join(mask_status), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display frame
            cv2.imshow("Agent Memory Playback", display)

            # Wait 100ms to simulate the 10 FPS runtime speed
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("[STOP] Playback aborted by user.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n[FINISHED] All expert runs played back.")


if __name__ == "__main__":
    main()
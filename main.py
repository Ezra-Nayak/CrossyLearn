import cv2
import numpy as np
import mss
import time
import win32gui
import win32ui
import win32con

def main():
    """
    Main function to capture and display the Crossy Road game window.
    """
    WINDOW_TITLE = "Crossy Road"
    TARGET_FPS = 60
    FRAME_DELAY = 1.0 / TARGET_FPS

    print("CrossyLearn Agent - Milestone 2: Refined Vision System")
    print("------------------------------------------------------")
    print(f"Attempting to find window: '{WINDOW_TITLE}'")

    with mss.mss() as sct:
        while True:
            loop_start_time = time.time()

            # Find the game window handle (hwnd)
            hwnd = win32gui.FindWindow(None, WINDOW_TITLE)
            if not hwnd:
                print(f"'{WINDOW_TITLE}' window not found. Please ensure the game is running. Retrying...")
                time.sleep(2)
                continue

            try:
                # Get the client area rectangle
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                # Convert client area coords to screen coords
                client_left, client_top = win32gui.ClientToScreen(hwnd, (left, top))
                client_right, client_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))

                # Define the capture region
                monitor = {
                    "top": client_top,
                    "left": client_left,
                    "width": client_right - client_left,
                    "height": client_bottom - client_top
                }

                # Grab the data
                img_bgra = sct.grab(monitor)

                # Convert to a format OpenCV can use (BGRA -> BGR)
                img_bgr = np.array(img_bgra)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

                # Display the resulting frame
                cv2.imshow('CrossyLearn Vision', img_rgb)

                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Precise frame rate limiting
                elapsed_time = time.time() - loop_start_time
                sleep_duration = FRAME_DELAY - elapsed_time
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            except win32ui.error as e:
                # This can happen if the window is closed between FindWindow and GetClientRect
                print(f"Window handle error: {e}. Retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

    print("Shutting down vision system.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
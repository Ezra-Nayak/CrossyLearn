import cv2
import numpy as np
import mss
import pygetwindow as gw
import time

def main():
    """
    Main function to capture and display the Crossy Road game window.
    """
    WINDOW_TITLE = "Crossy Road"
    TARGET_FPS = 60
    FRAME_DELAY = 1.0 / TARGET_FPS

    print("CrossyLearn Agent - Milestone 1: Vision System")
    print("---------------------------------------------")
    print(f"Attempting to find window: '{WINDOW_TITLE}'")

    with mss.mss() as sct:
        while True:
            try:
                # Find the game window
                game_window = gw.getWindowsWithTitle(WINDOW_TITLE)[0]
                if not game_window:
                    print(f"'{WINDOW_TITLE}' window not found. Please ensure the game is running. Retrying...")
                    time.sleep(2)
                    continue

                # Define the capture region based on the window's geometry
                monitor = {
                    "top": game_window.top,
                    "left": game_window.left,
                    "width": game_window.width,
                    "height": game_window.height
                }

                # Grab the data
                img_bgra = sct.grab(monitor)

                # Convert to a format OpenCV can use (BGRA -> BGR)
                img_bgr = np.array(img_bgra)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

                # Display the resulting frame
                cv2.imshow('CrossyLearn Vision', img_rgb)

                # Frame rate limiting
                time.sleep(FRAME_DELAY)

                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except IndexError:
                print(f"'{WINDOW_TITLE}' window not found. Please ensure the game is running. Retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

    print("Shutting down vision system.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
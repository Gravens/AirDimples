import finger_detection
import pyautogui
import keyboard


def move_cursor_on_screen(normalized_finger_coords):
    window_size = pyautogui.size()

    # Convert normalized coordinates to correspond to the screen
    screen_finger_coords = {}
    for finger, coords in normalized_finger_coords.items():
        screen_finger_coords[finger] = finger_detection.normalized_to_pixel_coordinates((coords.x, coords.y), window_size)

    print(screen_finger_coords)

    # Failsafe
    if keyboard.is_pressed('ctrl'):
        return

    # Move the cursor
    pyautogui.moveTo(screen_finger_coords['index'])


if __name__ == '__main__':
    pyautogui.FAILSAFE = False
    finger_detection.launch_detection_on_webcam()


import pyautogui
import keyboard

from utils import denormalize_coordinates, log

pyautogui.FAILSAFE = False


def move_cursor_on_screen(normalized_finger_coords):
    window_size = pyautogui.size()

    # Denormalize coordinates to correspond to the screen
    screen_finger_coords = {}
    for finger, coords in normalized_finger_coords.items():
        screen_finger_coords[finger] = denormalize_coordinates((coords.x, coords.y), window_size)

    log.debug(screen_finger_coords)

    # Failsafe
    if keyboard.is_pressed('ctrl'):
        return

    # Move the cursor
    pyautogui.moveTo(screen_finger_coords['index'])

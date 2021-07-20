import keyboard
import autopy

from utils import denormalize_coordinates, distance, up_status, log


def process(normalized_finger_coords,
            last_screen_x,
            last_screen_y,
            smooth_ratio):
    window_size = autopy.screen.size()

    # Denormalize coordinates to correspond to the screen
    screen_finger_coords = {}
    for finger, coords in normalized_finger_coords.items():
        screen_finger_coords[finger] = denormalize_coordinates((coords.x, coords.y), window_size)

    # log.debug(screen_finger_coords)

    # Failsafe
    if keyboard.is_pressed('ctrl'):
        return

    curr_screen_x = last_screen_x + (screen_finger_coords['INDEX_FINGER_TIP'][0] - last_screen_x) / smooth_ratio
    curr_screen_y = last_screen_y + (screen_finger_coords['INDEX_FINGER_TIP'][1] - last_screen_y) / smooth_ratio

    fingers_up_status = {'thumb': up_status(screen_finger_coords['THUMB_TIP'],
                                            screen_finger_coords['THUMB_IP'], thumb=True),
                         'index': up_status(screen_finger_coords['INDEX_FINGER_TIP'],
                                            screen_finger_coords['INDEX_FINGER_MCP']),
                         'middle': up_status(screen_finger_coords['MIDDLE_FINGER_TIP'],
                                             screen_finger_coords['MIDDLE_FINGER_MCP']),
                         'ring': up_status(screen_finger_coords['RING_FINGER_TIP'],
                                           screen_finger_coords['RING_FINGER_MCP']),
                         'pinky': up_status(screen_finger_coords['PINKY_TIP'],
                                            screen_finger_coords['PINKY_MCP'])}

    if fingers_up_status['index'] and not fingers_up_status['middle']:
        autopy.mouse.move(curr_screen_x, curr_screen_y)
        return curr_screen_x, curr_screen_y, False, False

    if fingers_up_status['index'] and fingers_up_status['middle']:
        if distance(screen_finger_coords['INDEX_FINGER_TIP'], screen_finger_coords['MIDDLE_FINGER_TIP']) < 45:
            autopy.mouse.click()
            return curr_screen_x, curr_screen_y, True, True
        return curr_screen_x, curr_screen_y, True, False

    return curr_screen_x, curr_screen_y, False, False

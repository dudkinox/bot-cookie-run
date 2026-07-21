import cv2
import numpy as np
import pyautogui
import time
import keyboard

# 안전장치: 마우스를 화면 구석으로 이동하면 pyautogui가 예외를 발생시켜 중지시킬 수 있음
pyautogui.FAILSAFE = True

points = []  # list of (x, y) screen coordinates

# Capture current screen to allow user to pick points
screenshot = pyautogui.screenshot()
img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
clone = img.copy()

window_name = 'Select points - Left click: add, Right click: remove last, s: start clicking, Esc: exit'

def draw_points(img, pts):
    disp = img.copy()
    for i, (x, y) in enumerate(pts, start=1):
        cv2.circle(disp, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(disp, str(i), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return disp


def mouse_callback(event, x, y, flags, param):
    global points, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point (screen coordinates already)
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove last point
        if points:
            points.pop()


cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, mouse_callback)

print(window_name)

while True:
    disp = draw_points(clone, points)
    cv2.imshow(window_name, disp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc
        print('Exiting without clicking.')
        cv2.destroyAllWindows()
        raise SystemExit(0)
    if key == ord('s'):
        break

cv2.destroyAllWindows()

if not points:
    print('No points selected. Exiting.')
    raise SystemExit(0)

# Ask for click interval
try:
    interval_input = input('Enter click interval in seconds (default 0.5): ').strip()
    interval = float(interval_input) if interval_input else 0.5
except Exception:
    interval = 0.5

print('Starting click loop. Press ESC to stop.')

try:
    while True:
        if keyboard.is_pressed('esc'):
            print('ESC detected — stopping.')
            break
        for (x, y) in points:
            # Double-check ESC between clicks
            if keyboard.is_pressed('esc'):
                break
            pyautogui.click(x, y)
            time.sleep(interval)
except KeyboardInterrupt:
    print('Interrupted by user.')

print('Done.')

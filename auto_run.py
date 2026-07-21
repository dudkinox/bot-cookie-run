"""Detect Cookie Run's home screen and press S to start.

On the first run, show a screenshot and let the user select the game area.
The selected region and a Play-button template are saved beside this script.
Run with ``--reset`` to select the game area again.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import keyboard
import numpy as np
import pyautogui


APP_DIR = Path(__file__).resolve().parent
CONFIG_FILE = APP_DIR / "auto_run_config.json"
PLAY_TEMPLATE_FILE = APP_DIR / "play_button.png"

# Position of the Play button in the supplied 1065x599 reference image.
# A little padding is included so template matching also sees its border.
PLAY_BOX = (0.585, 0.817, 0.915, 0.960)  # left, top, right, bottom
MATCH_THRESHOLD = 0.72
CHECK_INTERVAL = 0.10
CLICK_COOLDOWN = 3.0
GAME_CLICK_INTERVAL = 0.5
ESC_HOLD_SECONDS = 1.0

pyautogui.FAILSAFE = True


def screenshot_bgr(region: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Capture the whole screen or (left, top, width, height) as BGR."""
    shot = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)


def select_game_region() -> tuple[int, int, int, int]:
    """Let the user select the game rectangle from a current screenshot."""
    screen = screenshot_bgr()
    title = "Select Cookie Run window, then press ENTER (C = cancel)"
    x, y, width, height = map(
        int, cv2.selectROI(title, screen, showCrosshair=True, fromCenter=False)
    )
    cv2.destroyAllWindows()
    if width <= 0 or height <= 0:
        raise SystemExit("Selection cancelled.")
    return x, y, width, height


def play_box_pixels(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Convert the relative Play box into pixel coordinates."""
    height, width = frame.shape[:2]
    left, top, right, bottom = PLAY_BOX
    return (
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    )


def save_setup(region: tuple[int, int, int, int]) -> None:
    """Save the selected region and crop the Play template from it."""
    frame = screenshot_bgr(region)
    x1, y1, x2, y2 = play_box_pixels(frame)
    template = frame[y1:y2, x1:x2]
    if template.size == 0 or not cv2.imwrite(str(PLAY_TEMPLATE_FILE), template):
        raise RuntimeError("Could not save the Play-button template.")

    CONFIG_FILE.write_text(
        json.dumps({"region": list(region)}, indent=2), encoding="utf-8"
    )
    print(f"Saved game region: {region}")
    print(f"Saved Play template: {PLAY_TEMPLATE_FILE.name}")


def load_region() -> tuple[int, int, int, int]:
    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    region = tuple(map(int, data["region"]))
    if len(region) != 4 or region[2] <= 0 or region[3] <= 0:
        raise ValueError("Invalid saved game region")
    return region  # type: ignore[return-value]


def find_green_play(frame: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Find a wide green Play button near the bottom of either game screen."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Both Play buttons are bright yellow-green. Closing joins their gradient
    # bands and the white text into one stable, wide contour.
    mask = cv2.inRange(hsv, np.array([32, 75, 65]), np.array([92, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = cv2.contourArea(contour) / float(width * height)
        aspect = w / max(h, 1)
        if (
            y > height * 0.65
            and w > width * 0.25
            and h > height * 0.06
            and aspect > 2.5
            and area_ratio > 0.012
        ):
            # Score is only used for preview/debugging; valid geometry is the
            # important part of this detector.
            score = min(1.0, 0.75 + area_ratio * 4.0)
            candidates.append((score, (x + w // 2, y + h // 2)))

    return max(candidates, default=(0.0, (0, 0)), key=lambda item: item[0])


def has_pause_icon(frame: np.ndarray) -> bool:
    """Detect the round grey Pause control at the top-right during gameplay."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Keep bright, nearly grey UI pixels. The home-screen gear is cyan and is
    # therefore excluded by the low-saturation limit.
    grey = cv2.inRange(hsv, np.array([0, 0, 105]), np.array([179, 72, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        area = cv2.contourArea(contour)
        fill_ratio = area / max(float(w * h), 1.0)
        if (
            x > width * 0.88
            and y < height * 0.12
            and width * 0.028 < w < width * 0.080
            and height * 0.045 < h < height * 0.130
            and 0.72 < aspect < 1.30
            and fill_ratio > 0.48
        ):
            return True
    return False


def has_fast_start_icon(frame: np.ndarray) -> bool:
    """Detect the square green Fast Start icon near the screen centre."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([30, 55, 65]), np.array([95, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centre_x = x + w / 2
        centre_y = y + h / 2
        aspect = w / max(h, 1)
        if (
            width * 0.38 < centre_x < width * 0.62
            and height * 0.32 < centre_y < height * 0.64
            and width * 0.075 < w < width * 0.22
            and height * 0.11 < h < height * 0.31
            and 0.68 < aspect < 1.38
            and cv2.contourArea(contour) > width * height * 0.005
        ):
            return True
    return False


def has_mystery_box_icon(frame: np.ndarray) -> bool:
    """Detect the brown square question box shown near the screen centre."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, np.array([4, 75, 45]), np.array([30, 255, 255]))
    light_mark = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([35, 145, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    brown = cv2.morphologyEx(brown, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centre_x = x + w / 2
        centre_y = y + h / 2
        aspect = w / max(h, 1)
        fill_ratio = cv2.contourArea(contour) / max(float(w * h), 1.0)
        mark_ratio = cv2.countNonZero(light_mark[y:y + h, x:x + w]) / max(float(w * h), 1.0)
        if (
            width * 0.30 < centre_x < width * 0.70
            and height * 0.18 < centre_y < height * 0.76
            and width * 0.035 < w < width * 0.20
            and height * 0.055 < h < height * 0.30
            and 0.68 < aspect < 1.38
            and fill_ratio > 0.48
            and mark_ratio > 0.012
        ):
            return True
    return False


def find_result_ok(frame: np.ndarray) -> tuple[bool, tuple[int, int]]:
    """Detect the bright Result dialog and return its green OK button centre."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Result is the only handled screen dominated by the large pale dialog.
    pale_dialog = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([179, 80, 255]))
    pale_ratio = cv2.countNonZero(pale_dialog) / float(width * height)
    if pale_ratio < 0.48:
        return False, (0, 0)

    green = cv2.inRange(hsv, np.array([32, 75, 65]), np.array([92, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centre_x = x + w / 2
        centre_y = y + h / 2
        aspect = w / max(h, 1)
        if (
            width * 0.20 < centre_x < width * 0.48
            and centre_y > height * 0.72
            and width * 0.18 < w < width * 0.34
            and height * 0.07 < h < height * 0.20
            and aspect > 2.4
        ):
            candidates.append((cv2.contourArea(contour), (round(centre_x), round(centre_y))))

    if not candidates:
        return False, (0, 0)
    _, centre = max(candidates, key=lambda item: item[0])
    return True, centre


def find_open_all(frame: np.ndarray) -> tuple[bool, tuple[int, int]]:
    """Detect the Mystery Box screen and return the Open all button centre."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mystery Box has a predominantly dark blue background.
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 105]))
    dark_ratio = cv2.countNonZero(dark) / float(width * height)
    if dark_ratio < 0.42:
        return False, (0, 0)

    cyan = cv2.inRange(hsv, np.array([78, 80, 65]), np.array([105, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9))
    cyan = cv2.morphologyEx(cyan, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, tuple[int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        centre_x = x + w / 2
        centre_y = y + h / 2
        aspect = w / max(h, 1)
        if (
            width * 0.38 < centre_x < width * 0.62
            and centre_y > height * 0.76
            and width * 0.17 < w < width * 0.34
            and height * 0.07 < h < height * 0.20
            and aspect > 2.4
        ):
            candidates.append((cv2.contourArea(contour), (round(centre_x), round(centre_y))))

    if not candidates:
        return False, (0, 0)
    _, centre = max(candidates, key=lambda item: item[0])
    return True, centre


def find_play(frame: np.ndarray, template: np.ndarray) -> tuple[float, tuple[int, int], str]:
    """Return confidence, Play centre, and the detector that found it."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if (
        gray_template.shape[0] > gray_frame.shape[0]
        or gray_template.shape[1] > gray_frame.shape[1]
    ):
        green_score, green_centre = find_green_play(frame)
        return green_score, green_centre, "green button"

    result = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    centre = (
        location[0] + gray_template.shape[1] // 2,
        location[1] + gray_template.shape[0] // 2,
    )
    if score >= MATCH_THRESHOLD:
        return float(score), centre, "home template"

    green_score, green_centre = find_green_play(frame)
    return green_score, green_centre, "green button"


def run(region: tuple[int, int, int, int], show_preview: bool = False) -> None:
    template = cv2.imread(str(PLAY_TEMPLATE_FILE), cv2.IMREAD_COLOR)
    if template is None:
        raise SystemExit("Play template is missing. Run: python auto_run.py --reset")

    print("Watching for the Play screen. Hold ESC for 1 second to stop.")
    last_click = 0.0
    game_clicking = False
    mystery_opened = False
    fast_start_active = False
    esc_started: float | None = None
    while True:
        if keyboard.is_pressed("esc"):
            if esc_started is None:
                esc_started = time.monotonic()
            elif time.monotonic() - esc_started >= ESC_HOLD_SECONDS:
                print("ESC held for 1 second; stopping.")
                break
        else:
            esc_started = None

        frame = screenshot_bgr(region)
        score, centre, detector = find_play(frame, template)
        result_found, ok_centre = find_result_ok(frame)
        open_all_found, open_all_centre = find_open_all(frame)
        pause_found = has_pause_icon(frame)
        fast_start_found = has_fast_start_icon(frame)
        mystery_box_found = has_mystery_box_icon(frame)
        now = time.monotonic()

        if result_found:
            if game_clicking:
                print("Result screen found; stopping W-key mode.")
            game_clicking = False
            mystery_opened = False
            fast_start_active = False
            if now - last_click >= CLICK_COOLDOWN:
                screen_x = region[0] + ok_centre[0]
                screen_y = region[1] + ok_centre[1]
                print(f"Clicking Result OK at {screen_x}, {screen_y}")
                pyautogui.click(screen_x, screen_y)
                last_click = now
        elif open_all_found:
            game_clicking = False
            if now - last_click >= CLICK_COOLDOWN:
                screen_x = region[0] + open_all_centre[0]
                screen_y = region[1] + open_all_centre[1]
                action_name = "Confirm" if mystery_opened else "Open all"
                print(f"Clicking {action_name} at {screen_x}, {screen_y}")
                pyautogui.click(screen_x, screen_y)
                last_click = now
                mystery_opened = True
        else:
            if mystery_box_found:
                if game_clicking:
                    print("Mystery Box icon found; stopping W-key mode.")
                game_clicking = False
                fast_start_active = False
            elif fast_start_found:
                game_clicking = False
                if not fast_start_active:
                    screen_x = region[0] + region[2] // 2
                    screen_y = region[1] + region[3] // 2
                    print(f"Fast Start found; clicking centre at {screen_x}, {screen_y}")
                    pyautogui.click(screen_x, screen_y)
                    last_click = now
                fast_start_active = True
            elif pause_found and not game_clicking:
                fast_start_active = False
                game_clicking = True
                last_click = 0.0
                print("Pause icon found; pressing W every 0.5 seconds.")
            elif not pause_found and game_clicking:
                fast_start_active = False
                game_clicking = False
                print("Pause icon disappeared; stopping W-key mode.")
            elif not fast_start_found:
                fast_start_active = False

            if game_clicking and now - last_click >= GAME_CLICK_INTERVAL:
                pyautogui.press("w")
                last_click = now
            elif score >= MATCH_THRESHOLD and now - last_click >= CLICK_COOLDOWN:
                print(
                    f"Play found by {detector} ({score:.2f}); "
                    "pressing S"
                )
                pyautogui.press("s")
                last_click = now

        if show_preview:
            preview = frame.copy()
            color = (0, 255, 0) if (game_clicking or score >= MATCH_THRESHOLD) else (0, 0, 255)
            if result_found:
                mode = "RESULT: clicking OK"
            elif open_all_found:
                action_name = "Confirm" if mystery_opened else "Open all"
                mode = f"MYSTERY BOX: clicking {action_name}"
            elif mystery_box_found:
                mode = "MYSTERY BOX ICON: W stopped"
            elif fast_start_found:
                mode = "FAST START: clicking centre"
            elif game_clicking:
                mode = "PRESS W 0.5s"
            else:
                mode = f"Play match: {score:.2f}"
            cv2.putText(
                preview, mode, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
            cv2.imshow("auto_run preview - Q to stop", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(CHECK_INTERVAL)

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="select the game area again")
    parser.add_argument("--preview", action="store_true", help="show detection score")
    args = parser.parse_args()

    if args.reset or not CONFIG_FILE.exists() or not PLAY_TEMPLATE_FILE.exists():
        print("Select the full game area while the supplied Play screen is visible.")
        region = select_game_region()
        save_setup(region)
    else:
        region = load_region()

    try:
        run(region, show_preview=args.preview)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

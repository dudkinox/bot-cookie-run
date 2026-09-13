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
CONFIG_FILE = APP_DIR / "auto_run_box_config.json"
PLAY_TEMPLATE_FILE = APP_DIR / "play_button.png"

# Position of the Play button in the supplied 1065x599 reference image.
# A little padding is included so template matching also sees its border.
PLAY_BOX = (0.585, 0.817, 0.915, 0.960)  # left, top, right, bottom
MATCH_THRESHOLD = 0.72
CHECK_INTERVAL = 0.5
CLICK_COOLDOWN = 1.0
GAME_CLICK_INTERVAL = 0.5
ESC_HOLD_SECONDS = 0.5
COIN_PLAY_DELAY_SECONDS = 3.0

# Verify one action at a time. Increase this only after each step is confirmed.
# 1 = main S, 2 = D on Upgrade, 3 = A on Random Boost,
# 4 = D on Multi-Buy, 5 = S with Double Coins, 6 = Boost click,
# Step 7 is intentionally disabled in this box-only version.
# Step 8 = D on Result, 9 = D on Mystery Box.
VERIFY_THROUGH_STEP = 9

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
    mask = cv2.inRange(hsv, np.array([18, 60, 60]), np.array([95, 255, 255]))
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


def has_gameplay_heart_icon(frame: np.ndarray) -> bool:
    """Detect the pink energy-heart icon at the upper-left during a run."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    icon_area = hsv[
        round(height * 0.075):round(height * 0.215),
        round(width * 0.040):round(width * 0.120),
    ]

    # The icon combines a saturated pink heart with a large bright, nearly
    # white circular rim. Requiring both avoids confusing it with coins or UI.
    pink = cv2.inRange(icon_area, np.array([140, 65, 105]), np.array([179, 255, 255]))
    pale = cv2.inRange(icon_area, np.array([0, 0, 145]), np.array([179, 85, 255]))
    pink_ratio = cv2.countNonZero(pink) / float(max(pink.size, 1))
    pale_ratio = cv2.countNonZero(pale) / float(max(pale.size, 1))
    return pink_ratio > 0.035 and pale_ratio > 0.10


def has_fast_start_icon(frame: np.ndarray) -> bool:
    """Detect a Fast Start or Cookie Relay boost card near screen centre."""
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

    # Cookie Relay can appear over a green stage, causing its green card edge
    # to merge with the background. Both Relay and Fast Start cards still
    # contain a sizeable bright cyan/blue area in this tightly constrained
    # centre region, so use it as a fallback.
    centre = hsv[
        round(height * 0.34):round(height * 0.63),
        round(width * 0.41):round(width * 0.60),
    ]
    cyan = cv2.inRange(centre, np.array([78, 55, 55]), np.array([120, 255, 255]))
    cyan_ratio = cv2.countNonZero(cyan) / float(max(cyan.size, 1))
    if cyan_ratio > 0.025:
        return True

    # Relay's ninja artwork can be darker blue than the Fast Start artwork.
    # Accept one sizeable blue component in the same tightly bounded centre.
    blue = cv2.inRange(centre, np.array([95, 55, 35]), np.array([135, 255, 255]))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centre_height, centre_width = centre.shape[:2]
    for contour in contours:
        _, _, contour_width, contour_height = cv2.boundingRect(contour)
        if (
            contour_width > centre_width * 0.16
            and contour_height > centre_height * 0.16
            and cv2.contourArea(contour) > centre_width * centre_height * 0.012
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

    # The OK button can be lime/yellow-green (Hue around 20-30), depending on
    # the stage/theme. The previous lower bound of 32 missed this Result skin.
    green = cv2.inRange(hsv, np.array([18, 60, 60]), np.array([95, 255, 255]))
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

    if candidates:
        _, centre = max(candidates, key=lambda item: item[0])
        return True, centre

    # Fallback for gradient/outlined buttons whose green pixels do not form
    # one clean contour. On the Result screen, OK always occupies this lower-
    # left area; combine it with the pale-dialog check above to avoid matching
    # unrelated green UI elements.
    ok_area = green[
        round(height * 0.77):round(height * 0.95),
        round(width * 0.20):round(width * 0.50),
    ]
    green_ratio = cv2.countNonZero(ok_area) / float(max(ok_area.size, 1))
    if green_ratio >= 0.10:
        return True, (round(width * 0.36), round(height * 0.86))

    return False, (0, 0)


def find_open_all(frame: np.ndarray) -> tuple[bool, tuple[int, int]]:
    """Detect the Mystery Box screen and return the Open all button centre."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mystery Box has a predominantly dark blue background. Some stages use
    # brighter rays, so keep this as supporting evidence instead of requiring
    # the old, overly strict 42 percent ratio.
    dark = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 105]))
    dark_ratio = cv2.countNonZero(dark) / float(width * height)

    cyan = cv2.inRange(hsv, np.array([75, 60, 55]), np.array([112, 255, 255]))
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

    if candidates and dark_ratio >= 0.20:
        _, centre = max(candidates, key=lambda item: item[0])
        return True, centre

    # Fallback: Open all/Confirm is always a large cyan button in the tightly
    # constrained bottom-centre area. This remains reliable when the animated
    # Mystery Box background is too bright for the dark-background detector.
    button_area = cyan[
        round(height * 0.78):round(height * 0.98),
        round(width * 0.36):round(width * 0.64),
    ]
    cyan_ratio = cv2.countNonZero(button_area) / float(max(button_area.size, 1))
    if cyan_ratio >= 0.16 and dark_ratio >= 0.12:
        return True, (round(width * 0.50), round(height * 0.89))

    return False, (0, 0)



def coin_screen_visible(frame: np.ndarray) -> bool:
    """Return True on the pre-run boost shop screen."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # The screen has a large cyan/blue upgrade panel on the left and a wide
    # bright-green Play button at the lower right.
    left_panel = hsv[
        round(height * 0.15):round(height * 0.96),
        round(width * 0.08):round(width * 0.50),
    ]
    play_area = hsv[
        round(height * 0.76):round(height * 0.95),
        round(width * 0.52):round(width * 0.89),
    ]
    buy_area = hsv[
        round(height * 0.30):round(height * 0.48),
        round(width * 0.65):round(width * 0.84),
    ]
    cyan = cv2.inRange(left_panel, np.array([78, 55, 55]), np.array([112, 255, 255]))
    green = cv2.inRange(play_area, np.array([18, 60, 60]), np.array([95, 255, 255]))
    cyan_buy = cv2.inRange(buy_area, np.array([78, 75, 65]), np.array([112, 255, 255]))
    cyan_ratio = cv2.countNonZero(cyan) / float(max(cyan.size, 1))
    green_ratio = cv2.countNonZero(green) / float(max(green.size, 1))
    cyan_buy_ratio = cv2.countNonZero(cyan_buy) / float(max(cyan_buy.size, 1))
    return green_ratio > 0.12 and (cyan_ratio > 0.04 or cyan_buy_ratio > 0.12)


def home_play_screen_visible(frame: np.ndarray) -> bool:
    """Detect the main ranking/loadout screen shown before entering Boosts."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # This screen uniquely combines the cyan Pet/Cookie/Treasure bar with the
    # large green Play button directly underneath it on the right-hand side.
    loadout_area = hsv[
        round(height * 0.68):round(height * 0.84),
        round(width * 0.56):round(width * 0.93),
    ]
    play_area = hsv[
        round(height * 0.82):round(height * 0.98),
        round(width * 0.56):round(width * 0.93),
    ]
    cyan = cv2.inRange(loadout_area, np.array([78, 75, 65]), np.array([112, 255, 255]))
    green = cv2.inRange(play_area, np.array([18, 60, 60]), np.array([95, 255, 255]))
    cyan_ratio = cv2.countNonZero(cyan) / float(max(cyan.size, 1))
    green_ratio = cv2.countNonZero(green) / float(max(green.size, 1))
    # Text, highlights, and the button gradient reduce the solid-color ratios
    # considerably on the Plum Blossom Palace home skin. Requiring both
    # spatially separated controls still keeps this specific to the home page.
    return cyan_ratio > 0.08 and green_ratio > 0.10


def boost_upgrade_screen_visible(frame: np.ndarray) -> bool:
    """Detect the HP Upgrade screen before the Random Boost is selected."""
    if not coin_screen_visible(frame):
        return False

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    action_button = hsv[
        round(height * 0.32):round(height * 0.47),
        round(width * 0.67):round(width * 0.89),
    ]
    # Random Boost has a bright cyan Buy button here. HP Upgrade uses a much
    # darker teal button, which lets us stop repeating D as soon as it changes.
    bright_cyan = cv2.inRange(
        action_button, np.array([78, 75, 155]), np.array([112, 255, 255])
    )
    bright_ratio = cv2.countNonZero(bright_cyan) / float(max(bright_cyan.size, 1))
    return bright_ratio < 0.30


def random_boost_screen_visible(frame: np.ndarray) -> bool:
    """Detect Random Boost by its bright cyan Buy button."""
    if not coin_screen_visible(frame):
        return False

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    action_button = hsv[
        round(height * 0.32):round(height * 0.47),
        round(width * 0.67):round(width * 0.89),
    ]
    bright_cyan = cv2.inRange(
        action_button, np.array([78, 75, 155]), np.array([112, 255, 255])
    )
    bright_ratio = cv2.countNonZero(bright_cyan) / float(max(bright_cyan.size, 1))
    return bright_ratio >= 0.30


def multi_buy_dialog_visible(frame: np.ndarray) -> bool:
    """Detect the centred Pick desired Boosts / Multi-Buy dialog."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dialog = hsv[
        round(height * 0.08):round(height * 0.91),
        round(width * 0.16):round(width * 0.84),
    ]
    pale = cv2.inRange(dialog, np.array([0, 0, 135]), np.array([179, 95, 255]))
    pale_ratio = cv2.countNonZero(pale) / float(max(pale.size, 1))

    button = hsv[
        round(height * 0.75):round(height * 0.90),
        round(width * 0.38):round(width * 0.61),
    ]
    green = cv2.inRange(button, np.array([18, 60, 60]), np.array([95, 255, 255]))
    green_ratio = cv2.countNonZero(green) / float(max(green.size, 1))
    return pale_ratio > 0.50 and green_ratio > 0.16


def double_coins_visible(frame: np.ndarray) -> bool:
    """Detect the red Double Coins banner immediately above Play."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    banner = hsv[
        round(height * 0.69):round(height * 0.82),
        round(width * 0.53):round(width * 0.87),
    ]
    red_low = cv2.inRange(banner, np.array([0, 70, 55]), np.array([14, 255, 255]))
    red_high = cv2.inRange(banner, np.array([165, 70, 55]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red_low, red_high)
    red_ratio = cv2.countNonZero(red) / float(max(red.size, 1))
    return red_ratio > 0.18


def relative_point(
    region: tuple[int, int, int, int], x_ratio: float, y_ratio: float
) -> tuple[int, int]:
    """Convert a point relative to the selected game area to screen pixels."""
    return (
        region[0] + round(region[2] * x_ratio),
        region[1] + round(region[3] * y_ratio),
    )


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

    print("Watching for the Play screen. Hold ESC for 0.5 seconds to stop.")
    last_click = 0.0
    game_clicking = False
    mystery_opened = False
    fast_start_active = False
    coin_stage = "select_box"
    double_coins_ready_since: float | None = None
    upgrade_d_sent = False
    random_a_sent = False
    multi_d_sent = False
    result_d_sent = False
    mystery_d_sent = False
    mystery_confirm_d_sent = False
    esc_started: float | None = None
    while True:
        if keyboard.is_pressed("esc"):
            if esc_started is None:
                esc_started = time.monotonic()
            elif time.monotonic() - esc_started >= ESC_HOLD_SECONDS:
                print("ESC held for 0.5 seconds; stopping.")
                break
        else:
            esc_started = None

        frame = screenshot_bgr(region)
        score, centre, detector = find_play(frame, template)
        result_found, ok_centre = find_result_ok(frame)
        open_all_found, open_all_centre = find_open_all(frame)
        pause_found = has_pause_icon(frame)
        fast_start_found = False  # Centre-click Boost handling is disabled.
        home_play_screen = home_play_screen_visible(frame)
        boost_upgrade_screen = boost_upgrade_screen_visible(frame)
        random_boost_screen = random_boost_screen_visible(frame)
        coin_screen = coin_screen_visible(frame)
        multi_dialog = multi_buy_dialog_visible(frame)
        coins_ready = coin_screen and double_coins_visible(frame)
        now = time.monotonic()

        if not coins_ready:
            double_coins_ready_since = None

        if (
            VERIFY_THROUGH_STEP >= 8
            and result_found
            and now - last_click >= CLICK_COOLDOWN
        ):
            print("Result screen found; pressing D")
            pyautogui.press("d")
            last_click = now
        elif home_play_screen:
            game_clicking = False
            coin_stage = "select_box"
            double_coins_ready_since = None
            # A new main screen starts a new run. Rearm each one-shot menu key
            # here only, so a one-frame detector flicker cannot press it twice.
            upgrade_d_sent = False
            random_a_sent = False
            multi_d_sent = False
            result_d_sent = False
            mystery_d_sent = False
            mystery_confirm_d_sent = False
            if now - last_click >= CLICK_COOLDOWN:
                print("Main Play screen found; pressing S")
                pyautogui.press("s")
                last_click = now
        elif (
            VERIFY_THROUGH_STEP >= 2
            and boost_upgrade_screen
            and not upgrade_d_sent
            and now - last_click >= CLICK_COOLDOWN
        ):
            print("Boost selection screen found; pressing D")
            pyautogui.press("d")
            last_click = now
            upgrade_d_sent = True
        elif (
            VERIFY_THROUGH_STEP >= 5
            and coins_ready
        ):
            if double_coins_ready_since is None:
                double_coins_ready_since = now
                print("Double Coins found; waiting 3 seconds before pressing S")
            elif (
                now - double_coins_ready_since >= COIN_PLAY_DELAY_SECONDS
                and now - last_click >= CLICK_COOLDOWN
            ):
                print("Double Coins ready; pressing S")
                pyautogui.press("s")
                last_click = now
        elif (
            VERIFY_THROUGH_STEP >= 3
            and random_boost_screen
            and not random_a_sent
            and now - last_click >= CLICK_COOLDOWN
        ):
            print("Random Boost screen found; pressing A")
            pyautogui.press("a")
            last_click = now
            random_a_sent = True
        elif (
            VERIFY_THROUGH_STEP >= 4
            and multi_dialog
            and not multi_d_sent
            and now - last_click >= CLICK_COOLDOWN
        ):
            print("Multi-Buy dialog found; pressing D")
            pyautogui.press("d")
            last_click = now
            multi_d_sent = True
        elif (
            VERIFY_THROUGH_STEP >= 9
            and open_all_found
            and now - last_click >= CLICK_COOLDOWN
        ):
            print("Mystery Box found; pressing D")
            pyautogui.press("d")
            last_click = now
        elif VERIFY_THROUGH_STEP:
            # Step-by-step verification mode: intentionally do nothing after
            # the latest enabled step.
            pass
        elif multi_dialog:
            game_clicking = False
            if coin_stage == "multi_buy" and now - last_click >= CLICK_COOLDOWN:
                print("Multi-Buy dialog found; pressing D")
                pyautogui.press("d")
                last_click = now
                coin_stage = "waiting_for_double_coins"
        elif coin_screen:
            game_clicking = False
            if not coins_ready:
                double_coins_ready_since = None
            if coins_ready and coin_stage == "waiting_for_double_coins":
                if double_coins_ready_since is None:
                    double_coins_ready_since = now
                    print("Double Coins found; waiting 3 seconds before pressing S.")
                elif now - double_coins_ready_since >= COIN_PLAY_DELAY_SECONDS:
                    print("Double Coins ready; pressing S")
                    pyautogui.press("s")
                    last_click = now
                    coin_stage = "done"
                    double_coins_ready_since = None
            elif coin_stage == "select_box" and now - last_click >= CLICK_COOLDOWN:
                print("Boost selection screen found; pressing D")
                pyautogui.press("d")
                last_click = now
                coin_stage = "multi"
            elif coin_stage == "multi" and now - last_click >= CLICK_COOLDOWN:
                print("Random Boost screen found; pressing A")
                pyautogui.press("a")
                last_click = now
                coin_stage = "multi_buy"
            # While Multi-Buy rolls, deliberately do nothing until the
            # Double Coins banner appears. The "done" state also stays idle
            # here while the game is changing away from the boost screen.
        elif result_found:
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
            if fast_start_found:
                game_clicking = False
                if not fast_start_active:
                    screen_x = region[0] + region[2] // 2
                    screen_y = region[1] + region[3] // 2
                    print(f"Start/Relay Boost found; clicking centre at {screen_x}, {screen_y}")
                    pyautogui.click(screen_x, screen_y)
                    last_click = now
                fast_start_active = True
            elif pause_found and not game_clicking:
                fast_start_active = False
                coin_stage = "select_box"
                double_coins_ready_since = None
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

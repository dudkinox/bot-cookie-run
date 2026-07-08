"""
Cookie Run Auto Bot
Uses OpenCV for screen analysis and PyAutoGUI for automated clicks
"""

import cv2
import numpy as np
import pyautogui
import time
import threading
from PIL import ImageGrab

# Configuration
GAME_WINDOW_NAME = "Cookie Run"
CHECK_INTERVAL = 0.05  # Check every 50ms
ESCAPE_KEY_CODE = 27

# Color detection ranges (HSV)
# เปลี่ยนค่าเหล่านี้ตามสีของศัตรูในเกม
OBSTACLE_LOWER = np.array([0, 50, 50])
OBSTACLE_UPPER = np.array([10, 255, 255])

POWER_UP_LOWER = np.array([100, 100, 100])
POWER_UP_UPPER = np.array([120, 255, 255])

# Screen region for detection (x, y, width, height)
GAME_REGION = (0, 0, 800, 600)

class CookieRunBot:
    def __init__(self):
        self.running = False
        self.bot_thread = None
        
    def get_screen_region(self, x, y, width, height):
        """Capture a region of the screen"""
        screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return frame
    
    def detect_obstacles(self, frame):
        """Detect obstacles in the frame using color detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for obstacles
        mask = cv2.inRange(hsv, OBSTACLE_LOWER, OBSTACLE_UPPER)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                obstacles.append((x, y, w, h, area))
        
        return obstacles, mask
    
    def detect_power_ups(self, frame):
        """Detect power-ups in the frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, POWER_UP_LOWER, POWER_UP_UPPER)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        power_ups = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                power_ups.append((x, y, w, h, area))
        
        return power_ups, mask
    
    def make_decision(self, frame, obstacles, power_ups):
        """Make decision on where to move/jump"""
        h, w = frame.shape[:2]
        player_y = h - 50  # Approximate player Y position
        
        # Simple logic: jump if obstacle is close
        for ox, oy, ow, oh, _ in obstacles:
            if oy > player_y - 100 and ox < w * 0.6:  # Obstacle approaching
                return "jump"
        
        # Move towards power-ups if available
        if power_ups:
            px, py, pw, ph, _ = max(power_ups, key=lambda x: x[4])
            if px < w * 0.3:
                return "left"
            elif px > w * 0.7:
                return "right"
        
        return None
    
    def perform_action(self, action):
        """Perform the game action"""
        if action == "jump":
            # Simulate spacebar press (jump in Cookie Run)
            pyautogui.press('space')
            print("[BOT] JUMP!")
        elif action == "left":
            # Move left
            pyautogui.press('left')
            print("[BOT] MOVE LEFT")
        elif action == "right":
            # Move right
            pyautogui.press('right')
            print("[BOT] MOVE RIGHT")
    
    def bot_loop(self):
        """Main bot loop"""
        print("[BOT] Starting Cookie Run Bot... Press ESC to stop")
        time.sleep(2)  # Wait for game window to be active
        
        while self.running:
            try:
                # Capture screen
                frame = self.get_screen_region(*GAME_REGION)
                
                # Detect obstacles and power-ups
                obstacles, obs_mask = self.detect_obstacles(frame)
                power_ups, power_mask = self.detect_power_ups(frame)
                
                # Make decision
                action = self.make_decision(frame, obstacles, power_ups)
                
                # Perform action
                if action:
                    self.perform_action(action)
                
                # Optional: Display debug info
                # self.display_debug(frame, obstacles, power_ups)
                
                time.sleep(CHECK_INTERVAL)
                
            except Exception as e:
                print(f"[ERROR] {e}")
                break
    
    def start(self):
        """Start the bot"""
        if not self.running:
            self.running = True
            self.bot_thread = threading.Thread(target=self.bot_loop, daemon=True)
            self.bot_thread.start()
    
    def stop(self):
        """Stop the bot"""
        self.running = False
        if self.bot_thread:
            self.bot_thread.join(timeout=2)
        print("[BOT] Stopped")

def main():
    bot = CookieRunBot()
    bot.start()
    
    try:
        # Wait for ESC key to stop
        import keyboard
        while True:
            if keyboard.is_pressed('esc'):
                break
            time.sleep(0.1)
    except ImportError:
        print("Install 'keyboard' library: pip install keyboard")
        print("Running for 60 seconds...")
        time.sleep(60)
    finally:
        bot.stop()

if __name__ == "__main__":
    main()

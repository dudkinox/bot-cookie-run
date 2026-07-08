"""
Color Finder Tool
ใช้หาค่า HSV ที่ถูกต้องสำหรับศัตรูและ Power-ups ในเกม
"""

import cv2
import numpy as np
from PIL import ImageGrab
import threading

class ColorFinder:
    def __init__(self):
        self.h_min = 0
        self.s_min = 0
        self.v_min = 0
        self.h_max = 180
        self.s_max = 255
        self.v_max = 255
        self.frame = None
        self.running = True
        
    def trackbar_callback(self, value):
        """Trackbar callback - updates mask in real-time"""
        pass
    
    def capture_and_display(self):
        """Capture game screen and show with trackbars"""
        # Create window
        cv2.namedWindow('Color Finder - Adjust Sliders', cv2.WINDOW_NORMAL)
        
        # Create trackbars
        cv2.createTrackbar('H_MIN', 'Color Finder - Adjust Sliders', 0, 180, self.trackbar_callback)
        cv2.createTrackbar('H_MAX', 'Color Finder - Adjust Sliders', 180, 180, self.trackbar_callback)
        cv2.createTrackbar('S_MIN', 'Color Finder - Adjust Sliders', 0, 255, self.trackbar_callback)
        cv2.createTrackbar('S_MAX', 'Color Finder - Adjust Sliders', 255, 255, self.trackbar_callback)
        cv2.createTrackbar('V_MIN', 'Color Finder - Adjust Sliders', 0, 255, self.trackbar_callback)
        cv2.createTrackbar('V_MAX', 'Color Finder - Adjust Sliders', 255, 255, self.trackbar_callback)
        
        print("📸 กำลังจับภาพหน้าจอ... โปรดรอ")
        print("⏱️  คุณมี 3 วินาทีในการเตรียมเกม")
        
        import time
        time.sleep(3)
        
        while self.running:
            # Capture screen (800x600 from top-left)
            screenshot = ImageGrab.grab(bbox=(0, 0, 800, 600))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Get trackbar values
            h_min = cv2.getTrackbarPos('H_MIN', 'Color Finder - Adjust Sliders')
            h_max = cv2.getTrackbarPos('H_MAX', 'Color Finder - Adjust Sliders')
            s_min = cv2.getTrackbarPos('S_MIN', 'Color Finder - Adjust Sliders')
            s_max = cv2.getTrackbarPos('S_MAX', 'Color Finder - Adjust Sliders')
            v_min = cv2.getTrackbarPos('V_MIN', 'Color Finder - Adjust Sliders')
            v_max = cv2.getTrackbarPos('V_MAX', 'Color Finder - Adjust Sliders')
            
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Show results
            result = cv2.bitwise_and(frame, frame, mask=mask)
            combined = cv2.hconcat([frame, result])
            
            # Add text info
            info_text = f"Lower: ({h_min}, {s_min}, {v_min}) | Upper: ({h_max}, {s_max}, {v_max})"
            cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Color Finder - Adjust Sliders', combined)
            
            # Press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✅ ค่า HSV ที่พบ:")
                print(f"LOWER = np.array([{h_min}, {s_min}, {v_min}])")
                print(f"UPPER = np.array([{h_max}, {s_max}, {v_max}])")
                print("\nคัดลอกค่าเหล่านี้ไปวางใน cookie_run_bot.py")
                break
        
        cv2.destroyAllWindows()

def main():
    print("🎨 Color Finder for Cookie Run Bot")
    print("=" * 50)
    print("วิธีใช้:")
    print("1. เปิดเกม Cookie Run")
    print("2. รัน script นี้")
    print("3. ปรับ Sliders จนกว่าจะเห็นเฉพาะสี่ที่ต้องการ")
    print("4. กด 'Q' เพื่อจบการทำงาน")
    print("=" * 50)
    
    finder = ColorFinder()
    finder.capture_and_display()

if __name__ == "__main__":
    main()

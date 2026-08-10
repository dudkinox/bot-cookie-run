# Cookie Run Auto Bot ✨🎮

Auto-play bot สำหรับเกม Cookie Run โดยใช้ **OpenCV** + **PyAutoGUI**

## 🚀 วิธีติดตั้ง

### 1️⃣ ติดตั้ง Python
- ดาวน์โหลด Python 3.9+ จาก https://www.python.org/downloads/
- ✅ ตรวจสอบ "Add Python to PATH" ขณะติดตั้ง

### 2️⃣ ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ เรียกใช้ Bot
```bash
python cookie_run_bot.py
```

## 🎮 การใช้งาน

1. **เปิดเกม Cookie Run** บนหน้าจออพยพ
2. **รัน Bot**: `python cookie_run_bot.py`
3. **กด ESC** เพื่อหยุดการทำงาน

## ⚙️ การปรับแต่ง

### ปรับสีของศัตรูและ Power-ups
แก้ไขค่า HSV ใน `cookie_run_bot.py`:

```python
# ตรวจจับสี (เปลี่ยนตามเกม)
OBSTACLE_LOWER = np.array([0, 50, 50])      # เทียม: Red obstacles
OBSTACLE_UPPER = np.array([10, 255, 255])

POWER_UP_LOWER = np.array([100, 100, 100])  # เทียม: Blue power-ups
POWER_UP_UPPER = np.array([120, 255, 255])
```

### หา HSV Color Range ที่ถูกต้อง
```python
python color_finder.py
```

## 📋 ฟีเจอร์

✅ ตรวจจับศัตรูจากจอ  
✅ ตรวจจับ Power-ups  
✅ ข้ามผ่านอุปสรรคอัตโนมัติ  
✅ ควบคุมสปีด (CHECK_INTERVAL)  

## 🐛 Troubleshooting

**ปัญหา**: Bot ไม่ตอบสนอง
- ✅ ตรวจสอบว่าเกมเป็น active window
- ✅ ปรับค่า GAME_REGION ให้ตรงกับขนาดเกม

**ปัญหา**: ไม่ตรวจจับศัตรู
- ✅ รัน color_finder.py เพื่อหา HSV ที่ถูกต้อง
- ✅ ปรับ OBSTACLE_LOWER/UPPER

## 📝 License
MIT License - ใช้เพื่อการศึกษาเท่านั้น


$env:QT_QPA_PLATFORM_PLUGIN_PATH="$PWD\runtime\qt_plugins\platforms"
pyw -m meccha_chameleon_tools
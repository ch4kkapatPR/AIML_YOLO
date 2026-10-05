#  YOLO26 - Chopstick, Fork & Spoon Detection

โปรเจกต์ **Object Detection** สำหรับตรวจจับอุปกรณ์รับประทานอาหาร 3 ประเภท ได้แก่ **Chopstick, Fork และ Spoon** โดยใช้ YOLO26

ระบบรองรับการตรวจจับทั้งจาก **รูปภาพ** และ **Webcam แบบ Real-time** พร้อมมีโมเดลที่ผ่านการ Train แล้วเก็บไว้ในโฟลเดอร์ `Model/`

---

##  Project Overview

วัตถุประสงค์ของโปรเจกต์คือการพัฒนาโมเดล Computer Vision สำหรับตรวจจับอุปกรณ์รับประทานอาหารจากภาพ โดยแบ่งวัตถุออกเป็น 3 Class

| ID | Class |
|---:|---|
| 0 | Chopstick |
| 1 | Fork |
| 2 | Spoon |

---

##  Features

-  ตรวจจับ Chopstick, Fork และ Spoon
-  Object Detection ด้วย Bounding Box
-  ตรวจจับวัตถุจากรูปภาพ
-  ตรวจจับวัตถุแบบ Real-time ผ่าน Webcam
-  รองรับ Dataset ที่เตรียมจากการทำ Annotation
-  Train และ Validate ด้วย Ultralytics YOLO
-  มีโมเดลที่ Train แล้วใน `Model/best.pt`

---

#  Technology Stack

| Technology | ใช้สำหรับ |
|---|---|
| Python | พัฒนาโปรแกรม |
| YOLO26 | Object Detection |
| Ultralytics | Training และ Inference |
| OpenCV | ประมวลผลภาพและ Webcam |
| PyTorch | Deep Learning |
| Label Studio | ทำ Annotation / Bounding Box |

---

#  Project Structure

โครงสร้างไฟล์ปัจจุบันของโปรเจกต์:

```text
AIML.cam/
│
├── Model/
│   └── best.pt
│
├── dataset2/
│
├── .gitignore
│
├── 01-export_dataset.py
├── 02-train.py
├── 03-test-image.py
├── 05-test-camera.py
│
├── README.md
├── data.yml
├── requirements.txt
└── test_3.jpg
```

### รายละเอียดไฟล์

| File / Folder | Description |
|---|---|
| `Model/best.pt` | โมเดล YOLO ที่ Train แล้ว ใช้สำหรับตรวจจับ |
| `dataset2/` | Dataset ที่ใช้ในโปรเจกต์ |
| `01-export_dataset.py` | เตรียมและแปลง Dataset |
| `02-train.py` | ใช้สำหรับ Training Model |
| `03-test-image.py` | ทดสอบ Model กับรูปภาพ |
| `05-test-camera.py` | ตรวจจับวัตถุผ่าน Webcam แบบ Real-time |
| `data.yml` | กำหนด Dataset และ Class สำหรับ YOLO |
| `requirements.txt` | รายการ Python packages ที่จำเป็น |
| `test_3.jpg` | รูปภาพสำหรับใช้ทดสอบ |
| `.gitignore` | กำหนดไฟล์ที่ไม่ต้องการให้ Git ติดตาม |

---

#  Project Workflow

```text
                ┌──────────────────┐
                │   Prepare Image  │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │     Dataset      │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ 01-export_       │
                │ dataset.py       │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │     data.yml     │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │   02-train.py    │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │    Model/        │
                │    best.pt       │
                └───────┬──────────┘
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
    ┌─────────────────┐   ┌─────────────────┐
    │ 03-test-image.py│   │05-test-camera.py│
    │   Image Test    │   │  Webcam Test    │
    └─────────────────┘   └─────────────────┘
```

---

#  Installation

## 1. Clone Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd AIML.cam
```

> หาก Repository ถูก Clone มาแล้ว สามารถข้ามขั้นตอนนี้ได้

---

## 2. Create Virtual Environment

แนะนำให้สร้าง Virtual Environment เพื่อแยก Python packages ของโปรเจกต์ออกจากระบบหลัก

```bash
python -m venv env
```

### Windows PowerShell

```powershell
.\env\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

ติดตั้ง packages จาก `requirements.txt`

```bash
pip install -r requirements.txt
```

หรือสามารถติดตั้ง packages ที่จำเป็นด้วยคำสั่ง:

```bash
pip install ultralytics opencv-python torch torchvision
```

---

#  Dataset

Dataset ของโปรเจกต์อยู่ใน:

```text
dataset2/
```

และมีการกำหนดข้อมูลสำหรับ YOLO ผ่าน:

```text
data.yml
```

ตัวอย่างโครงสร้าง Dataset:

```text
dataset2/
├── images/
├── labels/
└── ...
```

> โครงสร้างภายใน `dataset2/` ให้ตรวจสอบตาม Dataset ที่ใช้งานจริงใน Repository

---

#  Training

ไฟล์สำหรับ Training คือ:

```text
02-train.py
```

รันด้วย:

```bash
python 02-train.py
```

โมเดลที่ได้จากการ Training สามารถนำมาใช้สำหรับ Inference โดยใช้ไฟล์:

```text
Model/best.pt
```

---

#  Test with Image

โปรเจกต์มีไฟล์:

```text
03-test-image.py
```

สำหรับทดสอบโมเดลกับรูปภาพ

รัน:

```bash
python 03-test-image.py
```

สามารถใช้รูป:

```text
test_3.jpg
```

เป็นตัวอย่างสำหรับการทดสอบได้

### Workflow

```text
test_3.jpg
      ↓
03-test-image.py
      ↓
Model/best.pt
      ↓
Object Detection
      ↓
ผลลัพธ์ Bounding Box
```

---

#  Real-time Webcam Detection

ไฟล์:

```text
05-test-camera.py
```

ใช้สำหรับเปิด Webcam และตรวจจับวัตถุแบบ Real-time

รัน:

```bash
python 05-test-camera.py
```

ระบบจะนำภาพจาก Webcam เข้า Model และแสดงผลการตรวจจับบนหน้าจอ

```text
Webcam
   ↓
OpenCV
   ↓
YOLO Model
   ↓
Object Detection
   ↓
Bounding Box + Class
```

กด:

```text
q
```

เพื่อออกจากโปรแกรม

---

#  Model

โมเดลที่ใช้สำหรับ Inference อยู่ที่:

```text
Model/best.pt
```

ไฟล์นี้เป็น Weight ที่ใช้สำหรับตรวจจับวัตถุใน:

- `03-test-image.py`
- `05-test-camera.py`

---

#  Detection Classes

ระบบสามารถตรวจจับวัตถุทั้งหมด 3 ประเภท:

```text
0 → Chopstick
1 → Fork
2 → Spoon
```

ตัวอย่างผลลัพธ์:

```text
┌─────────────────────────────┐
│                             │
│     ┌───────────────┐       │
│     │   Chopstick   │       │
│     │     0.94      │       │
│     └───────────────┘       │
│                             │
│             ┌─────────┐     │
│             │  Spoon  │     │
│             │   0.97  │     │
│             └─────────┘     │
│                             │
└─────────────────────────────┘
```

> ตัวเลข Confidence ในภาพด้านบนเป็นเพียงตัวอย่างรูปแบบการแสดงผล ไม่ใช่ค่าผลการทดสอบจริง

---

#  Model Performance

หากต้องการแสดงผลการประเมิน Model เช่น

- Precision
- Recall
- mAP50
- mAP50-95

ควรใช้ค่าที่ได้จากการ Validation ของ Model จริงใน Repository

ตัวอย่างรูปแบบ:

| Metric | Value |
|---|---:|
| Precision | - |
| Recall | - |
| mAP50 | - |
| mAP50-95 | - |

> ยังไม่ระบุค่าตัวเลขใน README นี้ เพราะจากโครงสร้างไฟล์ที่ให้มาไม่สามารถยืนยันค่าผล Validation ล่าสุดได้

---

#  Dataset Recommendations

เพื่อเพิ่มความสามารถในการตรวจจับ ควรมีภาพที่หลากหลาย เช่น

- มุมกล้องหลายมุม
- ระยะใกล้และระยะไกล
- พื้นหลังหลายรูปแบบ
- ระดับแสงแตกต่างกัน
- วัตถุหลายชิ้นในภาพเดียว
- ภาพจาก Webcam หรือสภาพแวดล้อมที่ใกล้เคียงกับการใช้งานจริง

โดยเฉพาะ **Chopstick** ซึ่งมีลักษณะบางและยาว อาจต้องใช้ Dataset ที่มีความหลากหลายเพื่อช่วยให้ Model เรียนรู้ลักษณะของวัตถุได้ดีขึ้น

---

#  Recommended Workflow

หากต้องการ Train Model ใหม่:

```bash
# 1. Prepare Dataset
python 01-export_dataset.py

# 2. Train Model
python 02-train.py

# 3. Test with Image
python 03-test-image.py

# 4. Test with Webcam
python 05-test-camera.py
```

---

#  Notes

- ตรวจสอบ Path ของ Dataset ก่อน Training
- ตรวจสอบ `data.yml` ให้ Class และ Path ถูกต้อง
- ตรวจสอบว่า `Model/best.pt` เป็น Model ที่ต้องการใช้งาน
- หากเปลี่ยน Model ให้แก้ Path ในไฟล์ Test
- ควรแยก Test Dataset ออกจาก Training Dataset หากต้องการประเมินความสามารถของ Model กับข้อมูลที่ไม่เคยเห็นมาก่อน

---

#  Project Summary

โปรเจกต์นี้เป็นระบบ **Object Detection ด้วย YOLO26** สำหรับตรวจจับอุปกรณ์รับประทานอาหาร ได้แก่

>  **Chopstick**  
>  **Fork**  
>  **Spoon**

โดยมีขั้นตอนตั้งแต่การเตรียม Dataset, Training Model, ทดสอบกับรูปภาพ และนำ Model ไปใช้งานกับ Webcam แบบ Real-time

โมเดลที่ผ่านการ Train แล้วถูกจัดเก็บไว้ที่:

```text
Model/best.pt
```

---

##  Main Components

```text
Dataset
   │
   ▼
Training
   │
   ▼
YOLO Model
   │
   ▼
best.pt
   │
   ├──────────────► Image Detection
   │                 03-test-image.py
   │
   └──────────────► Real-time Detection
                     05-test-camera.py
```

---

##  Project Status

**Status: Completed / Ready for Testing**

ระบบมีองค์ประกอบหลักสำหรับการ Train และนำ Model ไปทดสอบทั้งรูปภาพและ Webcam แล้ว


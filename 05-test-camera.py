import cv2
from ultralytics import YOLO
import os


def main():

    # -----------------------------------------
    # 1. Path ของโมเดล
    # -----------------------------------------
    model_path = r"E:\AIML.cam\runs\detect\train-6\weights\best.pt"

    print("Model path:", model_path)
    print("Model exists:", os.path.exists(model_path))

    # -----------------------------------------
    # 2. โหลดโมเดล
    # -----------------------------------------
    model = YOLO(model_path)

    print("\nClasses in model:")
    print(model.names)
    print()

    # -----------------------------------------
    # 3. เปิดกล้อง
    # -----------------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    # ตั้งความละเอียดกล้อง
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("กด 'q' เพื่อออกจากโปรแกรม")

    # -----------------------------------------
    # 4. Real-time Detection
    # -----------------------------------------
    while True:

        success, frame = cap.read()

        if not success:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        # Detect
        results = model.predict(
            source=frame,

            # ลดไว้ก่อนเพื่อ Debug
            conf=0.5,

            # เพิ่ม resolution ตอนเข้าโมเดล
            imgsz=640,

            device="cpu",
            verbose=False
        )

        result = results[0]

        # -----------------------------------------
        # แสดง Detection ใน Terminal
        # -----------------------------------------
        if result.boxes is not None and len(result.boxes) > 0:

            for box in result.boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[cls_id]

                print(
                    f"Detected: {class_name} "
                    f"| confidence: {confidence:.3f}"
                )

        # -----------------------------------------
        # วาด Bounding Box
        # -----------------------------------------
        annotated_frame = result.plot()

        cv2.imshow(
            "YOLO26 Real-time Detection",
            annotated_frame
        )

        # กด q เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
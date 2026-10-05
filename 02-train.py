from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="data.yml",
        epochs=250,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,

        # augmentation
        degrees=15,
        translate=0.1,
        scale=0.4,
        shear=3,
        perspective=0.0005,
        fliplr=0.5,

        mosaic=1.0,
        mixup=0.1,

        patience=40,
        plots=True
    )


if __name__ == "__main__":
    main()
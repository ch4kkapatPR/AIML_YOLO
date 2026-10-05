import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse


# =========================================================
# CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = Path(r"E:\AIML.cam\dataset\images")
OUTPUT_DIR = Path(r"E:\AIML.cam\dataset2")

TRAIN_SPLIT = 0.8
SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =========================================================
# FILE UTILITIES
# =========================================================
def find_json_file(folder: Path) -> Path:
    """หาไฟล์ JSON export จาก Label Studio."""

    json_files = sorted(folder.glob("*.json"))

    if not json_files:
        sys.exit(
            f"ERROR: ไม่พบไฟล์ JSON ใน {folder}\n"
            "ให้นำไฟล์ JSON ที่ export จาก Label Studio "
            "มาไว้โฟลเดอร์เดียวกับ script"
        )

    if len(json_files) > 1:
        print(f"WARNING: พบ JSON หลายไฟล์ ใช้ไฟล์: {json_files[0].name}")

        for file in json_files:
            print(f"  - {file.name}")

    return json_files[0]


def build_image_index(images_dir: Path) -> dict[str, list[Path]]:
    """
    สร้าง index รูปทั้งหมดแบบ recursive.

    Example:
        IMG001.jpg -> [E:/.../IMG001.jpg]
    """

    image_index = {}

    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_index.setdefault(path.name, []).append(path)

    return image_index


def remove_label_studio_prefix(filename: str) -> str:
    """
    ตัด prefix ที่ Label Studio เพิ่มตอน upload.

    Example:
        9b24222a-IMG001.jpg
        -> IMG001.jpg
    """

    return re.sub(
        r"^[0-9a-fA-F]{8}-",
        "",
        filename,
    )


def find_matching_image(
    filename: str,
    image_index: dict[str, list[Path]],
) -> tuple[list[Path], str]:
    """ค้นหารูปจริงจาก filename ที่ได้จาก Label Studio."""

    # ลองชื่อเดิมก่อน
    matches = image_index.get(filename, [])

    if matches:
        return matches, filename

    # ลองตัด Label Studio prefix
    clean_filename = remove_label_studio_prefix(filename)
    matches = image_index.get(clean_filename, [])

    if matches:
        return matches, clean_filename

    return [], filename


# =========================================================
# LABEL STUDIO
# =========================================================
def get_image_filename(task: dict) -> str | None:
    """ดึงชื่อไฟล์รูปจาก Label Studio task."""

    data = task.get("data", {})
    image_value = None

    # Key ที่ใช้บ่อย
    for key in ("image", "img", "picture", "photo"):
        value = data.get(key)

        if isinstance(value, str):
            image_value = value
            break

    # ค้นหาเพิ่มเติม
    if image_value is None:
        for key, value in data.items():
            if not isinstance(value, str):
                continue

            if (
                "image" in key.lower()
                or "/data/local-files" in value
                or "/data/upload/" in value
            ):
                image_value = value
                break

    if image_value is None:
        return None

    parsed = urlparse(image_value)

    # Local Files:
    # /data/local-files/?d=dataset/images/IMG001.jpg
    if "d=" in parsed.query:
        relative_path = unquote(
            parsed.query.split("d=", 1)[1]
        )

        relative_path = relative_path.replace("\\", "/")

        return Path(relative_path).name

    # Upload:
    # /data/upload/5/9b24222a-IMG001.jpg
    return unquote(Path(parsed.path).name)


def collect_classes(tasks: list[dict]) -> list[str]:
    """รวบรวมชื่อ class ทั้งหมดจาก annotation."""

    classes = set()

    for task in tasks:
        for annotation in task.get("annotations", []):
            if annotation.get("was_cancelled"):
                continue

            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue

                labels = result.get(
                    "value",
                    {},
                ).get(
                    "rectanglelabels",
                    [],
                )

                classes.update(labels)

    return sorted(classes)


def convert_to_yolo(
    task: dict,
    class_to_id: dict[str, int],
) -> list[str]:
    """
    แปลง Label Studio bounding box -> YOLO.

    YOLO:
        class_id center_x center_y width height
    """

    lines = []

    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue

        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue

            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])

            if not labels:
                continue

            x = value["x"]
            y = value["y"]
            width = value["width"]
            height = value["height"]

            # Label Studio: percent + top-left
            # YOLO: normalized + center
            center_x = (x + width / 2) / 100
            center_y = (y + height / 2) / 100
            width /= 100
            height /= 100

            for label in labels:
                class_id = class_to_id[label]

                lines.append(
                    f"{class_id} "
                    f"{center_x:.6f} "
                    f"{center_y:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

    return lines


# =========================================================
# DATASET
# =========================================================
def prepare_output_directory() -> None:
    """ลบ dataset2 เก่า แล้วสร้าง folder ใหม่."""

    if OUTPUT_DIR.exists():
        print(f"\nRemoving old dataset: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_tasks(
    tasks: list[dict],
    image_index: dict[str, list[Path]],
):
    """ตรวจสอบว่า task มีรูปและ annotation ครบ."""

    valid_tasks = []

    stats = {
        "missing_file": 0,
        "missing_filename": 0,
        "no_annotation": 0,
    }

    print("\nChecking Label Studio tasks...")

    for task in tasks:
        filename = get_image_filename(task)

        if filename is None:
            stats["missing_filename"] += 1

            print(
                f"NO IMAGE PATH: "
                f"{task.get('data', {})}"
            )

            continue

        matches, matched_filename = find_matching_image(
            filename,
            image_index,
        )

        if not matches:
            stats["missing_file"] += 1

            print(
                f"NOT FOUND: {filename}\n"
                f"  data: {task.get('data', {})}"
            )

            continue

        if len(matches) > 1:
            print(
                f"WARNING: Duplicate filename: "
                f"{matched_filename}"
            )

            for path in matches:
                print(f"  - {path}")

        if not task.get("annotations"):
            stats["no_annotation"] += 1
            print(f"NO ANNOTATION: {matched_filename}")
            continue

        # ใช้ชื่อไฟล์จริงบน disk
        valid_tasks.append(
            (
                task,
                matched_filename,
                matches[0],
            )
        )

    return valid_tasks, stats


def process_split(
    tasks,
    split_name: str,
    classes: list[str],
    class_to_id: dict[str, int],
):
    """สร้าง images/labels สำหรับ train หรือ val."""

    image_count = 0
    instance_count = 0
    class_counts = Counter()

    for task, filename, source_image in tasks:
        yolo_lines = convert_to_yolo(
            task,
            class_to_id,
        )

        if not yolo_lines:
            continue

        # -------------------------
        # Copy image
        # -------------------------
        destination_image = (
            OUTPUT_DIR
            / "images"
            / split_name
            / filename
        )

        shutil.copy2(
            source_image,
            destination_image,
        )

        # -------------------------
        # Save label
        # -------------------------
        label_path = (
            OUTPUT_DIR
            / "labels"
            / split_name
            / Path(filename).with_suffix(".txt")
        )

        label_path.write_text(
            "\n".join(yolo_lines) + "\n",
            encoding="utf-8",
        )

        # -------------------------
        # Statistics
        # -------------------------
        image_count += 1
        instance_count += len(yolo_lines)

        for line in yolo_lines:
            class_id = int(line.split()[0])
            class_name = classes[class_id]

            class_counts[class_name] += 1

    return image_count, instance_count, class_counts


def write_dataset_files(classes: list[str]) -> None:
    """สร้าง classes.txt และ data.yaml."""

    classes_file = OUTPUT_DIR / "classes.txt"

    classes_file.write_text(
        "\n".join(classes) + "\n",
        encoding="utf-8",
    )

    data_yaml = (
        f"path: {OUTPUT_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )

    (OUTPUT_DIR / "data.yaml").write_text(
        data_yaml,
        encoding="utf-8",
    )


def print_class_counts(
    class_counts: Counter,
    classes: list[str],
) -> None:
    """แสดงจำนวน instance แต่ละ class."""

    for class_name in classes:
        print(
            f"  {class_name:<12}: "
            f"{class_counts[class_name]}"
        )


# =========================================================
# MAIN
# =========================================================
def main():
    random.seed(SEED)

    # -------------------------
    # Check source
    # -------------------------
    if not IMAGES_DIR.exists():
        sys.exit(
            f"ERROR: ไม่พบโฟลเดอร์รูป:\n{IMAGES_DIR}"
        )

    # -------------------------
    # JSON
    # -------------------------
    json_path = find_json_file(SCRIPT_DIR)

    print(f"Using JSON file: {json_path.name}")

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        tasks = json.load(file)

    if not isinstance(tasks, list):
        sys.exit(
            "ERROR: JSON format ไม่ถูกต้อง "
            "(ต้องเป็น list ของ Label Studio tasks)"
        )

    # -------------------------
    # Images
    # -------------------------
    print(f"\nScanning images from:\n{IMAGES_DIR}")

    image_index = build_image_index(IMAGES_DIR)

    total_source_images = sum(
        len(paths)
        for paths in image_index.values()
    )

    print(f"Found {total_source_images} source image files")
    print(f"Label Studio tasks: {len(tasks)}")

    # -------------------------
    # Classes
    # -------------------------
    classes = collect_classes(tasks)

    if not classes:
        sys.exit(
            "ERROR: ไม่พบ rectanglelabels ใน JSON"
        )

    class_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(classes)
    }

    print(
        f"Found {len(classes)} classes: "
        f"{classes}"
    )

    # -------------------------
    # Validate
    # -------------------------
    valid_tasks, skipped = validate_tasks(
        tasks,
        image_index,
    )

    if not valid_tasks:
        sys.exit("ERROR: ไม่มี task ที่สามารถใช้งานได้")

    # -------------------------
    # Prepare output
    # -------------------------
    prepare_output_directory()

    # -------------------------
    # Split
    # -------------------------
    random.shuffle(valid_tasks)

    split_index = int(
        len(valid_tasks) * TRAIN_SPLIT
    )

    train_tasks = valid_tasks[:split_index]
    val_tasks = valid_tasks[split_index:]

    # -------------------------
    # Export train
    # -------------------------
    (
        train_images,
        train_instances,
        train_counts,
    ) = process_split(
        train_tasks,
        "train",
        classes,
        class_to_id,
    )

    # -------------------------
    # Export validation
    # -------------------------
    (
        val_images,
        val_instances,
        val_counts,
    ) = process_split(
        val_tasks,
        "val",
        classes,
        class_to_id,
    )

    if train_images == 0:
        sys.exit("ERROR: ไม่มี train images")

    # -------------------------
    # Config files
    # -------------------------
    write_dataset_files(classes)

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 44)
    print("              EXPORT SUMMARY")
    print("=" * 44)

    print(f"Total Label Studio tasks : {len(tasks)}")
    print(f"Source image files       : {total_source_images}")
    print(f"Valid tasks              : {len(valid_tasks)}")
    print(f"Missing source image     : {skipped['missing_file']}")
    print(f"Missing image path       : {skipped['missing_filename']}")
    print(f"No annotation            : {skipped['no_annotation']}")

    print("-" * 44)

    print(f"Train images             : {train_images}")
    print(f"Train instances          : {train_instances}")
    print_class_counts(train_counts, classes)

    print("-" * 44)

    print(f"Val images               : {val_images}")
    print(f"Val instances            : {val_instances}")
    print_class_counts(val_counts, classes)

    print("-" * 44)

    print(
        f"Total exported images    : "
        f"{train_images + val_images}"
    )

    print(
        f"Total exported instances : "
        f"{train_instances + val_instances}"
    )

    print("=" * 44)

    print(
        f"\ndata.yaml written to:\n"
        f"{OUTPUT_DIR / 'data.yaml'}"
    )

    print("\nReady to train:")

    print(
        f'yolo detect train '
        f'data="{OUTPUT_DIR / "data.yaml"}" '
        f'model=yolo26n.pt '
        f'epochs=250 '
        f'imgsz=640 '
        f'batch=4'
    )


if __name__ == "__main__":
    main()
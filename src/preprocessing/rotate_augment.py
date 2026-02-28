import os
import glob
from pathlib import Path

import cv2

classes = ["cacahuatl", "teocomitl", "tepetla", "tlapexohuiloni"]

src_img_root = "source_images_bmp"
src_lbl_root = "source_labels_yolo"

angles = [90, 180, 270]


def rotate_image(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Angle must be 90, 180 or 270")


def rotate_box_yolo(x, y, w, h, angle):
    if angle == 90:
        x_new = 1.0 - y
        y_new = x
        w_new = h
        h_new = w
    elif angle == 180:
        x_new = 1.0 - x
        y_new = 1.0 - y
        w_new = w
        h_new = h
    elif angle == 270:
        x_new = y
        y_new = 1.0 - x
        w_new = h
        h_new = w
    else:
        raise ValueError("Angle must be 90, 180 or 270")

    return x_new, y_new, w_new, h_new


def process_class(cls_name):
    img_dir = os.path.join(src_img_root, cls_name)
    lbl_dir = os.path.join(src_lbl_root, cls_name)
    os.makedirs(lbl_dir, exist_ok=True)

    patterns = ["*.bmp", "*.jpg", "*.jpeg", "*.png"]
    img_files = []
    for p in patterns:
        img_files.extend(glob.glob(os.path.join(img_dir, p)))

    for img_path in img_files:
        stem = Path(img_path).stem

        if "_r90" in stem or "_r180" in stem or "_r270" in stem:
            continue

        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_path):
            print(f"[WARN] Pas de label pour {img_path}, je saute.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Impossible de lire {img_path}, je saute.")
            continue

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        parsed_boxes = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            parsed_boxes.append((cls_id, x, y, w, h))

        if not parsed_boxes:
            print(f"[INFO] Aucun bbox dans {lbl_path}, je saute l'augmentation.")
            continue

        ext = Path(img_path).suffix

        for angle in angles:
            rot_img = rotate_image(img, angle)
            out_stem = f"{stem}_r{angle}"
            out_img_path = os.path.join(img_dir, out_stem + ext)
            out_lbl_path = os.path.join(lbl_dir, out_stem + ".txt")

            new_lines = []
            for cls_id, x, y, w, h in parsed_boxes:
                x_n, y_n, w_n, h_n = rotate_box_yolo(x, y, w, h, angle)
                x_n = min(max(x_n, 0.0), 1.0)
                y_n = min(max(y_n, 0.0), 1.0)
                w_n = min(max(w_n, 0.0), 1.0)
                h_n = min(max(h_n, 0.0), 1.0)
                new_lines.append(f"{cls_id} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}")

            cv2.imwrite(out_img_path, rot_img)
            with open(out_lbl_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

            print(f"[OK] {cls_name}: créé {out_img_path} et {out_lbl_path}")


def main():
    for cls in classes:
        print(f"\n=== Traitement de la classe: {cls} ===")
        process_class(cls)


if __name__ == "__main__":
    main()
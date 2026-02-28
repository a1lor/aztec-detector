import os
import shutil
import random
import glob
import pathlib

# Define your 4 classes
classes = ["cacahuatl", "teocomitl", "tepetla", "tlapexohuiloni"]

# Source folders
src_img = "source_images_bmp"
src_lbl = "source_labels_yolo"

# Output dataset folders
dst = "dataset"

# Split ratios: 70% train, 20% val, 10% test
splits = [("train", 0.7), ("val", 0.2), ("test", 0.1)]

# Make sure destination folders exist
for path in [
    f"{dst}/images/train",
    f"{dst}/images/val",
    f"{dst}/images/test",
    f"{dst}/labels/train",
    f"{dst}/labels/val",
    f"{dst}/labels/test",
]:
    os.makedirs(path, exist_ok=True)

# Collect all labeled images
all_items = []
for cls in classes:
    img_files = []
    for ext in ("*.bmp", "*.jpg", "*.jpeg", "*.png"):
        img_files += glob.glob(os.path.join(src_img, cls, ext))
    img_files.sort()
    for img_path in img_files:
        all_items.append((cls, img_path))

# Shuffle for random split
random.seed(42)
random.shuffle(all_items)

# Split indices
n = len(all_items)
n_train = int(splits[0][1] * n)
n_val = int(splits[1][1] * n)
train_items = all_items[:n_train]
val_items = all_items[n_train : n_train + n_val]
test_items = all_items[n_train + n_val :]

split_map = {"train": train_items, "val": val_items, "test": test_items}

# Copy images and labels
for split, items in split_map.items():
    for cls, img_src in items:
        stem = pathlib.Path(img_src).stem
        img_dst = os.path.join(dst, "images", split, os.path.basename(img_src))
        lbl_src = os.path.join(src_lbl, cls, stem + ".txt")
        lbl_dst = os.path.join(dst, "labels", split, stem + ".txt")

        # Copy image
        shutil.copy2(img_src, img_dst)

        # Copy label if exists, else create empty
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)
        else:
            open(lbl_dst, "w").close()

print("✅ Dataset split complete!")
print(f"Images and labels copied to: {os.path.abspath(dst)}")
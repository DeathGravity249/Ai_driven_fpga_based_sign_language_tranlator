# scripts/preprocess.py

import os
import argparse
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def process_image(img_path, save_path, size):
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
    img.save(save_path, quality=90)

def augment_and_save(img_path, out_dir, base_name, size):
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)

    # original
    img.save(os.path.join(out_dir, f"{base_name}.jpg"), quality=90)
    # flip
    img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(out_dir, f"{base_name}_f.jpg"), quality=90)
    # rotations
    for i, angle in enumerate([15, -15]):
        # Use expand=True to avoid cropping
        img.rotate(angle, expand=True).save(os.path.join(out_dir, f"{base_name}_r{i}.jpg"), quality=90)

def main(args):
    src = os.path.join("dataset", "raw")
    dst = os.path.join("dataset", "processed")
    train_dir = os.path.join(dst, "train")
    test_dir = os.path.join(dst, "test")

    ensure_dir(train_dir)
    ensure_dir(test_dir)

    classes = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    print("Found classes:", classes)

    for cls in classes:
        cls_src = os.path.join(src, cls)
        images = [
            os.path.join(cls_src, f)
            for f in os.listdir(cls_src)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not images:
            continue

        train_list, test_list = train_test_split(images, test_size=args.test_size, random_state=42, shuffle=True)

        cls_train = os.path.join(train_dir, cls)
        cls_test = os.path.join(test_dir, cls)
        ensure_dir(cls_train)
        ensure_dir(cls_test)

        # Process train images
        for i, img_path in enumerate(train_list):
            base = f"{i}_{os.path.splitext(os.path.basename(img_path))[0]}"
            if args.augment:
                augment_and_save(img_path, cls_train, base, args.img_size)
            else:
                process_image(img_path, os.path.join(cls_train, f"{base}.jpg"), args.img_size)

        # Process test images
        for i, img_path in enumerate(test_list):
            base = f"{i}_{os.path.splitext(os.path.basename(img_path))[0]}"
            process_image(img_path, os.path.join(cls_test, f"{base}.jpg"), args.img_size)

    print("Preprocessing done. Processed data is in:", os.path.join("dataset", "processed"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--augment", action="store_true", help="Apply augmentation on training set")
    args = parser.parse_args()
    main(args)

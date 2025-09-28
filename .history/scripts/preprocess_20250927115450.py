# scripts/preprocess.py
import os, argparse, random
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def process_image(in_path, out_path, size):
    img = Image.open(in_path).convert("RGB")
    img = ImageOps.fit(img, (size, size), Image.ANTIALIAS)
    img.save(out_path, quality=90)

def augment_and_save(img_path, out_dir, base_name, size):
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.fit(img, (size, size), Image.ANTIALIAS)
    # original
    img.save(os.path.join(out_dir, f"{base_name}.jpg"), quality=90)
    # flipped
    img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(out_dir, f"{base_name}_f.jpg"), quality=90)
    # small rotations
    for i,angle in enumerate([15, -15]):
        img.rotate(angle).save(os.path.join(out_dir, f"{base_name}_r{i}.jpg"), quality=90)

def main(args):
    src = os.path.join("dataset", "raw")
    dst = os.path.join("dataset", "processed")
    train_dir = os.path.join(dst, "train")
    test_dir  = os.path.join(dst, "test")
    ensure_dir(train_dir); ensure_dir(test_dir)

    classes = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    print("Found classes:", classes)
    for cls in classes:
        cls_src = os.path.join(src, cls)
        images = [os.path.join(cls_src, f) for f in os.listdir(cls_src) if f.lower().endswith((".png",".jpg",".jpeg"))]
        if not images:
            continue
        train_list, test_list = train_test_split(images, test_size=args.test_size, random_state=42, shuffle=True)
        # ensure class dirs
        ctrain = os.path.join(train_dir, cls); ensure_dir(ctrain)
        ctest  = os.path.join(test_dir, cls); ensure_dir(ctest)

        # saving
        for i, img_path in enumerate(train_list):
            base = f"{int(i)}_{os.path.splitext(os.path.basename(img_path))[0]}"
            if args.augment:
                augment_and_save(img_path, ctrain, base, args.img_size)
            else:
                process_image(img_path, os.path.join(ctrain, f"{base}.jpg"), args.img_size)

        for i, img_path in enumerate(test_list):
            base = f"{int(i)}_{os.path.splitext(os.path.basename(img_path))[0]}"
            process_image(img_path, os.path.join(ctest, f"{base}.jpg"), args.img_size)

    print("Preprocessing done. Check dataset/processed/")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--augment", action="store_true", help="create flips/rotations for training")
    args = p.parse_args()
    main(args)

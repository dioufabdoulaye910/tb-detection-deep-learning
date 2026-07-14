import os
import shutil
import random

base_dir = "data_raw"
output_dir = "data"

classes = ["Normal", "Tuberculosis"]

train_ratio = 0.8

for cls in classes:
    path = os.path.join(base_dir, cls)

    images = os.listdir(path)
    random.shuffle(images)

    split = int(len(images) * train_ratio)

    train_imgs = images[:split]
    val_imgs = images[split:]

    for img in train_imgs:
        dst = f"{output_dir}/train/{cls}"
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(path, img), os.path.join(dst, img))

    for img in val_imgs:
        dst = f"{output_dir}/val/{cls}"
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(path, img), os.path.join(dst, img))

print("Dataset split terminé ✔")
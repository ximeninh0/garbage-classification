import os
import shutil
import random

path = "grouped"
output_path = "yolo_data"

# Cria os diretórios de saída train/val
train_path = os.path.join(output_path, "train/")
val_path = os.path.join(output_path, "val/")
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(os.path.join(train_path, "images/"), exist_ok=True)
os.makedirs(os.path.join(train_path, "labels/"), exist_ok=True)
os.makedirs(os.path.join(val_path, "images/"), exist_ok=True)
os.makedirs(os.path.join(val_path, "labels/"), exist_ok=True)

train = 0.8

images_path = os.path.join(path, "images/")

images = os.listdir(images_path)
num_images = len(images)
num_train = int(num_images * train)

random.shuffle(images)
train_images = images[:num_train]
val_images = images[num_train:]

# Move as imagens e labels para os diretórios correspondentes
for img in train_images:
    shutil.copy(os.path.join(images_path, img), os.path.join(train_path, "images/", img))
    label_name = img.replace(".jpg", ".txt")
    if os.path.exists(os.path.join(path, "labels/", label_name)):
        shutil.copy(os.path.join(path, "labels/", label_name), os.path.join(train_path, "labels/", label_name))

for img in val_images:
    shutil.copy(os.path.join(images_path, img), os.path.join(val_path, "images/", img))
    label_name = img.replace(".jpg", ".txt")
    if os.path.exists(os.path.join(path, "labels/", label_name)):
        shutil.copy(os.path.join(path, "labels/", label_name), os.path.join(val_path, "labels/", label_name))


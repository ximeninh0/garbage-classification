import json
import os
import shutil
import cv2

def convert_json_to_yolo_string(data,img_width,img_height):
    shapes = data['shapes']
    lines = []
    for shape in shapes:
        x1, y1 = shape['points'][0]
        x2, y2 = shape['points'][1]

        # garantir valores positivos
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        # calcular centro e tamanhos
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width / 2
        y_center = ymin + height / 2

        # normalizar
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        # converter label para número
        label = shape['label']
        match label:
            case "paper":   label = 0
            case "glass":   label = 1
            case "plastic": label = 2
            case "metal":   label = 3

        # montar linha final
        line = f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        lines.append(line)
    return lines

def remove_sobras():
        # acessar as pastas e copiar todas as imagens e jsons para 'grouped'
    grp_labels_path = "grouped\labels"
    grp_images_path = "grouped\images"

    for file_name in os.listdir(grp_images_path):
        label_name = file_name.removesuffix(".jpg")
        label_name += ".json"
        if label_name not in os.listdir(grp_labels_path):
            print(f"{file_name} nao possui label, removendo...")
            os.remove(os.path.join(grp_images_path,file_name))

    for file_name in os.listdir(grp_labels_path):
        img_name = file_name.removesuffix(".json")
        img_name += ".jpg"
        if img_name not in os.listdir(grp_images_path):
            print(f"{file_name} nao possui imagem, removendo...")
            os.remove(os.path.join(grp_labels_path,file_name))

if __name__ == "__main__":
    # acessar as pastas e copiar todas as imagens e jsons para 'grouped'
    yolo_path = "yolo_data/"
    raw_data_path = "raw_data/"
    grouped_path = "grouped/"

    os.makedirs(yolo_path, exist_ok=True)
    os.makedirs(yolo_path + "train/" + "images/", exist_ok=True)
    os.makedirs(yolo_path + "train/" + "labels/", exist_ok=True)

    os.makedirs(yolo_path + "val/" + "images/", exist_ok=True)
    os.makedirs(yolo_path + "val/" + "labels/", exist_ok=True)

    os.makedirs(grouped_path + 'images/', exist_ok=True)
    os.makedirs(grouped_path + 'labels/', exist_ok=True)

    for raw_dir in os.listdir(raw_data_path): # metal
        raw_dir_path = os.path.join(raw_data_path,raw_dir) # metal

        if os.path.isdir(raw_dir_path):
            for raw_garbage_class_dir in os.listdir(raw_dir_path): # images | labels
                raw_dir_class_path = os.path.join(os.path.join(raw_dir_path, raw_garbage_class_dir))
                if os.path.isdir(raw_dir_class_path):
                    for file_name in os.listdir(raw_dir_class_path):
                            if file_name.endswith('.json'):
                                shutil.copy2(os.path.join(raw_dir_class_path, file_name), os.path.join(grouped_path + 'labels/', file_name))
                            else:
                                shutil.copy2(os.path.join(raw_dir_class_path, file_name), os.path.join(grouped_path + 'images/', file_name))

    remove_sobras()
    #transformar jsons em txt
    grp_label_path = grouped_path + "labels/"
    for file_name in os.listdir(grp_label_path):
        if not file_name.endswith('.json'):
            continue
        
        json_path = os.path.join(grp_label_path,file_name)

        file = open(json_path, "r")
        data = json.load(file)
        file.close()

        img_path = os.path.join(grouped_path + "images/", file_name.removesuffix(".json") + ".jpg")
        print(f"Processando {img_path}...")
        img = cv2.imread(img_path)
        img_height, img_width, _ = img.shape
        lines = convert_json_to_yolo_string(data,img_width,img_height)

        new_label_name = file_name.removesuffix(".json")
        new_label_name += '.txt'

        os.remove(json_path)

        with open(os.path.join(grp_label_path,new_label_name), "w") as label_file:
            for line in lines:
                label_file.write(line)

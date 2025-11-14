import os
from pathlib import Path
from PIL import Image

def rename_and_convert_images(folder_path, prefix):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Pasta inválida: {folder}")
        return

    count = 1
    for file in os.listdir(folder_path):
        number = str(count)
        os.rename(folder_path + "\\" + file, folder_path + "\\" + prefix + number + ".jpg")
        count += 1

    print(f"\n✅ {count - 1} imagens processadas.")

if __name__ == "__main__":
    folder = "data/garbage_classification_v3/plastic"
    prefix = input("Digite o prefixo (ex: leon ou mariaflor ou lixokk): ").strip()
    rename_and_convert_images(folder, prefix)
from PIL import Image
import os

# rename_convert_heic.py
import os
from pathlib import Path
from PIL import Image

# tenta registrar opener do pillow-heif (pacote: pillow-heif)
heif_available = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()  # agora PIL consegue abrir .heic/.heif
    heif_available = True
except Exception:
    try:
        # fallback usando pyheif
        import pyheif
        from PIL import Image
        heif_available = True  # ainda true, pois usaremos pyheif manualmente
    except Exception:
        heif_available = False

def convert_heif_with_pyheif(path):
    """Usa pyheif para ler e converter um HEIC em PIL Image"""
    import pyheif
    heif = pyheif.read(path)
    image = Image.frombytes(
        heif.mode,
        heif.size,
        heif.data,
        "raw",
        heif.mode,
        heif.stride,
    )
    return image

def rename_and_convert_images(folder_path, prefix):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Pasta inválida: {folder}")
        return

    count = 1
    for file in os.listdir(folder_path):
        # if p.is_dir():
        #     continue
        number = str(count)
        os.rename(folder_path + "\\" + file, folder_path + "\\" + prefix + number + ".jpg")
        count += 1
        # suffix = p.suffix.lower()
        # aceitar várias extensões comuns
        # if suffix not in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif', '.heic', '.heif', '.tiff'):
        #     continue

        # try:
        #     if suffix in ('.heic', '.heif'):
        #         if 'pillow_heif' in globals() or 'pillow_heif' in dir():
        #             # se registrou, PIL.open já funciona
        #             img = Image.open(p).convert("RGB")
        #         else:
        #             # tenta pyheif
        #             img = convert_heif_with_pyheif(str(p)).convert("RGB")
        #     else:
        #         img = Image.open(p).convert("RGB")

        #     new_name = f"{prefix}{count}.jpg"
        #     new_path = output / new_name
        #     img.save(new_path, "JPEG", quality=95)
        #     print(f"{p.name} -> {new_name}")
        # except Exception as e:
        #     print(f"⚠️ Erro ao processar {p.name}: {e}")

    print(f"\n✅ {count - 1} imagens processadas.")

# exemplo de uso:
# para a pasta "C:/imagens/leon", todas serão "leon1.jpg", "leon2.jpg", etc.
if __name__ == "__main__":
    folder = "data/garbage_classification_v3/plastic"
    prefix = input("Digite o prefixo (ex: leon ou mariaflor): ").strip()
    rename_and_convert_images(folder, prefix)
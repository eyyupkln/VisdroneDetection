import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def visdrone2yolo(dir_path):
    dir_path = Path(dir_path)
    (dir_path / 'labels').mkdir(parents=True, exist_ok=True)

    pbar = tqdm((dir_path / 'annotations').glob('*.txt'), desc=f'Converting {dir_path.name}')
    for f in pbar:
        # Resim boyutunu al
        img_path = (dir_path / 'images' / f.name).with_suffix('.jpg')
        if not img_path.exists():
            continue

        img_size = Image.open(img_path).size
        lines = []

        with open(f, 'r') as file:
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # 'ignored regions' sınıfını atla
                    continue
                cls = int(row[5]) - 1  # Sınıf ID'sini YOLO için 0'dan başlat

                # Bounding box dönüştürme (VisDrone -> YOLO)
                dw = 1. / img_size[0]
                dh = 1. / img_size[1]
                box = (
                    (int(row[0]) + int(row[2]) / 2) * dw,
                    (int(row[1]) + int(row[3]) / 2) * dh,
                    int(row[2]) * dw,
                    int(row[3]) * dh
                )
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")

        # Etiketi labels klasörüne yaz
        label_path = str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}')
        with open(label_path, 'w') as fl:
            fl.writelines(lines)


# Projedeki dataset yollarını buraya yaz
dataset_root = Path('datasets/VisDrone')
folders_to_convert = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']

for folder in folders_to_convert:
    visdrone2yolo(dataset_root / folder)
import os
import shutil
import zipfile

# ========================
# Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MASK_ZIP_PATH = os.path.join(BASE_DIR, 'mask_data.zip')
MASK_EXTRACT_PATH = os.path.join(BASE_DIR, '00_mask_dataset')
MASK_DATASET_ROOT = os.path.join(MASK_EXTRACT_PATH, 'FMD_DATASET')

# ========================
# Function: flatten folders
# ========================
def flatten_folder(main_path):
    if not os.path.exists(MASK_EXTRACT_PATH):
        with zipfile.ZipFile(MASK_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MASK_EXTRACT_PATH)
            print('ZIP extracted successfully')

    image_ext = (".jpg", ".jpeg", ".png")

    for root, dirs, files in os.walk(main_path):
        if root == main_path:
            continue

        for file in files:
            if not file.lower().endswith(image_ext):
                continue

            src_path = os.path.join(root, file)
            dst_path = os.path.join(main_path, file)

            if os.path.exists(dst_path):
                name, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(main_path, f"{name}_{i}{ext}")
                    i += 1

            shutil.move(src_path, dst_path)

    for root, dirs, files in os.walk(main_path, topdown=False):
        if root != main_path and not os.listdir(root):
            os.rmdir(root)

# ========================
# Run on all classes
# ========================
mask_paths = [
    os.path.join(MASK_DATASET_ROOT, 'incorrect_mask'),
    os.path.join(MASK_DATASET_ROOT, 'with_mask'),
    os.path.join(MASK_DATASET_ROOT, 'without_mask')
]

for path in mask_paths:
    print(f"Processing: {path}")
    flatten_folder(path)

print("Done ✅")

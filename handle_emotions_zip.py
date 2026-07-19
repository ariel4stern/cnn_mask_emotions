import zipfile
import os
import shutil

# ========================
# Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_PATH = os.path.join(BASE_DIR, 'emotions_data.zip')
EXTRACT_PATH = os.path.join(BASE_DIR, '00_emotions_dataset')

def remove_test_disgust(zip_path,extract_path):
    # ========================
    # Extract ZIP (if needed)
    # ========================
    if not os.path.exists(EXTRACT_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
            print('ZIP extracted successfully')

    test_path = os.path.join(EXTRACT_PATH,"test")
    shutil.rmtree(test_path)
    print('test removed successfully')

    disgust_path = os.path.join(EXTRACT_PATH,"train","disgust")
    shutil.rmtree(disgust_path)
    print('disgust removed successfully')

remove_test_disgust(ZIP_PATH,EXTRACT_PATH)



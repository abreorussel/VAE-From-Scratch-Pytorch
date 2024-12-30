import os
import cv2
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR , "train")
TEST_DIR = os.path.join(DATA_DIR , "test")
IMG_SHAPE = (28, 28)

def save_images(data, labels, output_dir):
    """
    Saves images from data array into respective digit directories.

    Args:
        data (np.ndarray): Image data array.
        labels (np.ndarray): Corresponding labels for the data.
        output_dir (str): Directory to save the images.
    """
    digit_count = defaultdict(int)
    for idx in tqdm(range(data.shape[0]), desc=f"Processing {output_dir}"):
        img = data[idx].reshape(IMG_SHAPE)
        digit_label = labels[idx]
        digit_dir = os.path.join(output_dir, str(digit_label))
        os.makedirs(digit_dir, exist_ok=True)
        
        filename = os.path.join(digit_dir, f"{digit_count[digit_label]}.png")
        digit_count[digit_label] += 1
        try:
            cv2.imwrite(filename, img)
        except Exception as e:
            print(f"Error saving {filename}: {e}")


train_df = pd.read_csv(os.path.join(DATA_DIR, "mnist_train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "mnist_test.csv"))

train_data = train_df.iloc[:, 1:].values
train_labels = train_df.iloc[:, 0].values
test_data = test_df.iloc[:, 1:].values
test_labels = test_df.iloc[:, 0].values

save_images(train_data, train_labels, TRAIN_DIR)
save_images(test_data, test_labels, TEST_DIR)
 
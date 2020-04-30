"""
Usage:

Reduce the size of raw images to 600 X 800 and split it to train, test, validation folders
"""

import glob
import random
import os

from PIL import Image
from tqdm import tqdm

'''
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../RawClicks', help="Directory with the dataset")
parser.add_argument('--output_dir', default='Data', help="Where to write the new data")

src_path = args.data_dir
dest_path = args.output_dir
'''
src_path = "../RawClicks"
dest_path = "../Data"
file_ext = "jpg"
target_size = (800, 600)


def resize_and_save(seq, filename, output_dir, size=target_size):
    """Resize the image contained in 'filename' and save it to the 'output_dir'"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize(target_size, Image.BILINEAR)
    # Rotate the image to maintain original orientation
    image = image.rotate(-90)
    new_fname = "{}.{}".format(str(seq), file_ext)
    small_fname = os.path.join(output_dir, new_fname)
    image.save(small_fname)


file_names = glob.glob(os.path.join(src_path, "*.{}".format(file_ext)))
print(os.path.join(src_path, "*.{}".format(file_ext)))
os.makedirs(dest_path, exist_ok=True)

print("{} files to resize from directory `{}` to target size:{}".format(
        len(file_names), src_path, target_size)
)

# Split the images into 80% train, 10% test and 10% validation
# Shuffle with a fixed seed so that the split is reproducible
random.seed(230)
file_names.sort()
random.shuffle(file_names)

split = int(0.8 * len(file_names))
split2 = int(0.9 * len(file_names))
train_filenames = file_names[:split]
val_filenames = file_names[split:split2]
test_filenames = file_names[split2:]

file_names = {'train': train_filenames,
                'val': val_filenames,
                'test': test_filenames}

if not os.path.exists(dest_path):
    os.mkdir(dest_path)
else:
    print("Warning: output dir {} already exists".format(dest_path))

# Preprocess train, val and test
for split in ['train', 'val', 'test']:
    output_dir_split = os.path.join(dest_path, '{}_cars'.format(split))
    if not os.path.exists(output_dir_split):
        os.mkdir(output_dir_split)
    else:
        print("Warning: dir {} already exists".format(output_dir_split))

    print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
    i=1
    for filename in tqdm(file_names[split]):
        resize_and_save(i, filename, output_dir_split, size=target_size)
        i+=1

print("\nResizing {} files Complete.\nSaved to directory: `{}`".format(i,dest_path))
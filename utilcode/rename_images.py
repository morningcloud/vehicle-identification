"""
Usage:

Reduce the size of raw images to 600 X 800 and split it to train, test, validation folders
"""
import os

src_path = "../Data/val_cars_bk"

print(os.path.join(src_path, "*"))

print("files to rename from directory `{}`".format(src_path)
)

# Preprocess train, val and test
fromseq = 1
destseq = 238
for fromseq in range(1,27):
    #os.rename(os.path.join(src_path, "{}.xml".format(fromseq)),os.path.join(src_path, "{}.xml".format(destseq)))
    os.rename(os.path.join(src_path, "{}.jpg".format(fromseq)),os.path.join(src_path, "{}.jpg".format(destseq)))
    destseq += 1

print("\Renaming {} files Complete.".format(src_path))
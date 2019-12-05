import os
import shutil
import glob

train_path = r"./datasets/train"
val_path   = r"./datasets/test"
test_path  = r"./datasets/val"

# for cnt, name in enumerate(glob.glob(train_path+"/*")):
#     print(cnt, name)

for cnt, name in enumerate(os.listdir(val_path)):
    print(cnt)
    if cnt % 2 == 0:
        shutil.move(os.path.join(val_path, name), os.path.join(test_path, name))
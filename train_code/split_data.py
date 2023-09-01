import os
import shutil
import random
from tqdm import tqdm


src_root = "../images"
save_root = "../dataset"

os.makedirs(save_root,exist_ok=True)

sets = ["train","valid"]

class_names = os.listdir(src_root)
for name in class_names:
    class_dir = os.path.join(src_root,name)
    files = os.listdir(class_dir)
    test_num = int(len(files)*0.2)
    random.shuffle(files)
    for i,file in tqdm(enumerate(files)):
        src_path = os.path.join(class_dir,file)
        if i<test_num:
            dst_cls_dir = os.path.join(save_root,"valid",name)
        else:
            dst_cls_dir = os.path.join(save_root, "train", name)
        os.makedirs(dst_cls_dir, exist_ok=True)
        dst_path = os.path.join(dst_cls_dir, file)

        shutil.copy(src_path,dst_path)



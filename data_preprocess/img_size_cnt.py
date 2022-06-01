from PIL import Image
import os
from tqdm import tqdm
from collections import defaultdict

img_folder_path = '/home/caoliwei/Dataset/visdrone/images'
img_names = os.listdir(img_folder_path)

imgsz_cnt_dict = defaultdict(int)
for img_name in tqdm(img_names):
    img = Image.open(os.path.join(img_folder_path, img_name))
    img_w, img_h = img.size
    imgsz_cnt_dict[(img_w, img_h)] += 1
print('imgsz_cnt_dict:', sorted(imgsz_cnt_dict.items(), key=lambda x:x[1], reverse=True))
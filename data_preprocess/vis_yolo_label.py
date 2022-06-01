import os
from PIL import Image
from tqdm import tqdm
import random
from utils import draw_rectangle
import matplotlib.pyplot as plt
import cv2

img_folder_path = '/home/caoliwei/Dataset/visdrone/images'
label_path = '/home/caoliwei/Dataset/visdrone/labels'

img_names = os.listdir(img_folder_path)
extract_num = 10
random.shuffle(img_names)
for img_name in img_names[:extract_num]:
    coordinates = []
    labels_text = []
    img_raw = cv2.imread(os.path.join(img_folder_path, img_name))
    img_h, img_w = img_raw.shape[:2]

    anns = open(os.path.join(label_path, img_name[:-4] + '.txt')).read().splitlines()
    for ann in anns:
        ann = ann.split(' ')
        cls_name = str(ann[0])
        labels_text.append(cls_name)
        ann = [float(i) for i in ann[1:]]

        # 1、求坐标
        coordinate = []
        coordinate.append((ann[0] - ann[2] / 2) * img_w )  # convert to actual img size, not img_info size
        coordinate.append((ann[1] - ann[3] / 2) * img_h )
        coordinate.append(ann[2] * img_w )
        coordinate.append(ann[3] * img_h )
        coordinates.append(coordinate)


    img = draw_rectangle(coordinates, labels_text, img_raw)
    plt.figure(figsize=(5, 5), dpi=200)
    plt.imshow(img[:, :, ::-1])
    plt.show()
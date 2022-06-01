import json
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

mode = 'train'
# mode = 'train_tiny'
# mode = 'test'
# mode = 'test_tiny'
# mode = 'val'

label_folder_path = '/home/caoliwei/Dataset/visdrone/labels'
img_txt_path = f'/home/caoliwei/Dataset/visdrone/{mode}.txt'
save_path = f'/home/caoliwei/Dataset/visdrone/{mode}.json'
img_paths = open(img_txt_path).read().splitlines()


id = 0
images = []
categories = []
annotations = []

category_dic = dict()
my_categories = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
for i in range(len(my_categories)):
    categories.append({
        # "supercategory": "none",
        # "id": i,  # in mmdetection v2, id start at 0
        "id": i+1,
        "name": my_categories[i]})

img_dic = dict()
for img_path in tqdm(img_paths):
    file_name = img_path.split('/')[-1][:-4]
    img_dic[file_name] = id
    img = Image.open(img_path)
    images.append({
        "id":id,
        "file_name":file_name+'.jpg',
        "height":img.size[1],
        "width":img.size[0]}
    )
    id += 1

annid = 0
for img_path in tqdm(img_paths):
    img = Image.open(img_path)
    img_w, img_h = img.size
    file_name = img_path.split('/')[-1][:-4]
    anns = open(os.path.join(label_folder_path, file_name+'.txt')).read().splitlines()
    for ann in anns:
        ann = ann.split(' ')
        x_ctr = float(ann[1]) * img_w
        y_ctr = float(ann[2]) * img_h
        w = float(ann[3]) * img_w
        h = float(ann[4]) * img_h
        x_min = x_ctr - w/2.0
        y_min = y_ctr - h/2.0
        x_max = x_min + w
        y_max = y_min + h
        category_id = int(ann[0])

        annotations.append({
            # "segmentation":[[points[0],points[1],points[2],points[3],points[4],points[5],points[6],points[7]]],
            #"segmentation": [],
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0,
            "image_id": img_dic[file_name],
            "bbox": [x_min, y_min, w, h],
            #"category_id": category_dic[category],
            #"category_id": category_dic[category] - 1,
            "category_id": category_id + 1,   # coco数据集中，cat_id从1开始；mmdetection和yolo format的数据集中，cat_id从0开始
            "id": int(annid)})
        annid += 1

jsonfile = {}
jsonfile['images'] = images
jsonfile['categories'] = categories
jsonfile['annotations'] = annotations
with open(save_path, 'w',encoding='utf-8') as f:
    json.dump(jsonfile, f, indent=4)


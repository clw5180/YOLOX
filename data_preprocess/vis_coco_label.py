#!/usr/bin/env python
# coding=UTF-8
import cv2
import os
import numpy as np
from tqdm import tqdm
import time
import json
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from utils import draw_rectangle

img_path = '/home/caoliwei/Dataset/visdrone/images'
annFile = '/home/caoliwei/Dataset/visdrone/test_tiny.json'

extract_num = 10

print('loading json:')
start = time.time()
data = json.load(open(annFile))
print('loading json done, time use %.3fs' % (time.time() - start ))

img_infos = data['images']
img_ids = []
imgid_imginfo_dict = {}
imgname_imgid_dict = {}
for img_info in tqdm(img_infos):
    img_id = img_info['id']
    imgid_imginfo_dict[img_id] = img_info
    file_name = img_info['file_name'].split('/')[-1]
    imgname_imgid_dict[file_name] = img_id
    img_ids.append(img_id)

annotations = data['annotations']
imgid_ann_dict = defaultdict(list)
for ann in tqdm(annotations):
    img_id = ann['image_id']
    imgid_ann_dict[img_id].append(ann)

img_names = os.listdir(img_path)
for i, img_id in enumerate(tqdm(img_ids[:extract_num])):
# for i, img_name in enumerate(tqdm(img_names[:extract_num])):
    #img_id = imgname_imgid_dict[img_name[:-12] + '.jpg'  ] # no '_resized'
    # if not '33349478' in img_name:
    #     continue
    # img_id = imgname_imgid_dict[img_name]
    img_info = imgid_imginfo_dict[img_id]
    image_name = img_info['file_name'].split('/')[-1]
    img_info_w = img_info['width']
    img_info_h = img_info['height']
    # print(image_name)
    anns = imgid_ann_dict[img_id]


    coordinates = []
    labels_text = []
    img_raw = cv2.imread(os.path.join(img_path, image_name))
    img_h, img_w = img_raw.shape[:2]

    for ann in anns:
        # 1、求坐标
        coordinate = []
        coordinate.append(ann['bbox'][0] / img_info_w * img_w )  # convert to actual img size, not img_info size
        coordinate.append(ann['bbox'][1] / img_info_h * img_h )
        coordinate.append(ann['bbox'][2] / img_info_w * img_w )
        coordinate.append(ann['bbox'][3] / img_info_h * img_h )
        coordinates.append(coordinate)

        cls_name = str(ann['category_id']  )
        labels_text.append(cls_name)

    img = draw_rectangle(coordinates, labels_text, img_raw)
    plt.figure(figsize=(5, 5), dpi=200)
    plt.imshow(img[:, :, ::-1])
    plt.show()

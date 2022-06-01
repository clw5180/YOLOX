# Visdrone-DET Data Splits
# Visdrone-DET training split comprises 6471 images.
# Visdrone-DET testing split comprises 548 images.
# Visdrone-DET validation split comprises 1580 images.
# Visdrone-DET test-dev split comprises 1610 images.
# Download from https://github.com/VisDrone/VisDrone-Dataset

import json
from tqdm import tqdm

label_path = '/home/caoliwei/Dataset/visdrone/train.txt'
label_path = '/home/caoliwei/Dataset/visdrone/test.txt'

file_paths = open(label_path).read().splitlines()

areas = []
for file_path in tqdm(file_paths):
    data = open(file_path).read().splitlines()

    for ann in data:
        ann = str(ann).split(',')
        w = float(ann[2])
        h = float(ann[3])
        areas.append(w*h)

cnt_small = 0
cnt_mid = 0
cnt_large = 0
for i in areas:
    if i > 0 and i <= 32*32:
        cnt_small+=1
    elif i > 32*32 and i <= 96*96:
        cnt_mid+=1
    else:
        cnt_large+=1
print(cnt_small, cnt_mid, cnt_large)

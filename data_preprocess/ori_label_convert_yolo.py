import os
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

img_folder_path = '/home/caoliwei/Dataset/visdrone/images'
label_path = '/home/caoliwei/Dataset/visdrone/labels_ori'
save_path = '/home/caoliwei/Dataset/visdrone/labels'
os.makedirs(save_path, exist_ok=True)

cls_id_cnt = defaultdict(int)
ori_label_names = os.listdir(label_path)
for ori_label_name in tqdm(ori_label_names):
    file_name = ori_label_name[:-4]
    img_name = file_name + '.jpg'
    w, h = Image.open(os.path.join(img_folder_path, img_name)).size

    data = open(os.path.join(label_path, ori_label_name)).read().splitlines()
    line_new = []
    for record in data:
        record = str(record).split(',')  # xmin,ymin,w,h,score,object_categry,truncation,occlusion
        cls_id = int(record[5])
        if cls_id == 0 or cls_id == 11:
            continue

        record[:4] = [float(i) for i in record[:4]]
        x_ctr_norm = (record[0] + record[2] / 2.) / w
        y_ctr_norm = (record[1] + record[3] / 2.) / h
        w_norm = record[2] / w
        h_norm = record[3] / h

        data_new_str = [str(x) for x in [cls_id-1, x_ctr_norm, y_ctr_norm, w_norm, h_norm]]
        data_new_str = ' '.join(data_new_str) + '\n'
        line_new.append(data_new_str)
        cls_id_cnt[cls_id-1] += 1

    with open(os.path.join(save_path, file_name + '.txt'), 'w') as f:
        f.writelines(line_new)

print('cls_id_cnt:', cls_id_cnt)








import cv2
import os
import numpy as np
import sys
sys.path.append(os.getcwd())
from lib.utils.plot import plot_3d_box,plot_2d_box,plot_bird_view

output_dir='./output/vis_test'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def calc_theta_ray(img, box_2d, proj_matrix):
    width = img.shape[1]
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = (box_2d[1][0] + box_2d[0][0]) / 2
    dx = center - (width / 2)

    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
    angle = angle * mult

    return angle

image_root='./data/object_detection/testing/image_2/'
calib_root='./data/object_detection/testing/calib/'
label_pred_root='./output/test/'

cls_dict={'Car':0,'Cyclist':1,'Pedestrian':2}

for pre_file in sorted(os.listdir(label_pred_root)):
    image_ind=pre_file.split('.')[0]

    calib_path=calib_root+image_ind+'.txt'
    with open(calib_path,'r') as f:
        for l in f.readlines():
            items=l.strip().split()
            if items[0]=='P2:':
                calib_values=[eval(item) for item in items[1:]]
                calib=np.array(calib_values).reshape(3,4)
                calib[:,-1]=np.array([0,0,0])
                break

    image_path=image_root+image_ind+'.png'
    img=cv2.imread(image_path)
    img1=img.copy()
    img2=img.copy()
    preds = []
    preds_3DB = []
    label_pred_path=label_pred_root+image_ind+'.txt'
    with open(label_pred_path,'r') as f:
        masks = np.zeros((img.shape), dtype=np.uint8)
        for l in f.readlines():
            items=l.strip().split()
            cls=items[0]
            cls_idx = cls_dict[cls]
            alpha=eval(items[3])
            box2d=[[eval(item) for item in items[4:6]],[eval(item) for item in items[6:8]]]
            dims=[eval(item) for item in items[8:11]]
            r_y=eval(items[-2])
            locs = [eval(item) for item in items[11:14]]
            locs[1]-=dims[0]/2
            preds.append([cls_idx, dims, locs, r_y])
            mask=plot_3d_box(img1, calib, r_y, dims, locs)
            masks+=mask
            plot_2d_box(img2, box2d)

        img1 = cv2.addWeighted(img1, 1, masks, 0.5, 0)

    bird_view=plot_bird_view(preds)
    bird_view=cv2.resize(bird_view, (img1.shape[0], img1.shape[0]))
    img1=np.concatenate([img1,bird_view],axis=1)

    cv2.imwrite(output_dir+'/{}.jpg'.format(image_ind), img1)
    cv2.imwrite(output_dir+'/{}_2d.jpg'.format(image_ind), img2)
from cProfile import label
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path
import numpy as np

img_path = Path('/home/hongyu/CMKD/data/kitti/training/image_2')
depth_path = Path('/home/hongyu/CMKD/data/kitti/training/depth_2')
label_path = Path('/home/hongyu/CMKD/data/kitti/training/label_2')


pass

idx = 0

ipath = img_path / (str(idx).zfill(6) + '.png')
i = plt.imread(ipath)

dpath = depth_path / (str(idx).zfill(6) + '.png')
d = plt.imread(dpath)

lpath = label_path/ (str(idx).zfill(6) + '.txt')
l = open(lpath, 'r').readlines()

box1 = l[0].split()[4:8]
box1 = [int(float(i)) for i in box1]

ir1 = i[box1[-3]:box1[-1],box1[-4]:box1[-2],:]
dr1 = d[box1[-3]:box1[-1],box1[-4]:box1[-2],:]
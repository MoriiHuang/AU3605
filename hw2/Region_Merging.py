# 用法: python3 Threshold_Segmentation.py --method (choose one from 1,2,3) --filepath (the dir path the Input mg from)
#  /usr/local/bin/python3.7 Region_Merging.py --filepath 噪声图像
from genericpath import isdir

import cv2
import os
from cv2 import Mat
import numpy as np
import argparse
from cv2 import sqrt
from math import pi

parser = argparse.ArgumentParser(description='Get The method and Input file name')

parser.add_argument('--filepath')
args=parser.parse_args()
InputPath=os.path.join(os.getcwd(),args.filepath)
if args.filepath=="噪声图像":
    OutputPath=os.path.join(os.getcwd(),"区域合并"+args.filepath+'Result')
else:
    OutputPath=os.path.join(os.getcwd(),args.filepath+'Result')
if not isdir(OutputPath):
    os.mkdir(OutputPath)
def save_img(filename:str,img:Mat):
    cv2.imwrite(filename,img)
def cv_show(name,img):
    cv2.startWindowThread()
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
class Point(object):
    def __init__(self , x , y):
        self.x = x
        self.y = y
    def getX(self):
        return self.x
    def getY(self):
        return self.y
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight, loc):
        self.mean_val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
        self.loc = loc
class Processor:
    def __init__(self,filename:str) -> None:
        self.img=cv2.imread(filename,0)
        self.img_copy=self.img.copy()
        self.total_max=self.img.max()
        self.total_min=self.img.min()
        self.total_mean=self.img.mean()
    def stop_split(self, img):
        std=img.std()
        if std<12 or img.max() - img.min() <= 20:
            return True
        return False
    def paint(self,location,meanval):
        first_point = location[0]
        second_point = location[1]
        for i in range(first_point.x, second_point.x):
            for j in range(first_point.y, second_point.y):
                    self.img_copy[i][j] = meanval
    def merge(self,img1,img2):
        mean=(img1.mean_val+img2.mean_val)//2
        first_point = img1.loc[0]
        second_point = img1.loc[1]
        for i in range(first_point.x, second_point.x):
            for j in range(first_point.y, second_point.y):
                    self.img_copy[i][j] = mean
        first_point = img2.loc[0]
        second_point = img2.loc[1]
        for i in range(first_point.x, second_point.x):
            for j in range(first_point.y, second_point.y):
                    self.img_copy[i][j] = mean
    def construct(self,img,loc):
        root = Node(None, False, None, None, None, None, loc)
        if img.shape[0]<4:
            root.isLeaf = True
            root.mean_val = img.mean()
            self.paint(loc, root.mean_val)
        elif self.stop_split(img):  # 判断是否继续分割
            root.isLeaf = True
            root.mean_val = img.mean()
            self.paint(loc, root.mean_val)
        else:  
            height = img.shape[0]
            width = img.shape[1]
            halfheight = height // 2
            halfwidth = width // 2
            root.isLeaf = False # 如果网格中有值不相等，这个节点就不是叶子节点
            # 自回归
            base_start_x = loc[0].x
            base_start_y = loc[0].y
            base_end_x = loc[1].x
            base_end_y = loc[1].y
            root.topLeft = self.construct(img[:halfheight, :halfwidth], [Point(base_start_x, base_start_y), Point(base_start_x + halfheight, base_start_y + halfwidth)])
            root.topRight = self.construct(img[:halfheight, halfwidth:], [Point(base_start_x, base_start_y + halfwidth), Point(base_start_x + halfheight, base_end_y)])
            root.bottomLeft = self.construct(img[halfheight:, :halfwidth], [Point(base_start_x + halfheight, base_start_y), Point(base_end_x, base_start_y + halfwidth)])
            root.bottomRight = self.construct(img[halfheight:, halfwidth:], [Point(base_start_x + halfheight, base_start_y + halfwidth), Point(base_end_x, base_end_y)])
            if (root.topLeft.isLeaf and root.topRight.isLeaf and np.abs(root.topRight.mean_val-root.topLeft.mean_val)<3):
                self.merge(root.topLeft,root.topRight)
            if (root.topLeft.isLeaf and root.bottomLeft.isLeaf and np.abs(root.bottomLeft.mean_val-root.topLeft.mean_val)<3):
                self.merge(root.topLeft,root.bottomLeft)
            if (root.bottomRight.isLeaf and root.bottomLeft.isLeaf and np.abs(root.bottomLeft.mean_val-root.bottomRight.mean_val)<3):
                self.merge(root.bottomRight,root.bottomLeft)
            if (root.bottomRight.isLeaf and root.topRight.isLeaf and np.abs(root.topRight.mean_val-root.bottomRight.mean_val)<3):
                self.merge(root.bottomRight,root.topRight)
        return root
    def process(self):
        height,width=self.img.shape
        init_locate = [Point(0, 0), Point(height, width)]
        self.construct(self.img,init_locate)
        return self.img_copy

files=os.listdir(InputPath)
imgs=[]
for file in files:
    if file.endswith('bmp') or file.endswith('jpg') or file.endswith('png') or file.endswith('tif'):
        imgs.append(os.path.join(InputPath,file))
        print(os.path.join(InputPath,file))
for img in imgs:
    processor=Processor(img)
    if (img.split('/')[-1].split('.')[1]=="tif"):
        fin_name=os.path.join(os.curdir,OutputPath,img.split('/')[-1].split('.')[0]+'.jpg')
    else:
        fin_name=os.path.join(os.curdir,OutputPath,img.split('/')[-1].split('.')[0]+'.'+img.split('/')[-1].split('.')[1])
    print(fin_name)
    save_img(fin_name,processor.process())


    
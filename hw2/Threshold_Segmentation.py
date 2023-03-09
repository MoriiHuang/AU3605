# 用法: python3 Threshold_Segmentation.py --method (choose one from 1,2,3) --filepath (the dir path the Input mg from)
# % /usr/local/bin/python3.7 Threshold_Segmentation.py --method 3  --filepath 阈值分割实验图像-简单图像
from genericpath import isdir
import cv2
import os
from cv2 import Mat
from matplotlib.pyplot import hist
import numpy as np
import scipy.signal
import argparse

mapping = {"1":"_otsu","2":"_adaptive_threshold"}

parser = argparse.ArgumentParser(description='Get The method and Input file name')

parser.add_argument('--method')
parser.add_argument('--filepath')

args=parser.parse_args()
print(args)

InputPath=os.path.join(os.getcwd(),args.filepath)
if args.filepath=="复杂图像":
    OutputPath=os.path.join(os.getcwd(),"阈值分割"+args.filepath+'Result')
elif args.filepath=="噪声图像":
    OutputPath=os.path.join(os.getcwd(),"阈值分割"+args.filepath+'Result')
else:
    OutputPath=os.path.join(os.getcwd(),args.filepath+'Result')
if not isdir(OutputPath):
    os.mkdir(OutputPath)
def save_img(filename:str,img:Mat):
    cv2.imwrite(filename,img)
class Processor:
    def __init__(self,filename:str,method:int,flag:bool=False) -> None:
        self.img=cv2.imread(filename,0)
        self.method=method
        self.flag=flag
    # 大津法
    def __otsu(self):
        # 调库
        if self.flag:
            ret,thresh=cv2.threshold(self.img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return thresh
        # 自己实现
        else:
            max_g=0
            suitable_th=0
            for threshold in range(0,256):
                bin_img = self.img>threshold
                bin_img_inv=self.img<=threshold
                fore_pix=np.sum(bin_img)
                back_pix=np.sum(bin_img_inv)
                if fore_pix==0 :
                    break
                if back_pix==0:
                    continue
                w0=float(fore_pix)/self.img.size
                u0=float(np.sum(self.img*bin_img))/fore_pix
                w1=float(back_pix)/self.img.size
                u1=float(np.sum(self.img*bin_img_inv))/back_pix
                u=float(np.sum(self.img))/self.img.size
                g=w0*(u0-u)**2+w1*(u1-u)**2
                if g>max_g:
                    max_g=g
                    suitable_th=threshold
            _,thresh=cv2.threshold(self.img,suitable_th,255,cv2.THRESH_BINARY)
            return thresh
    # 自适应阈值 set GuassKernel size=7
    def __adaptive_thresh(self):
        thresh = cv2.adaptiveThreshold(self.img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,37,6)
        return thresh
    def result(self):
        if self.method=='1':
            thresh=self.__otsu()
        else:
            thresh=self.__adaptive_thresh()
        return thresh
files=os.listdir(InputPath)
print(files)
imgs=[]
for file in files:
    if file.endswith('bmp') or file.endswith('jpg') or file.endswith('png'):
        imgs.append(os.path.join(InputPath,file))
        print(os.path.join(InputPath,file))
for img in imgs:
    processor=Processor(img,args.method)
    fin_name=os.path.join(os.curdir,OutputPath,img.split('/')[-1].split('.')[0]+mapping[args.method]+'.'+img.split('/')[-1].split('.')[1])
    print(fin_name)
    save_img(fin_name,processor.result())


        
            




            


    
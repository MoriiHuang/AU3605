# 用法: python3 Region_Growing.py --method (choose one from 1,2,3) --filepath (the dir path the Input mg from)
# % /usr/local/bin/python3.7 Region_Growing.py --filepath 噪声图像
from genericpath import isdir
import random 
import cv2
import os
from cv2 import Mat
import numpy as np
import argparse
from cv2 import sqrt
from math import pi
import scipy.signal
parser = argparse.ArgumentParser(description='Get input file name')

#参数:简单图1(6),花（6），大象(4），犀牛（5),篮子(6）,南瓜（4,50）,细胞（8）

parser.add_argument('--filepath')
args=parser.parse_args()


InputPath=os.path.join(os.getcwd(),args.filepath)
if args.filepath=="噪声图像":
    OutputPath=os.path.join(os.getcwd(),"区域生长"+args.filepath+'Result')
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
class Processor:
    def __init__(self,filename:str,thresh:int,distance:int=30) -> None:
        self.img=cv2.imread(filename,0)
        height,width=self.img.shape
        if (height>512):
            self.img=cv2.resize(self.img,(512,int(512*height/width)))
        self.thresh=thresh
        self.distance=distance
    def selectConnect(self,loc,flag:bool=True):
        x,y=loc[0],loc[1]
        if flag:
            connects=[(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]
        else:
            connects=[(x-1,y),(x,y-1),(x+1,y),(x,y+1)]
        return connects
    def getSeed(self):
        hist=cv2.calcHist([self.img],[0],None,[256],[0,256])
        peaks,_ = scipy.signal.find_peaks(hist.squeeze(),distance=self.distance)
        print(peaks)
        gray=0
        duration=int(255/(len(peaks)-1))
        result=[]
        for peak in peaks:
            seeds=np.column_stack(np.where(self.img==peak))
            for i in range(len(seeds)):
                # s=random.randint(0,len(seeds)-1)
                result.append([seeds[i],float(gray/256)])
            gray+=duration
        return result
    def getGrayDiff(self,currentloc,connectedloc):
        return np.abs(int(self.img[currentloc[0],currentloc[1]])-int(self.img[connectedloc[0],connectedloc[1]]))
    def RegionGrow(self):
        height,width=self.img.shape
        seed_list=self.getSeed()
        seed_mask=np.zeros(self.img.shape)
        root_mask=np.zeros(self.img.shape)
        is_visited=np.zeros(self.img.shape)
        while(len(seed_list)>0):
            seed_tmp=seed_list[0][0]
            seed_gray=seed_list[0][1]
            seed_mask[seed_tmp[0],seed_tmp[1]]=seed_gray
            is_visited[seed_tmp[0],seed_tmp[1]]=1
            seed_list.pop(0)
            connects=self.selectConnect(seed_tmp)     
            for connect in connects:
                if connect[0]<0 or connect[1]<0 or connect[0]>=height or connect[1]>=width:
                    continue
                diff=self.getGrayDiff(seed_tmp,connect)
                if diff<self.thresh and is_visited[connect[0],connect[1]]==0:
                    seed_mask[connect[0],connect[1]]=seed_gray
                    root_mask[connect[0],connect[1]]=diff
                    is_visited[connect[0],connect[1]]=1
                    seed_list.append([connect,seed_gray])
                if  diff<self.thresh and is_visited[connect[0],connect[1]]==1:
                    if diff<root_mask[connect[0],connect[1]]:
                        root_mask[connect[0],connect[1]]=diff
                        seed_mask[connect[0],connect[1]]=seed_gray
                        seed_list.append([connect,seed_gray])
        # operation_kernel = np.ones((3,3), np.uint8)
        # iter_times = 2
        # result_image = cv2.morphologyEx(seed_mask, cv2.MORPH_CLOSE, operation_kernel,iterations=iter_times)

        return seed_mask*255.0
                    
# cv_show('RegionGrow',processor.RegionGrow())
files=os.listdir(InputPath)
imgs=[]
if args.filepath=="噪声图像":
    T=[4,4,4]
    peaks_distance=[30,30,30]
else:
    T=[6,8,4,6,6,5,5]
    peaks_distance=[30,20,20,30,50,20,20]

for file in files:
    if file.endswith('bmp') or file.endswith('jpg') or file.endswith('png') or file.endswith('tif'):
        imgs.append(os.path.join(InputPath,file))
        print(os.path.join(InputPath,file))
for img,t,d in list(zip(imgs,T,peaks_distance)):
    processor=Processor(img,t,d)
    if (img.split('/')[-1].split('.')[1]=="tif"):
        fin_name=os.path.join(os.curdir,OutputPath,img.split('/')[-1].split('.')[0]+'.jpg')
    else:
        fin_name=os.path.join(os.curdir,OutputPath,img.split('/')[-1].split('.')[0]+'.'+img.split('/')[-1].split('.')[1])
    print(fin_name)
    # cv_show('RegionGrow',processor.RegionGrow())
    save_img(fin_name,processor.RegionGrow())



        
            




            


    
import numpy as np
import cv2 as cv
import os
from genericpath import isdir
ImgPath=os.path.join(os.getcwd(),'血管抽取DRIVE-2/training/1st_manual')
OutputPath=os.path.join(os.getcwd(),'血管抽取DRIVE-2/training/Processed')
imgfiles=os.listdir(ImgPath)
imgs=[]
if not isdir(OutputPath):
    os.mkdir(OutputPath)
for file in imgfiles:
    if file.endswith('gif') or file.endswith('jpg') or file.endswith('png') :
        imgs.append(os.path.join(ImgPath,file))
for img in imgs:

    gif = cv.VideoCapture(img)
    ret, frame = gif.read()
    cv.imwrite(os.path.join(OutputPath,img.split('/')[10].split('.')[0]+'.jpg'), frame)
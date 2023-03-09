from ThresholdProcessor import Threshold_Processer
import matplotlib.pyplot as plt
import numpy as np
from genericpath import isdir
import cv2
import os
T=[40,100,160]
edge_imgs=[]
thresh_imgs=[]
if not isdir('Sensitivity_Analysis'):
    os.mkdir('Sensitivity_Analysis')
for t in T:
    tp=Threshold_Processer('PRLetter-images/1_gray.bmp',t,False,False)
    _,edge=cv2.threshold(tp.Gradient_Mag(),t,255,cv2.THRESH_BINARY_INV)
    edge_imgs.append(edge)
    ts=tp.select_Threshold()
    _,thresh=cv2.threshold(tp.gray,ts,255,cv2.THRESH_BINARY)
    thresh_imgs.append(thresh)
    tp.histogram('Sensitivity_Analysis/hist'+str(t)+'.jpg')
edge_fin=np.hstack(edge_imgs)
thresh_fin=np.hstack(thresh_imgs)
cv2.imwrite('Sensitivity_Analysis/edge.jpg',edge_fin)
cv2.imwrite('Sensitivity_Analysis/thresh.jpg',thresh_fin)

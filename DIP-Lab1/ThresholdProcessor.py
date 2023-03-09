from ast import Try
from genericpath import isdir
import cv2
from cv2 import imread
from cv2 import sqrt
import numpy as np
from sympy import false,true
import matplotlib.pyplot as plt
import os
import scipy.signal
InputPath=os.path.join(os.getcwd(),'PRLetter-images')
OutputPath=os.path.join(os.getcwd(),'Result')
if not isdir(OutputPath):
    os.mkdir(OutputPath)
# egde 代表图像边缘，EGDE代表正方形边缘
def load_img(filepath):
    img=imread(filepath)
    return img
def cv_show(name,img):
    cv2.startWindowThread()
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
def _interp(x,xp,fp):
    output = fp[0] + (x - int(xp[0])) * ((int(fp[1]) - int(fp[0]))/(int(xp[1]) -int(xp[0])))
    return output
class Threshold_Processer:
    def __init__(self,filepath,t,rev_flag,multi_flag) -> None:
        self.img=load_img(filepath)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.w,self.h=self.gray.shape
        self.T=t
        self.k=1
        self.flag=rev_flag
        self.mul_flag=multi_flag
    def Gradient_Mag(self):
        # Prewitt 算子计算梯度
        kernelx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
        kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
        x = cv2.filter2D(self.gray, cv2.CV_64F, kernelx)
        y = cv2.filter2D(self.gray, cv2.CV_64F, kernely)
        Grad_Mag = sqrt(x**2+y**2)
        return Grad_Mag
    def Laplacian(self,use_CV2_mol=false):
        if (use_CV2_mol):
            cv_show('lap',cv2.Laplacian(self.gray,cv2.CV_64F))
            return cv2.Laplacian(self.gray,cv2.CV_64F)
        else:
            lap_kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=int)
            lap=cv2.filter2D(self.gray,cv2.CV_64F,lap_kernel)
            return lap
    def GET_EDGE(self,loc):
        edge1=(loc,[loc[0]+self.k,loc[1]])
        edge2=(loc,[loc[0],loc[1]+self.k])
        edge3=([loc[0]+self.k,loc[1]],[loc[0]+1,loc[1]+self.k])
        edge4=([loc[0],loc[1]+self.k],[loc[0]+1,loc[1]+self.k])
        return [edge1,edge2,edge3,edge4]
    def check_EDGE(self,lap,grad,edges):
        count=0
        flag=false
        cross_edges=[]# 记录相交的edge
        for edge in edges:
            if (lap[edge[0][0],edge[0][1]]*lap[edge[1][0],edge[1][1]]<0
             and grad[edge[0][0],edge[0][1]]+grad[edge[0][0],edge[0][1]]>=2*self.T):
                count+=1
                cross_edges.append(edge)
        if (count>=2):
            flag=True
        return flag,cross_edges
    def _interpolation(self,edge,lap):
        start=lap[edge[0][0]][edge[0][1]]
        end=lap[edge[1][0]][edge[1][1]]
        inter_gray=_interp(0,[start,end],[self.gray[edge[0][0]][edge[0][1]],self.gray[edge[1][0]][edge[1][1]]])
        return inter_gray
    def edge_dectect(self):
        FIN_edge=[]
        FIN_EDGE_IMG=np.zeros((self.w,self.h))
        lap=self.Laplacian()
        grad=self.Gradient_Mag()
        for i in range(self.w-self.k):
            for j in range(self.h-self.k):
                edges=self.GET_EDGE([i,j]) 
                is_FIN_edge,cross_edges=self.check_EDGE(lap,grad,edges)
                if (is_FIN_edge):
                    for cross_edge in cross_edges:
                        FIN_edge.append(self._interpolation(cross_edge,lap))
                    FIN_EDGE_IMG[i][j]=self.gray[i][j]
        return FIN_edge,FIN_EDGE_IMG
    def histogram(self,filename):
        fin_edge,_=self.edge_dectect()
        fd=np.array(fin_edge).astype(np.float32)
        plt.figure(figsize=(33, 12), dpi=80)
        plt.hist(fd, 256*2, [0, 256])
        plt.savefig(filename)
        plt.close()
    def select_Threshold(self):
        fin_edge,_=self.edge_dectect()
        return np.average(fin_edge)
    def select_Threshold_mul(self):
        fin_edge,_=self.edge_dectect()
        hist,gray_scales=np.histogram(fin_edge,256*2,(0,255))
        peaks,_ = scipy.signal.find_peaks(hist,distance=150)
        threshes=[]
        for peak in peaks:
            threshes.append(gray_scales[peak])
        return threshes
    def process(self,filename):
        if self.mul_flag:
            threshes=self.select_Threshold_mul()
            res=[]
            back_groud_index=0
            for ts in threshes:
                _,thresh = cv2.threshold(self.gray,ts,255,cv2.THRESH_BINARY)
                if back_groud_index==1:
                    res.append(res[0]-thresh)
                back_groud_index+=1
                res.append(thresh)
            res=np.hstack(res)
            cv2.imwrite(filename,res)
        else:
            ts=self.select_Threshold()
            if (self.flag):
                ret,thresh = cv2.threshold(self.gray,ts,255,cv2.THRESH_BINARY_INV)
            else:
                ret,thresh = cv2.threshold(self.gray,ts,255,cv2.THRESH_BINARY)
            cv2.imwrite(filename,thresh)

if __name__ == '__main__':
    files=os.listdir(InputPath)
    imgs=[]
    T=[75,30,185,165,122,148,90]
    is_Inverse=[False,False,False,False,True,False,False]
    is_multi_thresh=[False,True,True,False,False,False,False]
    for file in files:
        if file.endswith('bmp') or file.endswith('jpg') or file.endswith('png'):
            imgs.append(os.path.join(os.curdir,'PRLetter-images',file))
    for img,t,flag,mul_flag in zip(imgs,T,is_Inverse,is_multi_thresh):
            print(img,t,flag,mul_flag)
            tp=Threshold_Processer(img,t,flag,mul_flag)
            outputname=os.path.join(os.curdir,'Result',img.split('/')[2])
            histname=os.path.join(os.curdir,'Result','hist_'+img.split('/')[2].split('.')[0]+'.jpg')
            tp.histogram(histname)
            tp.process(outputname)

        


       



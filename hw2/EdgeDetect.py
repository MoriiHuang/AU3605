# 用法: python3 Threshold_Segmentation.py --method (choose one from 1,2,3) --filepath (the dir path the Input mg from)
# % /usr/local/bin/python3.7 EdgeDetect.py --method 3  --filepath 边缘检测图像-简单-复杂图像
from genericpath import isdir
from re import S
import cv2
import os
from cv2 import Mat
import numpy as np
import argparse
from cv2 import sqrt
from math import pi
mapping = {"1":"_Robert","2":"_Prewitt","3":"_Sobel","4":"_Laplacian","5":"_LoG","6":"_Canny"}

parser = argparse.ArgumentParser(description='Get The method and Input file name')

parser.add_argument('--method')
parser.add_argument('--filepath')

args=parser.parse_args()
print(args)
InputPath=os.path.join(os.getcwd(),args.filepath)
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
# LoG 高梯度阈值
def High_Grad_Threshold_LoG(filename,k,T=40):
    img=cv2.imread(filename,0)
    img_grad_x= cv2.Sobel(img,cv2.CV_64F,1,0)
    img_grad_y= cv2.Sobel(img,cv2.CV_64F,0,1)
    img_grad=sqrt(img_grad_x**2+img_grad_y**2)
    mask=np.where(img_grad>T,1,0)
    img_gaussian = cv2.GaussianBlur(img, (k,k), 1)
    LoG = cv2.Laplacian(img_gaussian, cv2.CV_64F, ksize=k)
    LoG=LoG*mask   
    return LoG  
# LoG 单像素宽
def single_edge_LoG(filename,k,T=40):
    LoG=High_Grad_Threshold_LoG(filename,k,T)
    return np.where(LoG>0,LoG,0)
class Processor:
    def __init__(self,filename:str,method:int,k:int=5) -> None:
        self.img=cv2.imread(filename,0)
        self.method=method
        self.k=k
    def Robert(self,thresh=40):
        kernelx = np.array([[-1,0],[0,1]],dtype=int)
        kernely = np.array([[0,-1],[1,0]])
        x = cv2.filter2D(self.img, cv2.CV_64F, kernelx)
        y = cv2.filter2D(self.img, cv2.CV_64F, kernely)
        Robert = sqrt(x**2+y**2).astype(np.uint8)
        _,Rthresh=cv2.threshold(Robert,thresh,255,cv2.THRESH_BINARY)
        # 归一化
        return Rthresh
    def Prewitt(self,thresh=40):
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],dtype= int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],dtype= int)
        x = cv2.filter2D(self.img, cv2.CV_64F, kernelx)
        y = cv2.filter2D(self.img, cv2.CV_64F, kernely)
        Prewitt = sqrt(x**2+y**2).astype(np.uint8)
        _,Pthresh=cv2.threshold(Prewitt,thresh,255,cv2.THRESH_BINARY)
        return Pthresh
    def Sobel(self,thresh=40):
        kernelx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],dtype= int)
        kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],dtype= int)
        x = cv2.filter2D(self.img, cv2.CV_64F, kernelx)
        y = cv2.filter2D(self.img, cv2.CV_64F, kernely)
        Sobel = sqrt(x**2+y**2).astype(np.uint8)
        _,Sthresh=cv2.threshold(Sobel,thresh,255,cv2.THRESH_BINARY)
        return Sthresh
    def Laplacian(self,thresh=10):
        lap_kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=int)
        lap=cv2.filter2D(self.img,cv2.CV_64F,lap_kernel)
        _,lthresh=cv2.threshold(lap,thresh,255,cv2.THRESH_BINARY)
        return lthresh
    def LoG(self):
        img_gaussian = cv2.GaussianBlur(self.img, (self.k, self.k), 1)
        LoG = cv2.Laplacian(img_gaussian, cv2.CV_64F, ksize=self.k)
        # LoG_kernel=np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,0,-1,0,0],[0,-1,-2,-1,0]])
        # LoG=cv2.filter2D(self.img,cv2.CV_64F,LoG_kernel)
        return LoG
    def Gradient(self):
        gauss=cv2.GaussianBlur(self.img,(self.k,self.k),0)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]],dtype= int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],dtype= int)
        x = cv2.filter2D(gauss, cv2.CV_64F, kernelx)
        y = cv2.filter2D(gauss, cv2.CV_64F, kernely)
        np.seterr(divide='ignore', invalid='ignore')
        theta=np.arctan(y/x)*180/np.pi   
        isnan=np.isnan(theta)
        theta[isnan]=90
        Gradint=sqrt(x**2+y**2)
        return Gradint,theta
    def Canny(self,lowThresh=80,highThresh=120):
        gra,theta=self.Gradient()
        results=np.zeros(gra.shape)
        def NMS():
            for i in range(1,len(gra)-1):
                for j in range(1,len(gra[0])-1):
                    if gra[i,j]==0:
                        continue
                    else:
                        dTmp1 = 0
                        dTmp2 = 0
                        if theta[i,j] >= 0 and theta[i,j] < 45:
                            g1, g2, g3, g4 = gra[i+1,j-1], gra[i+1,j], gra[i-1,j+1], gra[i-1,j]
                            W = abs(np.tan(theta[i,j]*np.pi/180))
                            dTmp1 = W * g1 + (1-W) * g2
                            dTmp2 = W * g3 + (1-W) * g4
                        elif theta[i,j] >= 45 and theta[i,j] < 90:
                            g1, g2, g3, g4 = gra[i+1,j-1], gra[i, j-1], gra[i-1, j+1], gra[i, j+1]
                            W = abs(np.tan((theta[i,j]-90)*np.pi/180))
                            dTmp1 = W * g1 + (1-W) * g2
                            dTmp2 = W * g3 + (1-W) * g4
                        elif theta[i,j] >= -90 and theta[i,j] < -45:
                            g1, g2, g3, g4 = gra[i-1, j-1], gra[i, j-1], gra[i+1, j+1], gra[i, j+1]
                            W = abs(np.tan((theta[i,j]-90)*np.pi/180))
                            dTmp1 = W * g1 + (1-W) * g2
                            dTmp2 = W * g3 + (1-W) * g4
                        elif theta[i,j]>=-45 and theta[i,j]<0:
                            g1, g2, g3, g4 = gra[i+1, j+1], gra[i+1, j], gra[i-1, j-1], gra[i-1, j]
                            W = abs(np.tan(theta[i,j] * np.pi / 180))
                            dTmp1 = W * g1 + (1-W) * g2
                            dTmp2 = W * g3 + (1-W) * g4
                        if dTmp1 < gra[i,j] and dTmp2 < gra[i,j]:   
                                results[i,j]=gra[i,j]
            return results
        nms_results=NMS()
        def double_threshold():
            canny=np.zeros(nms_results.shape)
            for i in range(1,len(nms_results)-1):
                for j in range(1,len(nms_results[0])-1):
                    if nms_results[i,j]>highThresh:
                        canny[i,j]=255
                    elif nms_results[i,j]>lowThresh:
                        for n in [i-1,i,i+1]:
                            for m in [j-1,j,j+1]:
                                if nms_results[n,m]>highThresh:
                                    canny[i,j]=255
                                    break
            cv_show('Weak_Edge',canny.astype(np.uint8))
            return canny.astype(np.uint8)
        canny=double_threshold().astype(np.uint8)
        return canny   
    def canny_(self):
        return cv2.Canny(self.img,150,200)
    def default(self):
        assert("The method should be within 1-6")
    def result(self):
        # The time complexity of canny is bigger than others
        if (self.method=='6'):
            result=self.Canny()
        else:
            switcher={
                '1':self.Robert(),
                '2':self.Prewitt(),
                '3':self.Sobel(),
                '4':self.Laplacian(),
                '5':self.LoG(),
            }
            result=switcher.get(self.method,self.default)
        return result
# cv_show('canny',processor.Canny())
if __name__=="__main__":
    # LoG需要多尺度测试，故单独测试
    if (args.method=='5'):
        for k in [3,5,7,13]:
            processor=Processor("边缘检测图像-简单-复杂图像/多尺度测试图像2.jpg",args.method,k)
            save_img("边缘检测图像-简单-复杂图像Result/多尺度测试图像3_LoG_"+str(k)+".jpg",processor.result())
        for k in [3,5,7,13]:
            processor=Processor("边缘检测图像-简单-复杂图像/多尺度测试图像3.jpg",args.method,k)
            save_img("边缘检测图像-简单-复杂图像Result/多尺度测试图像3_LoG_"+str(k)+".jpg",processor.result())
        for k in [3,5,7,13]:
            LoG=High_Grad_Threshold_LoG("边缘检测图像-简单-复杂图像/多尺度测试图像2.jpg",k,90)
            save_img("边缘检测图像-简单-复杂图像Result/多尺度测试图像2_LoG_HighThresh_"+str(k)+".jpg",LoG)
        for k in [3,5,7,13]:
            LoG=single_edge_LoG("边缘检测图像-简单-复杂图像/多尺度测试图像2.jpg",k,90)
            save_img("边缘检测图像-简单-复杂图像Result/多尺度测试图像2_LoG_SingleEgde_HighThresh_"+str(k)+".jpg",LoG)
    else:
        files=os.listdir(InputPath)
        imgs=[]
        for file in files:
            if file.endswith('bmp') or file.endswith('jpg') or file.endswith('png'):
                imgs.append(os.path.join(os.curdir,'边缘检测图像-简单-复杂图像',file))
        for img in imgs:
            processor=Processor(img,args.method)
            try:
                fin_name=os.path.join(os.curdir,OutputPath,img.split('/')[2].split('.')[0]+mapping[args.method]+'.'+img.split('/')[2].split('.')[1])
                save_img(fin_name,processor.result())
            except:
                print("The method you input is illegal")
                break
  
            




            


    
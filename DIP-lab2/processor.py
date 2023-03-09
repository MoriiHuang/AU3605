import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import enum
import heapq
import os
from genericpath import isdir
INF=1e6
EPS=1e-6

class Area(enum.Enum):
    KNOWN=0
    BAND=1
    INSIDE=2
def cv_show(name,img):
    cv2.startWindowThread()
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def solve_eikonal(y1,x1,y2,x2,height,width,dists,flags):
    if y1<0 or y1>=height or x1<0 or x1>=width:
        return INF
    if y2<0 or y2>=height or x2<0 or x2>=width:
        return INF
    flag1=flags[y1,x1]
    flag2=flags[y2,x2]
    if flag1 == Area.KNOWN.value and flag2 == Area.KNOWN.value:
        dist1 = dists[y1, x1]
        dist2 = dists[y2, x2]
        d = 2.0 - (dist1 - dist2) ** 2
        if d > 0.0:
            r = sqrt(d)
            s = (dist1 + dist2 - r) / 2.0
            if s >= dist1 and s >= dist2:
                return s
            s += r
            if s >= dist1 and s >= dist2:
                return s
            # unsolvable
            return INF

    # only 1st pixel is known
    if flag1 == Area.KNOWN.value:
        dist1 = dists[y1, x1]
        return 1.0 + dist1

    # only 2d pixel is known
    if flag2 == Area.KNOWN.value:
        dist2 = dists[y2, x2]
        return 1.0 + dist2

    return INF
def compute_outside_dists(mask,height, width,radius):
    dists=np.full((height,width),INF,dtype=float)
    flags=mask.astype(int)*Area.INSIDE.value
    band = []
    mask_y, mask_x = mask.nonzero()
    for y, x in zip(mask_y, mask_x):
# look for BAND pixels in neighbors (top/bottom/left/right)
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for ny, nx in neighbors:
            # neighbor out of frame
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            # neighbor already flagged as BAND
            if flags[ny, nx] == Area.BAND.value:
                continue
            # neighbor out of mask => mask contour
            if mask[ny, nx] == 0:
                flags[ny, nx] = Area.BAND.value
                dists[ny, nx] = 0.0
                heapq.heappush(band, (0.0, ny, nx))
    band_ = band.copy()
    orig_flags = flags
    flags_ = orig_flags.copy()
    # swap INSIDE / OUTSIDE
    flags_[orig_flags == Area.KNOWN.value] = Area.INSIDE.value
    flags_[orig_flags == Area.INSIDE.value] = Area.KNOWN.value

    last_dist = 0.0
    double_radius = radius * 2
    while band_:
        # reached radius limit, stop FFM
        if last_dist >= double_radius:
            break

        # pop BAND pixel closest to initial mask contour and flag it as KNOWN
        _, y, x = heapq.heappop(band_)
        flags_[y, x] = Area.KNOWN.value

        # process immediate neighbors  (top/bottom/left/right)
        neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
        for nb_y, nb_x in neighbors:
            # skip out of frame
            if nb_y < 0 or nb_y >= height or nb_x < 0 or nb_x >= width:
                continue

            # neighbor already processed, nothing to do
            if flags_[nb_y, nb_x] !=  Area.INSIDE.value:
                continue

            # compute neighbor distance to inital mask contour
            last_dist = min([
                solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags),
                solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                solve_eikonal(nb_y - 1, nb_x, nb_y, nb_x + 1, height, width, dists, flags),
                solve_eikonal(nb_y + 1, nb_x, nb_y, nb_x - 1, height, width, dists, flags)
            ])

            dists[nb_y, nb_x] = last_dist

            # add neighbor to narrow band
            flags_[nb_y, nb_x] = Area.BAND.value
            heapq.heappush(band_, (last_dist, nb_y, nb_x))

    # distances are opposite to actual FFM propagation direction, fix it
    dists *= -1.0
    return dists, flags, band
class Processor:
    def __init__(self,filepath,maskpath,radius) -> None:
        self.img=cv2.imread(filepath)
        self.mask_img = cv2.imread(maskpath)
        self.mask=self.mask_img[:,:,0].astype(bool,copy=False)
        self.radius=radius
        self.h,self.w=self.img.shape[0:2]
        self.dists,self.flags,self.band=compute_outside_dists(self.mask,self.h,self.w,self.radius)
    def pixel_gradient(self,y, x):
        val = self.dists[y, x]
        # compute grad_y
        prev_y = y - 1
        next_y = y + 1
        if prev_y < 0 or next_y >= self.h:
            grad_y = INF
        else:
            flag_prev_y = self.flags[prev_y, x]
            flag_next_y = self.flags[next_y, x]

            if flag_prev_y != Area.INSIDE.value and flag_next_y != Area.INSIDE.value:
                grad_y = (self.dists[next_y, x] - self.dists[prev_y, x]) / 2.0
            elif flag_prev_y != Area.INSIDE.value:
                grad_y = val - self.dists[prev_y, x]
            elif flag_next_y != Area.INSIDE.value:
                grad_y = self.dists[next_y, x] - val
            else:
                grad_y = 0.0

        # compute grad_x
        prev_x = x - 1
        next_x = x + 1
        if prev_x < 0 or next_x >= self.w:
            grad_x = INF
        else:
            flag_prev_x = self.flags[y, prev_x]
            flag_next_x = self.flags[y, next_x]

            if flag_prev_x != Area.INSIDE.value and flag_next_x != Area.INSIDE.value:
                grad_x = (self.dists[y, next_x] - self.dists[y, prev_x]) / 2.0
            elif flag_prev_x != Area.INSIDE.value:
                grad_x = val - self.dists[y, prev_x]
            elif flag_next_x != Area.INSIDE.value:
                grad_x = self.dists[y, next_x] - val
            else:
                grad_x = 0.0

        return grad_y, grad_x
    def inpaint_pixel(self,y,x):
        dist=self.dists[y,x]
        dist_grad_y,dist_grad_x=self.pixel_gradient(y,x)
        pixel_sum=np.zeros((3),dtype=float)
        weight_sum=0.0
        for nb_y in range(y - self.radius, y + self.radius + 1):
        #  pixel out of frame
            if nb_y < 0 or nb_y >= self.h:
                continue
            for nb_x in range(x - self.radius, x + self.radius + 1):
                # pixel out of frame
                if nb_x < 0 or nb_x >= self.w:
                    continue
                # skip unknown pixels (including pixel being inpainted)
                if self.flags[nb_y, nb_x] == Area.INSIDE.value:
                    continue
                # vector from point to neighbor
                dir_y = y - nb_y
                dir_x = x - nb_x
                dir_length_square = dir_y ** 2 + dir_x ** 2
                dir_length = sqrt(dir_length_square)
                # pixel out of neighborhood
                if dir_length > self.radius:
                    continue
                # compute weight
                # neighbor has same direction gradient => contributes more
                dir_factor = abs(dir_y * dist_grad_y + dir_x * dist_grad_x)
                if dir_factor == 0.0:
                    dir_factor = EPS

                # neighbor has same contour distance => contributes more
                nb_dist = self.dists[nb_y, nb_x]
                level_factor = 1.0 / (1.0 + abs(nb_dist - dist))

                # neighbor is distant => contributes less
                dist_factor = 1.0 / (dir_length * dir_length_square)

                weight = abs(dir_factor * dist_factor * level_factor)

                pixel_sum[0] += weight * self.img[nb_y, nb_x, 0]
                pixel_sum[1] += weight * self.img[nb_y, nb_x, 1]
                pixel_sum[2] += weight * self.img[nb_y, nb_x, 2]

                weight_sum += weight

        return pixel_sum / weight_sum
    def inpant(self):
        while(self.band):
            _,y,x=heapq.heappop(self.band)
            self.flags[y,x]=Area.KNOWN.value
            neighbors = [(y - 1, x), (y, x - 1), (y + 1, x), (y, x + 1)]
            for ny, nx in neighbors:
                # pixel out of frame
                if ny < 0 or ny >= self.h or nx < 0 or nx >= self.w:
                    continue
                 # neighbor outside of initial mask or already processed, nothing to do
                if self.flags[ny, nx] != Area.INSIDE.value:
                    continue
                # compute neighbor distance to inital mask contour
                nb_dist = min([
                    solve_eikonal(ny-1,nx,ny,nx-1,self.h,self.w,self.dists,self.flags),
                    solve_eikonal(ny+1,nx,ny,nx+1,self.h,self.w,self.dists,self.flags),
                    solve_eikonal(ny-1,nx,ny,nx+1,self.h,self.w,self.dists,self.flags),
                    solve_eikonal(ny+1,nx,ny,nx-1,self.h,self.w,self.dists,self.flags)
                ])
                self.dists[ny, nx] = nb_dist

                # inpaint neighbor
                pixel_vals = self.inpaint_pixel(ny, nx)
                self.img[ny, nx, 0] = pixel_vals[0]
                self.img[ny, nx, 1] = pixel_vals[1]
                self.img[ny, nx, 2] = pixel_vals[2]

                # add neighbor to narrow band
                self.flags[ny, nx] = Area.BAND.value
                # push neighbor on band
                heapq.heappush(self.band, (nb_dist, ny, nx))
        return self.img
if __name__=="__main__":

    ImgPath=os.path.join(os.getcwd(),'results')
    MaskPath=os.path.join(os.getcwd(),'segmentation_result2')
    OutputPath=os.path.join(os.getcwd(),'血管抽取DRIVE-2/training/Results')
    if not isdir(OutputPath):
        os.mkdir(OutputPath)
    imgfiles=os.listdir(ImgPath)
    imgs=[]
    maskfiles=os.listdir(MaskPath)
    masks=[]
    for file in imgfiles:
        if file.endswith('tif') or file.endswith('jpg') or file.endswith('png') :
            imgs.append(os.path.join(ImgPath,file))
    for file in maskfiles:
        if file.endswith('tif') or file.endswith('jpg') or file.endswith('png') :
            masks.append(os.path.join(MaskPath,file))  
    imgs.sort()
    masks.sort()
    for img,mask in zip(imgs,masks):  
        print(img,mask)   
        Ps=Processor(img,mask,5)
        s=Ps.inpant()
        outputname=os.path.join(OutputPath,img.split('/')[8])
        print(outputname)
        cv2.imwrite(outputname,s)

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:36:21 2018

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import  cv2


impath='thumb/frm_lsa20187x3.mp4416.png'

'''cv2:BGR   plt RGB'''
img=cv2.imread(impath)

def  sigmoid(inX):
    try:
        sigout=1.0/(1.0+numpy.exp(-inX)+1e-6)
    except:
        print inx
        shp=inX.shape
        pass
            
    return sigout


HarrisParameter = [2,3,0.04]
HarrisThresh=0.01
 

def hist():
    histB=cv2.calcHist([img],[0],None,[256],[0,256])
    histG=cv2.calcHist([img],[1],None,[256],[0,256])
    histR=cv2.calcHist([img],[2],None,[256],[0,256])
    
    plt.imshow(img)


def getCorner(imgIn):
    gray = cv2.cvtColor(imgIn,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(img[:,:,0],HarrisParameter[0],HarrisParameter[1],HarrisParameter[2])

    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255]
    
    
    return dst



def strenthenDot(imgIn=img):
    dst = cv2.cornerHarris(img[:,:,0],HarrisParameter[0],HarrisParameter[1],HarrisParameter[2])
    dst = cv2.dilate(dst,None)
    conerMap=np.copy(imgIn[:,:,0])*0
    conerMap[np.where(dst>dst*dst.max())]=1
    
    
    return
   